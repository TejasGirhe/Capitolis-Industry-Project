"""
JointSimulator: one N-factor correlated Monte Carlo across every risk factor
in the book at once (rate + all equities + FX), per MARKET_DATA.md Sec.6
("collect the pairwise correlations among all factors...and simulate them
jointly") and the PFE/EE requirement for genuinely correlated scenario NPVs.

Factor ordering and correlation lookup reuse what already exists rather than
reinventing them: FactorSet.all() (risk_engine.factors.types) gives the
canonical factor order, and MarketState.correlation(a, b)
(capitolis_pricers.market) gives the pairwise lookup, keyed here by each
factor's str() (matching VolSurface.factor_key's convention: 'RATE_USD', an
ISIN, 'FX_USDJPY') so the same correlation dict a student populates for
MarketState.correlations also drives this simulation.

Each factor may need more than one state variable (a 2-factor rate model has
2; GBM/FXGBM have [log_growth, zeta] = 2 as well, but only the log_growth
column is actually a driven Brownian -- zeta is deterministic bookkeeping).
The joint correlated draw is built at "Brownian driver" granularity: one
driver per rate-model factor (LGM1F=1, LGM2F=2) and one driver per equity/FX
factor (GBM's diffusion is 1-dimensional regardless of SV, since the CIR
variance process draws its OWN independent normal inside each model's
simulate_paths -- only the level/rate Brownian is correlated across factors,
matching the standard convention that vol-of-vol shocks are idiosyncratic to
each factor rather than jointly correlated with everything).
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from ..models._shared import year_frac


@dataclass
class JointSimResult:
    factor_order: List[object]
    rate_states: Dict[object, np.ndarray]     # RateFactor -> (n_paths, n_dates, n_rate_state)
    spot_states: Dict[object, np.ndarray]     # EquityFactor/FxFactor -> (n_paths, n_dates, 2)
    rate_models: Dict[object, object]          # factor -> CalibratedRateModel
    spot_models: Dict[object, object]          # factor -> CalibratedSpotModel
    horizon_dates: List[object]                # the FIXED simulation grid dates (precache)
    sim_times: List[float]                      # year-fractions of horizon_dates from ref_date
    ref_date: object
    corr_chol: np.ndarray = None                # (n_drivers, n_drivers) Cholesky factor L, rho = L @ L.T --
                                                 # same correlation structure the grid-date draws used;
                                                 # reused by state_at(add_bridge_noise=True) so off-grid
                                                 # bridge noise is correlated across factors consistently.
    slot_ranges: Dict[object, tuple] = None     # factor -> (start, n_drivers) into corr_chol's axes
    drift_rate_factor_by_spot: Dict[object, object] = None     # spot factor -> its DOMESTIC drift RateFactor --
                                                                 # stored explicitly rather than inferred (the old
                                                                 # "if only one rate factor, it must be this one"
                                                                 # fallback in scenario_market.py breaks the moment
                                                                 # a second rate factor, e.g. a real JPY factor,
                                                                 # exists alongside USD).
    foreign_rate_factor_by_spot: Dict[object, object] = None   # FxFactor -> its OPTIONAL real foreign-currency
                                                                 # RateFactor (e.g. JPY), for spot models with a
                                                                 # genuinely simulated (not deterministic-implied)
                                                                 # foreign leg -- see models/fx.py, joint.py's
                                                                 # add_spot(foreign_rate_factor=...).

    def slice_paths(self, path_ids) -> "JointSimResult":
        """A new JointSimResult holding only the given paths' state arrays.

        Multiprocess pricing sends one JointSimResult PER WORKER over IPC;
        without slicing, every worker would receive a full copy of every
        path's state for every factor (an N_workers-fold over-transfer that
        hit a Windows named-pipe resource limit at 10,000 paths x 37+
        factors x 24 workers). Models (rate_models/spot_models) are small and
        shared as-is; only the (n_paths, n_dates, ...) arrays are sliced.
        corr_chol/slot_ranges are small (n_drivers x n_drivers) and copied
        as-is to every worker, same as the models dicts.
        """
        idx = np.asarray(path_ids)
        return JointSimResult(
            factor_order=self.factor_order,
            rate_states={f: arr[idx] for f, arr in self.rate_states.items()},
            spot_states={f: arr[idx] for f, arr in self.spot_states.items()},
            rate_models=self.rate_models,
            spot_models=self.spot_models,
            horizon_dates=self.horizon_dates,
            sim_times=self.sim_times,
            ref_date=self.ref_date,
            corr_chol=self.corr_chol,
            slot_ranges=self.slot_ranges,
            drift_rate_factor_by_spot=self.drift_rate_factor_by_spot,
            foreign_rate_factor_by_spot=self.foreign_rate_factor_by_spot,
        )


class JointSimulator:
    def __init__(self):
        self._rate_factors = []       # [(factor, calibrated_rate_model)]
        self._spot_factors = []       # [(factor, calibrated_spot_model, rate_factor_for_drift, foreign_rate_factor)]

    def add_rate(self, factor, calibrated_rate_model):
        self._rate_factors.append((factor, calibrated_rate_model))
        return self

    def add_spot(self, factor, calibrated_spot_model, drift_rate_factor, foreign_rate_factor=None):
        """drift_rate_factor: which rate factor (already added via add_rate)
        this spot model's drift is path-consistent with -- needed so
        build_market_state / level_at can be handed the matching rate_state.

        foreign_rate_factor: OPTIONAL second rate factor (also already
        added via add_rate) for an FX spot model with a REAL simulated
        foreign-currency rate leg (e.g. a CalibratedFXGBM whose foreign_
        rate_model is a genuine JPY LGM factor, not a deterministic
        implied curve -- see models/fx.py's module docstring). None
        (default) preserves the single-rate-factor behavior every equity
        factor (and the pre-real-JPY-data FX path) still uses."""
        self._spot_factors.append((factor, calibrated_spot_model, drift_rate_factor, foreign_rate_factor))
        return self

    def _driver_layout(self):
        """One Brownian driver slot per rate-model factor dimension, one per
        spot factor. Returns (factor_order, slot_ranges) where slot_ranges
        maps factor -> (start, n_drivers) into the joint z array's last axis."""
        factor_order, slot_ranges = [], {}
        cursor = 0
        for factor, model in self._rate_factors:
            n = model.n_factors
            slot_ranges[factor] = (cursor, n)
            factor_order.append(factor)
            cursor += n
        for factor, _, _, _ in self._spot_factors:
            slot_ranges[factor] = (cursor, 1)
            factor_order.append(factor)
            cursor += 1
        return factor_order, slot_ranges, cursor

    def _build_correlation_matrix(self, market_state, factor_order, slot_ranges, n_drivers):
        corr = np.eye(n_drivers)
        for i, fi in enumerate(factor_order):
            si, ni = slot_ranges[fi]
            for j, fj in enumerate(factor_order):
                if j <= i:
                    continue
                sj, nj = slot_ranges[fj]
                rho_ij = market_state.correlation(str(fi), str(fj))
                # single-driver-per-slot cross terms only (rate-model internal
                # factor correlation is handled inside that model's own rho,
                # not here); if a multi-factor rate model is cross-correlated
                # to another factor, apply rho_ij uniformly across its drivers
                # -- a documented simplification (no data source distinguishes
                # a 2F rate model's factor-1 vs factor-2 correlation to equity).
                for a in range(ni):
                    for b in range(nj):
                        corr[si + a, sj + b] = rho_ij
                        corr[sj + b, si + a] = rho_ij
        return _nearest_psd(corr)

    def simulate(self, market_state, n_paths: int, horizon_dates: List, rng, ref_date,
                 shock_sampler=None, antithetic: bool = False) -> JointSimResult:
        """shock_sampler: EXPERIMENTAL, default None (standard pure-Gaussian
        rng.standard_normal draw, bit-identical to prior behavior). If
        given, called as shock_sampler(rng, (n_paths, n_steps, n_drivers))
        to draw the INDEPENDENT pre-correlation shocks instead -- e.g. a
        standardized sum of several differently-distributed random
        variables (see examples/mixed_rv_comparison.py). Whatever is
        returned is still correlated via the SAME Cholesky factor as the
        pure-Gaussian path (z_indep @ chol.T), so cross-factor correlation
        structure is preserved regardless of the marginal shock
        distribution -- only the MARGINAL shape of each independent driver
        changes. NOTE: every downstream closed-form formula (LGM/GBM
        discount/level reconstruction, Brownian-bridge interpolation) is
        derived assuming Gaussian increments; a non-Gaussian shock_sampler
        makes those reconstructions approximate rather than exact -- this
        parameter exists for controlled experimentation/comparison, not as
        a production option (see examples/mixed_rv_comparison.py's
        docstring for the full caveat).

        antithetic: EXPERIMENTAL, default False (bit-identical to prior
        behavior). If True, draws only n_paths // 2 independent shock
        vectors Z and mirrors each into a second path using -Z, giving
        n_paths total paths whose first half and second half are exact
        mirror images of each other pre-correlation (still correctly
        correlated post-Cholesky, since -Z @ chol.T = -(Z @ chol.T)). This
        is the standard antithetic-variates variance-reduction technique:
        for any estimator that is (to first order) linear or odd-symmetric
        in the driving normals, pairing Z with -Z cancels first-order Monte
        Carlo sampling error in the AVERAGE of the pair, typically
        shrinking the standard error of mean-based metrics (EE, EEPE) for
        the same total path count. Requires n_paths to be even (an odd
        n_paths would leave one path unpaired); raises ValueError otherwise
        rather than silently dropping a path. Composes with shock_sampler:
        if both are given, shock_sampler draws the independent half and
        antithetic mirrors THAT (whatever its marginal distribution), not
        assumed-Gaussian negation semantics beyond "the additive inverse of
        whatever was drawn" -- for a genuinely non-Gaussian, non-symmetric
        shock_sampler this technically still cancels linear/odd-order error
        but the theoretical variance-reduction guarantee is strongest (and
        best-understood) for the Gaussian case tested here.
        """
        factor_order, slot_ranges, n_drivers = self._driver_layout()
        corr = self._build_correlation_matrix(market_state, factor_order, slot_ranges, n_drivers)
        chol = np.linalg.cholesky(corr)

        times = [0.0] + [year_frac(ref_date, d) for d in horizon_dates]
        n_steps = len(times) - 1
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError(f"antithetic=True requires an even n_paths, got {n_paths}")
            half = n_paths // 2
            z_half = (rng.standard_normal((half, n_steps, n_drivers)) if shock_sampler is None
                      else shock_sampler(rng, (half, n_steps, n_drivers)))
            z_indep = np.concatenate([z_half, -z_half], axis=0)
        elif shock_sampler is None:
            z_indep = rng.standard_normal((n_paths, n_steps, n_drivers))
        else:
            z_indep = shock_sampler(rng, (n_paths, n_steps, n_drivers))
        z_joint = z_indep @ chol.T   # (n_paths, n_steps, n_drivers), correlated across the last axis

        rate_states, rate_models = {}, {}
        for factor, model in self._rate_factors:
            s, n = slot_ranges[factor]
            ext_z = z_joint[:, :, s:s + n]
            rate_states[factor] = model.simulate_paths(n_paths, horizon_dates, rng, external_z=ext_z)
            rate_models[factor] = model

        spot_states, spot_models = {}, {}
        for factor, model, _drift_factor, _foreign_factor in self._spot_factors:
            s, n = slot_ranges[factor]
            ext_z = z_joint[:, :, s:s + n]
            spot_states[factor] = model.simulate_paths(n_paths, horizon_dates, rng, external_z=ext_z)
            spot_models[factor] = model

        return JointSimResult(
            factor_order=factor_order, rate_states=rate_states, spot_states=spot_states,
            rate_models=rate_models, spot_models=spot_models,
            horizon_dates=list(horizon_dates), sim_times=[year_frac(ref_date, d) for d in horizon_dates],
            ref_date=ref_date, corr_chol=chol, slot_ranges=slot_ranges,
            drift_rate_factor_by_spot={f: df for f, _, df, _ in self._spot_factors},
            foreign_rate_factor_by_spot={f: ff for f, _, _, ff in self._spot_factors if ff is not None},
        )

    def drift_rate_factor_for(self, spot_factor):
        for factor, _, drift_factor, _foreign_factor in self._spot_factors:
            if factor == spot_factor:
                return drift_factor
        raise KeyError(f"no spot factor registered matching {spot_factor}")

    def foreign_rate_factor_for(self, spot_factor):
        """The OPTIONAL second (foreign-currency) rate factor for an FX spot
        model with a real simulated foreign leg -- None if this spot factor
        has no foreign_rate_factor registered (every equity factor, and any
        FX factor still using the deterministic implied-curve path)."""
        for factor, _, _drift_factor, foreign_factor in self._spot_factors:
            if factor == spot_factor:
                return foreign_factor
        raise KeyError(f"no spot factor registered matching {spot_factor}")


def _nearest_psd(corr: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues to a small positive floor and renormalize to
    unit diagonal -- real collected pairwise correlations are not guaranteed
    to form a valid (positive-semidefinite) matrix, but Cholesky requires
    one.

    The old guard here skipped repair whenever every eigenvalue was
    >= -1e-10, on the assumption that "close enough to PSD" was good enough.
    It isn't: LAPACK's Cholesky (np.linalg.cholesky) applies its own
    internal pivot check and can still raise "Matrix is not positive
    definite" on a matrix with a merely-tiny negative eigenvalue (observed:
    -1.85e-15, well inside that tolerance, on a genuine sourced correlation
    matrix -- not a synthetic edge case). Floor is now a strictly positive
    epsilon (1e-10) applied via eigenvalue clipping unconditionally, so the
    reconstructed matrix always has a strictly positive spectrum -- cheap
    (one extra eigh call) and removes this failure mode entirely."""
    vals, vecs = np.linalg.eigh(corr)
    vals_clipped = np.maximum(vals, 1e-10)
    psd = vecs @ np.diag(vals_clipped) @ vecs.T
    d = np.sqrt(np.diag(psd))
    return psd / np.outer(d, d)
