"""
Per-trade independent simulation: each trade gets its OWN small Cholesky
decomposition over ONLY its own risk factors (e.g. a plain single-name
equity TRS needs just {rate, its equity} -- a 2x2 Cholesky; a compo trade
needs {rate, equity, FX} -- 3x3; a bond forward needs just {rate} -- 1x1),
with its OWN independent random draw -- NOT sharing any Brownian increments
with any other trade's simulation.

This is a DELIBERATE alternative to the book-wide JointSimulator used by
exposure_profile.py/greeks_report.py, confirmed explicitly with the user:
path index `p` in one trade's simulation has NO relationship to path index
`p` in another trade's simulation. Concretely this means NETTED exposure
computed from per-trade simulations will NOT reflect real cross-trade
correlation (e.g. two USD-rate trades held by the same counterparty will
see genuinely different, uncorrelated simulated rate paths) -- see
risk_engine.netting.independent_aggregate for the (clearly labeled,
NOT-a-netted-number) counterparty-level combination this implies.

Implementation note: JointSimulator itself needs ZERO changes to support
this. Its correlation-matrix construction (simulation/joint.py's
_build_correlation_matrix) already builds a Cholesky over exactly the
factors that were add_rate/add_spot'd to it -- it has only ever been
CALLED at book scope before. Calling it once per trade, with only that
trade's own factors added and a trade-specific independent rng, gives
exactly the "Cholesky per trade" the user asked for, using the existing,
already-tested correlation machinery unmodified.
"""
import hashlib
from typing import Dict

import numpy as np

from capitolis_pricers.pricers.equity_trs import EquityTRS
from capitolis_pricers.pricers.bond_forward import BondForwardTrade
from capitolis_pricers.pricers.bond_trs import BondTRS

from ..factors.extract import extract_factors
from .joint import JointSimulator, JointSimResult


def _trade_seed(trade_id: str) -> int:
    """Deterministic per-trade seed derived from the trade id, so repeated
    runs are reproducible without trades sharing an rng STATE (a shared
    np.random.default_rng advanced sequentially across trades would still
    be reproducible but would make trade B's draw depend on how many
    numbers trade A's simulation consumed first -- an unnecessary, fragile
    coupling). hashlib (not Python's salted built-in hash()) so the seed is
    stable across processes/runs, which matters here since multiprocess
    pricing workers are a real code path elsewhere in this engine."""
    digest = hashlib.sha256(trade_id.encode()).digest()
    return int.from_bytes(digest[:4], "big")


def extract_trade_factors(trade):
    """The single-trade analogue of factors.extract.extract_factors --
    works by construction (extract_factors is a pure loop over whatever
    trade dict it's given; a single-trade dict is a mechanically valid,
    just previously undocumented, call shape)."""
    if isinstance(trade, EquityTRS):
        return extract_factors(equity_trs={"_": trade})
    if isinstance(trade, BondForwardTrade):
        return extract_factors(bond_forwards={"_": trade})
    if isinstance(trade, BondTRS):
        return extract_factors(bond_trs={"_": trade})
    raise TypeError(f"unrecognized trade type {type(trade).__name__}")


def build_trade_simulator(trade, rate_model_name: str, equity_model_name: str, fx_model_name: str,
                           usd_curve, rate_vol_surface, equity_spot: Dict[str, float],
                           equity_vol_surface: Dict[str, object], equity_dividend_rates: Dict[str, float],
                           fx_spot: float = None, fx_vol_surface=None, implied_foreign_curve=None,
                           rate_model_kwargs: dict = None, **rate_priors) -> JointSimulator:
    """Builds a JointSimulator wired ONLY for `trade`'s own factors -- the
    per-trade factor-subset restriction that gives this mode its small,
    trade-scoped Cholesky (see module docstring). Calibration inputs
    (curves/spots/vols) are the same book-wide market data used elsewhere;
    only the SET OF FACTORS ADDED, and later the rng used to simulate, are
    trade-scoped.

    rate_model_kwargs: passed to the rate model's CONSTRUCTOR (e.g.
        {'mean_reversion_a1': 0.08} for a book with long-duration bonds --
        LGM's H(u) shift grows with u/a1, and the 0.03 default amplifies a
        routine simulated state into an unrealistic price swing for 30-40yr
        cashflows; see exposure_profile.py's identical comment for the full
        investigation). Distinct from **rate_priors, which are passed to
        .calibrate() (SV prior overrides, not constructor args).
    """
    from ..models.registry import get_rate_model, get_spot_model

    trade_factors = extract_trade_factors(trade)
    sim = JointSimulator()

    rate_factor = trade_factors.rates[0]  # always present -- every trade type discounts off USD
    rate_calibrated = get_rate_model(rate_model_name, **(rate_model_kwargs or {})).calibrate(
        usd_curve, rate_vol_surface, **rate_priors)
    sim.add_rate(rate_factor, rate_calibrated)

    for eq_factor in trade_factors.equities:
        calibrated = get_spot_model(equity_model_name).calibrate(
            equity_spot[eq_factor.isin], rate_calibrated, equity_vol_surface[eq_factor.isin],
            dividend_rate=equity_dividend_rates.get(eq_factor.isin, 0.0))
        sim.add_spot(eq_factor, calibrated, drift_rate_factor=rate_factor)

    for fx_factor in trade_factors.fx:
        if fx_spot is None or implied_foreign_curve is None:
            raise ValueError(f"trade has an FX factor ({fx_factor}) but no fx_spot/implied_foreign_curve given")
        fx_calibrated = get_spot_model(fx_model_name).calibrate(
            fx_spot, rate_calibrated, fx_vol_surface, implied_foreign_curve=implied_foreign_curve)
        sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

    return sim


def simulate_trade(trade_id: str, trade, sim: JointSimulator, n_paths: int, horizon_dates, ref_date) -> JointSimResult:
    """One independent simulate() call for `trade`, using its OWN rng
    (seeded off trade_id, NOT shared with any other trade -- see
    _trade_seed's docstring for why this is a fresh rng per trade, not a
    shared advancing one)."""
    from capitolis_pricers.market import MarketState
    rng = np.random.default_rng(_trade_seed(trade_id))
    return sim.simulate(MarketState(ref_date=ref_date), n_paths=n_paths, horizon_dates=horizon_dates,
                         rng=rng, ref_date=ref_date)


def simulate_all_trades_independently(trades: Dict[str, object], n_paths: int, horizon_dates, ref_date,
                                       rate_model_name: str, equity_model_name: str, fx_model_name: str,
                                       usd_curve, rate_vol_surface, equity_spot, equity_vol_surface,
                                       equity_dividend_rates, fx_spot=None, fx_vol_surface=None,
                                       implied_foreign_curve=None, rate_model_kwargs: dict = None,
                                       **rate_priors) -> Dict[str, JointSimResult]:
    """{trade_id: JointSimResult} -- one independent simulation per trade,
    each with its own small Cholesky over only that trade's own factors and
    its own independent rng. This is the top-level entry point for
    per-trade mode; feed the result into pricing.price_curves_per_trade.

    rate_model_kwargs: see build_trade_simulator's docstring -- rate model
        CONSTRUCTOR kwargs (e.g. mean_reversion_a1 for long-duration books),
        applied uniformly to every trade's own rate model instance."""
    out = {}
    for tid, trade in trades.items():
        sim = build_trade_simulator(
            trade, rate_model_name, equity_model_name, fx_model_name, usd_curve, rate_vol_surface,
            equity_spot, equity_vol_surface, equity_dividend_rates, fx_spot, fx_vol_surface,
            implied_foreign_curve, rate_model_kwargs=rate_model_kwargs, **rate_priors)
        out[tid] = simulate_trade(tid, trade, sim, n_paths, horizon_dates, ref_date)
    return out
