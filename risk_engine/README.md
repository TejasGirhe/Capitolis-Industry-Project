# risk_engine

Sibling package to `capitolis_pricers/` -- depends on it as a library, never
modifies it. Covers two pieces of the simulation engine: risk factor
extraction from the book, and a pluggable LGM stochastic-model library for
the rate factor.

```python
from capitolis_pricers.underlyings_loader import load_equities, load_bonds
from capitolis_pricers.trade_loader import load_equity_trs, load_bond_forward, load_bond_trs
from capitolis_pricers.curves import zero_curve

from risk_engine import extract_factors, simulate
from risk_engine.calibration.market_surface import flat_vol_surface

baskets = load_equities("trade_data/underlyings/equities.csv")
bonds   = load_bonds("trade_data/underlyings/bonds.csv")
eqtrs   = load_equity_trs("trade_data/equity_trs.csv", baskets)
bfwd    = load_bond_forward("trade_data/bond_forward.csv", bonds)
btrs    = load_bond_trs("trade_data/bond_trs.csv", bonds)

factors = extract_factors(eqtrs, bfwd, btrs)
# On the shipped book: 1 RateFactor(USD), 37 EquityFactor (41 basket rows,
# 4 ISINs repeated across trades), 1 FxFactor(USD, JPY) -- EQTRS_0005 and
# EQTRS_0006 hold JPY-currency basket rows against USD trade_ccy (compo).

usd_curve = zero_curve("2026-01-15", [0.5, 1, 2, 5, 10],
                        [0.0430, 0.0420, 0.0405, 0.0395, 0.0410])
vol_surface = flat_vol_surface("RATE_USD", flat_vol=0.010)  # or load_vol_surface(csv_path, "RATE_USD")

result = simulate("LGM2F_SV", factors.rates[0], usd_curve, vol_surface,
                   n_paths=20_000, horizon_dates=[...])
```

## Model selection is by name

`simulate(model_name, ...)` looks the model up in
`risk_engine.models.registry.MODEL_REGISTRY`:

| Name | Class | Notes |
|---|---|---|
| `LGM1F` | `LGM1F` | single-factor Hull-White/LGM |
| `LGM1F_SV` | `LGM1FSV` | + CIR stochastic vol |
| `LGM2F` | `LGM2F` | two correlated factors |
| `LGM2F_SV` | `LGM2FSV` | two factors + stochastic vol |

All four share the same `calibrate(curve, vol_surface) -> CalibratedModel`
contract, so swapping models is a one-line change.

## Forward matching

Every model's `discount_factor(state, t, T)` reproduces `DF(0,T)/DF(0,t)` in
expectation over simulated paths -- by construction of the LGM state (a
driftless Brownian motion; all curve-dependence sits in the deterministic
shift `H(t)` and today's `DF(0,.)`, not in a drift term). This is not a
post-hoc adjustment -- see `risk_engine/models/lgm.py`'s module docstring for
the closed-form derivation, and `tests/test_lgm_forward_matching.py` for the
numerical check.

## Vol surface: extended beyond MARKET_DATA.md

Capitolis's `MARKET_DATA.md` Sec.5 specifies one volatility per
`(factor, tenor)` -- a term structure only, no strike axis. The `-SV` model
variants need an actual smile to have identifiable vol-of-vol (`eta`) and
correlation (`rho`) parameters, so this package's own collection spec (not
editing the Capitolis-provided file) adds a `strike_or_moneyness` column:

```
factor,tenor,strike_or_moneyness,volatility
RATE_USD,1Y,0.98,0.0130
RATE_USD,1Y,1.00,0.0110
RATE_USD,1Y,1.02,0.0125
...
```

Load with `risk_engine.calibration.market_surface.load_vol_surface(path, factor_key)`.
When only a term structure is available (no smile), use `flat_vol_surface(...)`
instead -- `eta`/`rho` then fall back to literature-typical priors
(`risk_engine/calibration/priors.py`), and the fallback is recorded on
`CalibratedModel.assumptions` for the technical report rather than silently
assumed.

## Pricing the whole book under simulation

`risk_engine.pricing.price_book` reprices every trade at every simulated
`(path, horizon_date)` node, using `risk_engine.simulation.JointSimulator`
for one correlated Monte Carlo across all factors (rate + every equity + FX)
at once, and `risk_engine.simulation.build_market_state` to turn a node's
simulated state into a real `MarketState` -- `pricer.npv(market)` runs
completely unmodified.

```python
from risk_engine.simulation.joint import JointSimulator
from risk_engine.pricing import price_book
from risk_engine.models.registry import get_rate_model, get_spot_model
from risk_engine.calibration.implied_fx_curve import build_implied_jpy_curve

rate_calibrated = get_rate_model("LGM2F_SV").calibrate(usd_curve, rate_vol_surface)

sim = JointSimulator()
sim.add_rate(rate_factor, rate_calibrated)
for eq_factor in factors.equities:
    eq_calibrated = get_spot_model("GBM_SV").calibrate(spot, rate_calibrated, vol_surface, dividend_rate=q)
    sim.add_spot(eq_factor, eq_calibrated, drift_rate_factor=rate_factor)
fx_calibrated = get_spot_model("FXGBM_SV").calibrate(fx_spot, rate_calibrated, fx_vol_surface,
                                                       implied_foreign_curve=build_implied_jpy_curve(...))
sim.add_spot(fx_factor, fx_calibrated, drift_rate_factor=rate_factor)

joint_result = sim.simulate(market_state, n_paths=10_000, horizon_dates=[...], rng=rng, ref_date=ref_date)
result = price_book(trades, joint_result, equity_dividend_rates=equity_div)
```

See `risk_engine/examples/price_book_lgm2fsv.py` for a full runnable example
against the shipped 16-trade book (`python -m risk_engine.examples.price_book_lgm2fsv`).

**Type-safe model interfaces**: `RateModel`/`CalibratedRateModel` (rates,
`discount_factor(state, t, T)`) and `SpotModel`/`CalibratedSpotModel`
(equity/FX, `level_at(state, t, rate_state)`) are separate ABCs with separate
registries (`get_rate_model`/`RATE_MODEL_REGISTRY`,
`get_spot_model`/`SPOT_MODEL_REGISTRY`) -- a rate model can never be handed to
code expecting a spot level or vice versa.

**Path-consistent drift**: equity/FX drift is read off each scenario's OWN
simulated rate path (via the paired `CalibratedRateModel`'s
`discount_factor`), not a static curve -- two paths with different realized
rates get different equity/FX drifts. FX's foreign (JPY) leg is a
deterministic curve implied from FX forward points via covered interest
parity (`calibration/implied_fx_curve.py`), since the book has no JPY rate
factor (MARKET_DATA.md Sec.2.2).

**Joint correlation**: `JointSimulator` builds one N-driver correlated normal
draw per time step (Cholesky of the full rate+equity+FX correlation matrix,
looked up via the existing `MarketState.correlation`), so scenario NPVs are
genuinely correlated across all 16 trades, not independently shocked per leg.

## Known limitations (this slice)

- `bonds.csv` populates `issuer='US TREASURY N/B'` on every bond even though
  README Sec.7/8 describes them as risk-free -- `extract_factors(...,
  include_risky_bond_credit=True)` will therefore add a `CreditFactor` for
  Treasuries if switched on. Off by default; flagging so it isn't
  accidentally enabled and mistaken for a real credit-risky book.
- 2-factor SV mean-reversion-of-variance (`kappa`) has no identifying market
  data in this book (no swaption/cap grid) and is always a literature prior,
  never fit -- true for rates, equity, and FX alike.
- `build_market_state` (and `price_book`) assume a single rate factor drives
  every spot factor's drift; a genuinely multi-currency book (more than one
  `RateFactor`) would need the drift-factor mapping threaded through
  explicitly rather than inferred.
- The joint correlation matrix applies one correlation value uniformly across
  all of a multi-factor rate model's internal state variables (e.g. LGM2F's
  two factors both get the same correlation to a given equity) -- a
  documented simplification, since no market data source in this book
  distinguishes a 2F rate model's short-factor vs long-factor correlation to
  equity/FX.
