# Interactive Risk Dashboard

A 5-tab Plotly Dash app over the saved risk-engine outputs. No simulation runs
at launch -- it reads JSON artifacts only.

## Run

```bash
pip install dash plotly pandas          # one-time
python risk_engine/examples/dashboard.py
# open http://127.0.0.1:8050
```

## Data it reads (regenerate upstream first if stale)

| File | Produced by | Contents |
|---|---|---|
| `report_inputs.json` | `report_inputs.py` | statics, t0 MTM, **SA-CCR RC/PFE/EAD (Basel CRE52)**, IR DV01 |
| `exposure_profile.json` | `exposure_profile.py` | 10k-path EE/NEE/PFE/MPE/EEPE, counterparty + book |
| `greeks_report.json` | `greeks_report.py` | SA-CCR delta + vega (EE/MPE bump-and-reprice), 10k |
| `xva_report.json` | `xva_report.py` | CVA/DVA/FVA, 10k |
| `xva_greeks.json` | `xva_greeks.py` | xVA vega (directional, 300 paths) |
| `exposure_profile_per_trade.json` | `exposure_profile_per_trade.py` | per-trade EE/EEPE/MPE (300 paths, ref 2026-01-15) |

Full refresh:

```bash
python risk_engine/examples/exposure_profile.py 10000     # ~15-20 min
python risk_engine/examples/greeks_report.py 10000        # ~60 min (4 pipeline runs)
python risk_engine/examples/xva_report.py 10000
python risk_engine/examples/report_inputs.py              # < 1 min
python risk_engine/examples/dashboard.py
```

## Tabs

1. **Counterparty Base** -- Name, Notional, Trade Count, Maturities, Avg Maturity,
   Credit Spread, Recovery, Risk Factors, **CVA, DVA, FVA, Net XVA** (separate
   columns), SV01, DV01, MTM, EE, EEPE, MPE, 95%, 99%, plus SA-CCR RC/PFE/EAD.
   Sort / filter / export CSV.
2. **Trade Base** -- CPTY, Trade ID, Type, MTM, Notional, Risk Factors, IR DV01,
   EE, EEPE, MPE 95/99. Trade EE/EEPE/MPE are **300-path indicative** (banner in tab).
3. **Risk Metrics (Shocks)** -- every counterparty metric (incl. **CVA / DVA / FVA /
   Net XVA separately**) under each shock: `{Delta, Vega} x {IR, FX, EQ, CD}`.
   Vega = 10k re-sim for EE/EEPE/MPE + 300p directional for CVA/DVA/FVA; IR delta =
   analytic +1bp; FX/EQ delta = SA-CCR supervisory delta (shown in DV01 column);
   CD = N/A. Quick-filter dropdown isolates one shock.
4. **Base vs Shocked** -- Base / Shocked / Absolute Change / % Change / Shock Type /
   Risk Factor. Rows per scope per shock for EE, EEPE, MPE 95/99, **CVA, DVA, FVA**
   (dir.), MTM (IR delta P&L), SA-CCR delta (IR/FX/EQ effective notional), SA-CCR
   EAD (vol-independent -> 0). Header flags the largest deterioration.

## Glossary

- **SV01** -- credit-spread DV01 (a.k.a. CS01): P&L for a +1bp shift in the
  counterparty credit spread. **N/A here** -- bonds are modelled risk-free, so there
  is no issuer spread curve to shift; SV01 is undefined, not zero.
- **DV01** -- interest-rate DV01: P&L for a +1bp parallel shift of the USD curve.
- **10k / 300p** -- Monte Carlo path count. Base exposure & XVA = 10,000 paths.
  The XVA-vega run and the per-trade run were done at 300 paths ("300p"); those
  figures are directional (sign/rough magnitude), not precise.
- **Net XVA** = CVA - DVA + FVA.
- **MPE** -- maximum potential exposure = peak percentile PFE across reporting dates.
- **EEPE** -- Basel effective expected positive exposure (time-weighted running-max EE).
5. **Graphs** -- toggle counterparty vs trade level. 16+ counterparty bar charts
   (Notional, Trade Count, MTM, EE, EEPE, MPE 95/99, XVA, DV01, SV01, Credit Spread,
   Avg Maturity, SA-CCR EAD/PFE/RC) + exposure term structure; trade-level MTM /
   Notional / DV01 / EE / EEPE / MPE.

## SA-CCR (Basel CRE52) -- what's computed and assumed

- `EAD = 1.4 x (RC + PFE)`, `RC = max(V - C, 0)`, `PFE = multiplier x AddOn_agg`.
- Netting sets **uncollateralised** (`C = 0`, unmargined maturity factor) -- the
  trade data has no CSA. This is a stated assumption, not an invented input.
- Supervisory factors IR 0.50% / FX 4% / single-name equity 32%; rho 50% (FX, EQ);
  IR 3-bucket aggregation (70/70/60).
- Equity add-on at trade level (basket-constituent weights not decomposed) -- an
  upper bound vs a fully decomposed calc.
- **Regulatory SA-CCR metrics are shown in their own columns and never added to the
  simulation-based EE/EEPE/MPE.**

## Known limitations (surfaced in-app)

- No trade-level 10k exposure run -> Tab 2 EE/EEPE/MPE are 300-path, older ref date.
- CD (credit) delta/vega: N/A -- bonds modelled risk-free, no credit-spread factor.
- xVA vega: 300-path directional only (sign/magnitude, not a precise 10k figure).
- CPTY_C 99% rate-vega is large -- thin 99% tail on a short-dated book, flagged.
