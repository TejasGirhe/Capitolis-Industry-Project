# Capitolis Industry Project — Counterparty Credit Risk Engine

A Berkeley MFE x Capitolis industry project: an end-to-end stochastic simulation and
counterparty credit risk engine covering risk factor extraction, pluggable rate/equity/FX
models, joint Monte Carlo simulation, exposure profiling (EE/PFE/MPE/EEPE), SA-CCR delta,
bump-and-reprice vega, and xVA (CVA/DVA/FVA) — built on real market data (FRED, Yahoo
Finance) wherever a free source exists, with every placeholder/proxy explicitly documented.

## Layout

- `capitolis_pricers/` — trade/instrument definitions and deterministic pricers (Equity TRS,
  Bond Forward, Bond TRS) given a `MarketState`.
- `risk_engine/` — the risk engine: stochastic models (LGM rate models, GBM equity, FXGBM FX),
  joint/per-trade simulation, the pricing pipeline, exposure engine, netting, Greeks (SA-CCR
  delta + vega), xVA, real market-data sourcing, and runnable example scripts
  (`risk_engine/risk_engine/examples/`).
- `risk_engine/tests/` — pytest suite covering model calibration, exposure computation,
  netting, SA-CCR/vega correctness, per-trade independence, and xVA formulas.
- `risk_engine/risk_engine/examples/Capitolis_Risk_Engine_Report.pdf` — full methodology +
  results report.

## Running

```bash
cd risk_engine
pytest tests/
python risk_engine/examples/exposure_profile.py [n_paths] [n_workers] [ref_date]
python risk_engine/examples/greeks_report.py [n_paths] [n_workers] [ref_date]
python risk_engine/examples/xva_report.py [n_paths] [n_workers] [ref_date]
```

See `risk_engine/README.md` and the module docstrings under `risk_engine/risk_engine/` for
methodology detail, and the PDF report for a narrated walkthrough of the full build.
