"""
Compute the fast, non-simulation-heavy inputs the PPT-style report needs that
are NOT already saved by the exposure / greeks / xva scripts:

  * per-trade static attributes: counterparty, notional, direction, maturity,
    years-to-maturity, asset class, risk-factor tags
  * per-trade mark-to-market at the reference date (t0), from a single
    joint-simulation path (the t0 slice is deterministic -- every path shares
    the same t0 state -- so 1 path is exact for the t0 mark)
  * per-counterparty aggregates of the above
  * a full SA-CCR RC / PFE / EAD chain (Basel CRE52), computed from the
    notionals / maturities / supervisory factors already in the engine plus
    the documented uncollateralised-netting-set assumptions below
  * analytic IR DV01 for the rate-sensitive trades (bond forwards / bond TRS)

Everything here is cheap (< 1 min). The exposure-profile EE/EEPE/MPE numbers,
the vega bump-and-reprice numbers, and the CVA/DVA/FVA numbers are NOT
recomputed here -- they are read from the existing 10k-path JSON outputs by
build_report_pdf.py.

    python risk_engine/examples/report_inputs.py
"""
import json
import math
import os
import sys
from datetime import date, timedelta

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from risk_engine.pricing import price_curves, trade_maturity
from risk_engine.netting import build_netting_hierarchy
from risk_engine.greeks.sensitivities import (
    notional_of, direction_sign, asset_class_of, trade_start_end, is_compo,
)
from risk_engine.greeks.sa_ccr import supervisory_duration, SUPERVISORY_FACTOR
from risk_engine.examples._sourced_book import build_sourced_book

OUT_PATH = os.path.join(os.path.dirname(__file__), "report_inputs.json")

# ---- SA-CCR assumptions (documented, not invented inputs) -------------------
# CRE52 alpha. Fixed by the standard.
ALPHA = 1.4
# Netting sets are uncollateralised: the trade book ships no CSA / margin
# terms (no threshold, no MTA, no independent amount), so per CRE52.20 we
# treat each counterparty netting set as unmargined:
#   RC = max(V - C, 0)  with C = 0
#   maturity factor MF = sqrt(min(M, 1yr) / 1yr)   (unmargined form)
# This is an assumption forced by absent CSA data, flagged as such in the
# report, NOT a fabricated input.
MPOR_DAYS = None  # unmargined -> no MPOR in the MF

SUPERVISORY_CORREL = {          # CRE52.75 single-factor rho by asset class
    "interest_rate": None,      # IR uses hedging-set / maturity-bucket rules, not a single rho
    "foreign_exchange": 0.5,
    "equity_single_name": 0.5,
}


def _yearfrac(d0, d1):
    return max((d1 - d0).days / 365.0, 0.0)


def risk_factor_tags(trade):
    """Coarse risk-factor labels for the report's 'Risk Factors' column."""
    tags = set()
    ac = asset_class_of(trade)
    if ac == "interest_rate":
        tags.add("IR:USD")
    elif ac == "equity":
        tags.add("EQ")
        tags.add("IR:USD")  # discounting + funding leg
    elif ac == "equity_fx_compo":
        tags.update({"EQ", "FX:USDJPY", "IR:USD"})
    return sorted(tags)


def saccr_addon_ir(trades_ir, ref_date):
    """SA-CCR interest-rate add-on for one netting set (single currency USD =
    one hedging set; three maturity buckets <1y / 1-5y / >5y aggregated per
    CRE52.50-52.56 with the 70% inter-bucket correlation)."""
    buckets = {0: 0.0, 1: 0.0, 2: 0.0}  # effective notional D_i per bucket
    for t in trades_ir:
        S, E = trade_start_end(t, ref_date)
        sd = supervisory_duration(S, E)
        d = notional_of(t) * direction_sign(t) * sd
        M = _yearfrac(ref_date, trade_maturity(t))
        mf = math.sqrt(min(M, 1.0) / 1.0) if M > 0 else 0.0
        d *= mf
        b = 0 if E < 1 else (1 if E <= 5 else 2)
        buckets[b] += d
    D0, D1, D2 = buckets[0], buckets[1], buckets[2]
    effective = math.sqrt(
        D0 * D0 + D1 * D1 + D2 * D2
        + 1.4 * D0 * D1 + 1.4 * D1 * D2 + 0.6 * D0 * D2
    )
    return SUPERVISORY_FACTOR["interest_rate"] * effective, buckets


def saccr_addon_linear(trades_ac, ref_date, asset_class, sf_key):
    """SA-CCR add-on for FX or equity (delta = +-1, one hedging set per
    currency pair for FX; per-name hedging sets for single-name equity, each
    aggregated with rho=0.5 across names)."""
    sf = SUPERVISORY_FACTOR[sf_key]
    rho = SUPERVISORY_CORREL[sf_key]
    if asset_class == "fx":
        # one hedging set (USD/JPY): sum signed effective notionals
        eff = 0.0
        for t in trades_ac:
            M = _yearfrac(ref_date, trade_maturity(t))
            mf = math.sqrt(min(M, 1.0) / 1.0) if M > 0 else 0.0
            eff += notional_of(t) * direction_sign(t) * mf
        return sf * abs(eff), {"USDJPY": eff}
    # equity single-name: per-name add-on then rho aggregation
    by_name = {}
    for i, t in enumerate(trades_ac):
        M = _yearfrac(ref_date, trade_maturity(t))
        mf = math.sqrt(min(M, 1.0) / 1.0) if M > 0 else 0.0
        # every equity TRS references a basket; treat each trade as its own
        # single-name-equivalent hedging set (documented simplification --
        # basket constituent weights are not in scope for the SA-CCR add-on)
        key = f"eq{i}"
        by_name[key] = by_name.get(key, 0.0) + notional_of(t) * direction_sign(t) * mf
    addon_names = {k: sf * abs(v) for k, v in by_name.items()}
    systematic = sum(rho * a for a in addon_names.values())
    idiosyncratic = sum((1 - rho * rho) * a * a for a in addon_names.values())
    aggregate = math.sqrt(systematic * systematic + idiosyncratic)
    return aggregate, by_name


def compute_report_inputs(book, restrict_to_trade_ids=None):
    """The full report_inputs.json-shaped payload, computed from an
    ALREADY-LOADED book (risk_engine.examples._sourced_book.build_sourced_book
    output) -- factored out of main() so dashboard_runner.py can call this
    directly against the SAME book object exposure_profile.py/greeks_report.py/
    xva_report.py already loaded for a given pricing date, instead of
    reloading the book a second time or reimplementing this SA-CCR/DV01/MTM
    logic (this function's CRE52 hedging-set aggregation with proper
    cross-bucket correlations and the multiplier formula is the real,
    validated implementation -- do not re-derive a simplified version of it
    elsewhere).

    restrict_to_trade_ids: optional iterable of trade ids to keep (e.g. to
    match a production run that dropped a trade whose market data failed to
    fetch) -- None keeps every trade in book["trades"].
    """
    ref_date = book["ref_date"]
    trades = dict(book["trades"])
    if restrict_to_trade_ids is not None:
        keep = set(restrict_to_trade_ids)
        dropped = [t for t in trades if t not in keep]
        if dropped:
            print(f"Restricting to the production trade set; dropping {dropped}")
            trades = {t: v for t, v in trades.items() if t in keep}
    grid = book["grid"]
    equity_div = book["market_data"]["equity_dividend_rates"]

    # ---- t0 mark: one path is exact for the t0 slice --------------------
    sim = book["build_simulator"]()
    rng = np.random.default_rng(0)
    precache = sim.simulate(book["market_state_for_correlation"], n_paths=1,
                            horizon_dates=grid.dates, rng=rng, ref_date=ref_date)
    # price only at the reference date
    result = price_curves(trades, precache, [ref_date],
                          equity_dividend_rates=equity_div, n_workers=1)
    mtm0 = {tid: result.npv0[(0, ref_date, tid)] for tid in trades}

    # ---- per-trade statics -------------------------------------------------
    trade_rows = {}
    for tid, t in trades.items():
        mat = trade_maturity(t)
        S, E = trade_start_end(t, ref_date)
        trade_rows[tid] = {
            "counterparty": t.counterparty,
            "type": type(t).__name__,
            "notional": float(notional_of(t)),
            "direction_sign": direction_sign(t),
            "maturity": mat.isoformat(),
            "years_to_maturity": round(E, 4),
            "asset_class": asset_class_of(t),
            "is_compo": bool(is_compo(t)),
            "risk_factors": risk_factor_tags(t),
            "mtm0": float(mtm0[tid]),
        }

    # ---- analytic IR DV01 (bump the USD curve 1bp, reprice, difference) ---
    # cheap: reprice the rate-sensitive trades only, 1 path, +1bp parallel.
    from risk_engine.calibration.market_surface import flat_vol_surface
    from risk_engine.models.registry import get_rate_model
    usd_curve = book["market_data"]["usd_curve"]

    def _bumped_curve(bp):
        # capitolis_pricers.curves.Curve stores sorted (t, ln DF); a parallel
        # +bp shift in the continuously-compounded zero rate is
        # ln DF(t) -> ln DF(t) - bp * t
        import copy
        c = copy.deepcopy(usd_curve)
        c._lndf = [y - bp * t for t, y in zip(c._t, c._lndf)]
        return c

    dv01 = {}
    try:
        base_rc = book["rate_calibrated"]
        up_rc = get_rate_model("LGM2F_SV").calibrate(
            _bumped_curve(0.0001), flat_vol_surface("RATE_USD", flat_vol=0.010))
        sim_up = book["build_simulator"]()
        # rebuild sim with bumped rate model
        # simplest: swap rate model object
        # (JointSimulator stores calibrated rate; rebuild fresh)
        import risk_engine.simulation.joint as _j
        s2 = _j.JointSimulator()
        rf = book["factors"].rates[0]
        s2.add_rate(rf, up_rc)
        # only need rate-sensitive trades, but pricer needs equity/fx too for
        # equity TRS; restrict to pure IR trades:
        ir_trades = {tid: t for tid, t in trades.items()
                     if asset_class_of(t) == "interest_rate"}
        pc_up = s2.simulate(book["market_state_for_correlation"], n_paths=1,
                            horizon_dates=grid.dates, rng=np.random.default_rng(0),
                            ref_date=ref_date)
        r_up = price_curves(ir_trades, pc_up, [ref_date],
                            equity_dividend_rates=equity_div, n_workers=1)
        for tid in ir_trades:
            dv01[tid] = float(r_up.npv0[(0, ref_date, tid)] - mtm0[tid])
    except Exception as e:  # pragma: no cover - diagnostic
        dv01 = {"_error": repr(e)}

    # ---- SA-CCR RC / PFE / EAD per counterparty --------------------------
    counterparties = build_netting_hierarchy(trades)
    saccr = {}
    for c in counterparties:
        cid = c.id
        cp_items = [(tid, t) for tid, t in trades.items() if t.counterparty == cid]
        cp_trades = [t for _, t in cp_items]
        V = sum(mtm0[tid] for tid, _ in cp_items)          # netting-set value
        RC = max(V, 0.0)                                    # C = 0, unmargined
        ir = [t for t in cp_trades if asset_class_of(t) == "interest_rate"]
        eq = [t for t in cp_trades if asset_class_of(t) in ("equity", "equity_fx_compo")]
        fx = [t for t in cp_trades if is_compo(t)]
        addon_ir, ir_buckets = saccr_addon_ir(ir, ref_date) if ir else (0.0, {})
        addon_eq, eq_names = saccr_addon_linear(eq, ref_date, "equity", "equity_single_name") if eq else (0.0, {})
        addon_fx, fx_hs = saccr_addon_linear(fx, ref_date, "fx", "foreign_exchange") if fx else (0.0, {})
        addon_agg = addon_ir + addon_eq + addon_fx
        # multiplier (CRE52.23): min(1, 0.05 + 0.95*exp((V-C)/(2*0.95*addon)))
        if addon_agg > 0:
            mult = min(1.0, 0.05 + 0.95 * math.exp(min(V, 0.0) / (2 * 0.95 * addon_agg)))
        else:
            mult = 1.0
        PFE = mult * addon_agg
        EAD = ALPHA * (RC + PFE)
        saccr[cid] = {
            "netting_set_value_V": float(V),
            "collateral_C": 0.0,
            "RC": float(RC),
            "addon_ir": float(addon_ir),
            "addon_equity": float(addon_eq),
            "addon_fx": float(addon_fx),
            "addon_aggregate": float(addon_agg),
            "multiplier": float(mult),
            "PFE": float(PFE),
            "alpha": ALPHA,
            "EAD": float(EAD),
            "ir_buckets": {str(k): float(v) for k, v in ir_buckets.items()},
        }

    # ---- per-counterparty statics ---------------------------------------
    cp_rows = {}
    for cid in sorted({t.counterparty for t in trades.values()}):
        rows = [r for r in trade_rows.values() if r["counterparty"] == cid]
        mats = sorted(r["maturity"] for r in rows)
        notl = sum(r["notional"] for r in rows)
        cp_rows[cid] = {
            "trade_count": len(rows),
            "gross_notional": notl,
            "net_notional_signed": sum(r["notional"] * r["direction_sign"] for r in rows),
            "maturities": mats,
            "earliest_maturity": mats[0],
            "latest_maturity": mats[-1],
            "notional_weighted_ttm": (
                sum(r["notional"] * r["years_to_maturity"] for r in rows) / notl if notl else 0.0
            ),
            "avg_ttm": sum(r["years_to_maturity"] for r in rows) / len(rows),
            "mtm0": sum(r["mtm0"] for r in rows),
            "risk_factors": sorted({rf for r in rows for rf in r["risk_factors"]}),
            "dv01": sum(dv01.get(tid, 0.0) for tid, r in trade_rows.items()
                        if r["counterparty"] == cid and isinstance(dv01, dict)
                        and not isinstance(dv01.get(tid), str)),
        }

    payload = {
        "ref_date": ref_date.isoformat(),
        "assumptions": {
            "saccr_alpha": ALPHA,
            "saccr_collateral": "uncollateralised (no CSA in trade data) -> C=0, RC=max(V,0), unmargined MF",
            "saccr_mf": "MF = sqrt(min(M,1yr)/1yr), unmargined",
            "saccr_equity_hedging_set": "one single-name hedging set per equity TRS (basket constituent weights out of scope for add-on)",
            "mtm0": "single simulated path; t0 slice is path-independent so exact for the t0 mark",
        },
        "trades": trade_rows,
        "counterparties": cp_rows,
        "saccr": saccr,
        "dv01_by_trade": dv01,
    }
    return payload


def main():
    # Pin to the SAME reference date as the saved 10k-path exposure / greeks /
    # xva production runs (exposure_profile.json et al.), so every static /
    # SA-CCR number in the report reconciles against those simulation results.
    PROD_REF_DATE = date(2026, 8, 24)
    book = build_sourced_book(ref_date=PROD_REF_DATE)
    # The production exposure run priced 15 trades (EQTRS_0007's underlying
    # failed to fetch at run time and the trade was dropped). Match that
    # trade set exactly for reconciliation; EQTRS_0007 is documented as an
    # excluded trade in the report.
    exp_json = os.path.join(os.path.dirname(__file__), "exposure_profile.json")
    restrict = None
    if os.path.exists(exp_json):
        restrict = set(json.load(open(exp_json))["trades"])

    payload = compute_report_inputs(book, restrict_to_trade_ids=restrict)
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved {OUT_PATH}")
    for cid, s in payload["saccr"].items():
        print(f"{cid}: RC={s['RC']:,.0f}  PFE={s['PFE']:,.0f}  EAD={s['EAD']:,.0f}")


if __name__ == "__main__":
    main()
