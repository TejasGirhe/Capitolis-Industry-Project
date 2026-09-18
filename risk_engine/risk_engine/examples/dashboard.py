"""
Interactive counterparty-credit-risk / XVA / SA-CCR dashboard.

    python risk_engine/examples/dashboard.py
    -> open http://127.0.0.1:8050

Flow (this is the interactive version -- see dashboard_runner.py for the
pipeline it drives):

    1. SETUP screen: pick a pricing date and a counterparty scope (one
       counterparty, or all). The trade set for that scope is shown before
       you commit to running anything.
    2. Click Run -> PROGRESS screen: the full pipeline (exposure -> greeks
       -> xva -> statics, all at 10,000 paths) runs in a background thread;
       the page polls and shows live stage/percent status. This takes
       15-45+ minutes -- it is a full production-scale Monte Carlo run,
       not a toy demo, same cost as running exposure_profile.py /
       greeks_report.py / xva_report.py by hand.
    3. RESULTS: the same 5 tabs the static version always had, now built
       from THIS run's live results (filtered to the counterparty scope
       chosen in step 1) instead of pre-saved JSON files.

Tabs
    1  Counterparty-level info -- base scenario
    2  Trade-level info -- base scenario
    3  Risk metrics -- every counterparty metric x shock type (delta/vega, IR/FX/EQ/CD)
    4  Base vs shocked values
    5  Graphs / plots -- counterparty and trade level, base scenario

SA-CCR figures follow Basel CRE52 (see report_inputs.py): EAD = 1.4 x (RC + PFE),
uncollateralised netting sets, supervisory factors IR 0.50% / FX 4% / EQ 32%.
Regulatory SA-CCR metrics are shown in their own columns and never added to the
simulation-based EE / EEPE / MPE.
"""
import os
import sys
import threading
import time
import uuid
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update, ALL, ctx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

BLUE = "#1e3c6e"
STEEL = "#4a7fb5"
AMBER = "#b0781e"
RED = "#963022"
GREEN = "#1e6e3c"
CPTY_PALETTE = [BLUE, STEEL, AMBER, "#6e1e5e", "#1e6e5a", "#8a5a1e"]


# ================================================================ RUN REGISTRY
# Dash dcc.Store values must be JSON-serializable; this project's run
# results (ExposureProfile objects, GreeksReport, XVAResult, Pricer
# instances) are not. Results are kept here, in a plain in-process dict
# keyed by a run id, and the dcc.Store only ever carries that id (a
# string) plus small UI state -- standard pattern for a single-user local
# Dash app where "the server process IS the session".
_RUNS = {}     # run_id -> {"status": "running"|"done"|"error", "stage": str,
               #            "pct": float, "result": dict|None, "error": str|None}
_RUNS_LOCK = threading.Lock()


SAVED_RUNS_DIR = os.path.join(HERE, "dashboard_saved_runs")


def _saved_run_path(run_id):
    return os.path.join(SAVED_RUNS_DIR, f"{run_id}.json")


def _save_run_to_disk(run_id, pricing_date_str, n_paths, view_all_scope):
    """Persists the ALL-SCOPE view (already plain dicts / JSON-serializable
    -- see build_view, which never returns the raw Pricer/ExposureProfile/
    GreeksReport/XVAResult objects) so a completed run survives a server
    restart and can be reloaded later without re-running the pipeline.
    Saved once per completed run, at the ALL scope -- a single-counterparty
    scope is just a filtered VIEW of the same underlying data (see
    build_view's cpty_scope filtering), so there is no need to save one
    file per scope choice; re-filtering on load is cheap and exact."""
    os.makedirs(SAVED_RUNS_DIR, exist_ok=True)
    import json
    payload = {"run_id": run_id, "pricing_date": pricing_date_str, "n_paths": n_paths,
              "saved_at": time.time(), "view_all_scope": view_all_scope}
    with open(_saved_run_path(run_id), "w") as fh:
        json.dump(payload, fh)


def list_saved_runs():
    """[{run_id, pricing_date, n_paths, saved_at}, ...] newest first --
    metadata only (doesn't load each full view), for the setup screen's
    'view saved results' picker."""
    import json
    if not os.path.isdir(SAVED_RUNS_DIR):
        return []
    out = []
    for fname in os.listdir(SAVED_RUNS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SAVED_RUNS_DIR, fname)) as fh:
                data = json.load(fh)
            out.append({"run_id": data["run_id"], "pricing_date": data["pricing_date"],
                       "n_paths": data["n_paths"], "saved_at": data["saved_at"]})
        except Exception:
            continue   # a partially-written or corrupt file -- skip rather than crash the setup screen
    out.sort(key=lambda r: r["saved_at"], reverse=True)
    return out


def _load_saved_run(run_id):
    """Loads a saved run's ALL-SCOPE view back into _RUNS as a 'done' run
    (view_all_scope is already the finished product of build_view, so no
    dashboard_runner re-run is needed) -- returns True if found."""
    import json
    path = _saved_run_path(run_id)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        data = json.load(fh)
    with _RUNS_LOCK:
        _RUNS[run_id] = {"status": "done", "stage": "Loaded from saved results", "pct": 100.0,
                         "result": None, "error": None, "started": data["saved_at"],
                         "saved_view_all_scope": data["view_all_scope"]}
    return True


def _start_run(pricing_date, n_paths=10_000):
    run_id = uuid.uuid4().hex
    with _RUNS_LOCK:
        _RUNS[run_id] = {"status": "running", "stage": "Starting...", "pct": 0.0,
                          "result": None, "error": None, "started": time.time()}

    def _progress_cb(stage, pct):
        with _RUNS_LOCK:
            _RUNS[run_id]["stage"] = stage
            _RUNS[run_id]["pct"] = pct

    def _worker():
        try:
            from risk_engine.examples.dashboard_runner import run_full_pipeline
            result = run_full_pipeline(pricing_date, n_paths=n_paths, progress_cb=_progress_cb)
            with _RUNS_LOCK:
                _RUNS[run_id]["result"] = result
                _RUNS[run_id]["status"] = "done"
            # Persist the ALL-SCOPE view so this run survives a restart --
            # build_view() is cheap (no re-simulation), safe to call again here.
            try:
                view_all = build_view(result, "ALL")
                _save_run_to_disk(run_id, pricing_date.isoformat(), n_paths, view_all)
            except Exception:
                pass   # saving is a nice-to-have; a save failure must not un-do a completed run
        except Exception as e:
            import traceback
            with _RUNS_LOCK:
                _RUNS[run_id]["status"] = "error"
                _RUNS[run_id]["error"] = f"{e}\n\n{traceback.format_exc()}"

    threading.Thread(target=_worker, daemon=True).start()
    return run_id


def _preview_trades(pricing_date):
    """Load the book (cheap relative to a full simulation -- market-data
    sourcing only, no Monte Carlo) so the setup screen can show the trade
    set BEFORE the user commits to a 15-45min run. Returns
    {trade_id: {counterparty, type, notional, maturity}}."""
    from risk_engine.examples._sourced_book import build_sourced_book
    from risk_engine.greeks.sensitivities import notional_of
    from risk_engine.pricing import trade_maturity
    book = build_sourced_book(ref_date=pricing_date)
    out = {}
    for tid, t in book["trades"].items():
        out[tid] = {
            "counterparty": t.counterparty,
            "type": type(t).__name__,
            "notional": float(notional_of(t)),
            "maturity": trade_maturity(t).isoformat(),
        }
    dropped = sorted(set(book["all_trades"]) - set(book["trades"]))
    return out, dropped


# ================================================================ SMALL HELPERS
def usd(x):
    try:
        return f"${x:,.0f}"
    except (TypeError, ValueError):
        return x


def money_cols(df, cols):
    df = df.copy()
    for c in cols:
        if c in df:
            df[c] = df[c].map(lambda v: usd(v) if isinstance(v, (int, float, np.floating)) else v)
    return df


def dtable(df, tid, page=25, height=None):
    return dash_table.DataTable(
        id=tid,
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        sort_action="native",
        filter_action="native",
        page_size=page,
        export_format="csv",
        style_table={"overflowX": "auto", **({"maxHeight": height, "overflowY": "auto"} if height else {})},
        style_cell={"fontFamily": "system-ui, sans-serif", "fontSize": "12px",
                    "padding": "6px 8px", "textAlign": "right", "whiteSpace": "normal"},
        style_header={"backgroundColor": BLUE, "color": "white", "fontWeight": "bold",
                      "textAlign": "center"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f2f5f9"},
            {"if": {"filter_query": '{Name} = "BOOK" || {Name} = "BOOK_TOTAL" || {CPTY} = "BOOK"'},
             "fontWeight": "bold", "backgroundColor": "#dde6f0"},
        ],
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"}
            for c in ("Name", "CPTY", "Trade ID", "Risk Factors", "Maturities", "Metric",
                      "Shock Type", "Risk Factor", "Type", "Direction", "Counterparty / Portfolio")
        ],
    )


def note(text, color="#666"):
    return html.Div(text, style={"fontSize": "11px", "color": color, "fontStyle": "italic",
                                 "margin": "6px 0 16px"})


def h(text):
    return html.H4(text, style={"color": BLUE, "margin": "18px 0 6px"})


def peak_ee(prof_dict):
    return max(prof_dict["ee"]) if prof_dict["ee"] else 0.0


# ================================================================ RESULT -> DASHBOARD DATA
# Convert a dashboard_runner.run_full_pipeline() result (live Python
# objects: ExposureProfile, GreeksReport, XVAResult, Pricer) into the
# same plain-dict/JSON shapes the original static dashboard consumed
# (report_inputs.json / exposure_profile.json / greeks_report.json /
# xva_report.json), so the tab-building logic below -- ported from the
# original static dashboard -- barely has to change.
def _profile_to_dict(p):
    return {"dates": [d.isoformat() for d in p.dates], "ee": p.ee, "nee": p.nee,
            "pfe_95": p.pfe_95, "pfe_99": p.pfe_99, "mpe_95": p.mpe_95, "mpe_99": p.mpe_99,
            "eepe": p.eepe, "tail_ee_95": p.tail_ee_95, "tail_ee_99": p.tail_ee_99,
            "tail_eepe_95": p.tail_eepe_95, "tail_eepe_99": p.tail_eepe_99}


def _get_view(run, cpty_scope):
    """View dict for a run, from EITHER a freshly-completed run (has
    run["result"], the raw dashboard_runner.run_full_pipeline() output --
    build a fresh view) OR a run loaded from a saved file on disk (has
    run["saved_view_all_scope"] instead -- re-filter that already-built
    view). Returns None if neither is present (shouldn't happen for a
    "done" run, but callers check for None regardless)."""
    if run.get("result") is not None:
        return build_view(run["result"], cpty_scope)
    if run.get("saved_view_all_scope") is not None:
        return _rescope_view(run["saved_view_all_scope"], cpty_scope)
    return None


def _rescope_view(view_all_scope, cpty_scope):
    """Re-filters an ALREADY-BUILT all-scope view dict (plain JSON-shaped
    data -- the output of build_view(result, "ALL"), whether freshly built
    or reloaded from a saved run on disk) to a different counterparty
    scope, WITHOUT needing the original raw pipeline result. This is what
    lets a saved run (which only has the all-scope view persisted, per
    _save_run_to_disk's docstring) still support switching the counterparty
    filter after being loaded."""
    all_cptys = view_all_scope["all_cptys"]
    cptys = all_cptys if cpty_scope == "ALL" else [cpty_scope]
    return dict(view_all_scope, cptys=cptys)


def build_view(result, cpty_scope):
    """result: dashboard_runner.run_full_pipeline()'s return dict.
    cpty_scope: "ALL" or one counterparty id -- filters which counterparty
    rows appear in every tab (BOOK_TOTAL is always shown too, since it is
    a real, already-computed netting-set-independent aggregate, not a sum
    over the filtered scope)."""
    ri = result["report_inputs"]
    profiles = {k: _profile_to_dict(v) for k, v in result["profiles"].items()}
    greeks_report_obj = result["greeks_report"]
    xva_by_cpty = result["xva_by_cpty"]

    all_cptys = sorted(ri["counterparties"].keys())
    cptys = all_cptys if cpty_scope == "ALL" else [cpty_scope]

    greeks_dict = greeks_report_obj.to_dict() if hasattr(greeks_report_obj, "to_dict") else greeks_report_obj
    xva_dict = {
        "xva": {c: {"cva": r.cva, "dva": r.dva, "fva": r.fva, "net_xva": r.net_xva}
                for c, r in xva_by_cpty.items()},
        "book_total": {
            "cva": sum(r.cva for r in xva_by_cpty.values()),
            "dva": sum(r.dva for r in xva_by_cpty.values()),
            "fva": sum(r.fva for r in xva_by_cpty.values()),
            "net_xva": sum(r.cva - r.dva + r.fva for r in xva_by_cpty.values()),
        },
        "counterparty_spreads": {},  # xva_report.py's spread sourcing isn't returned by main(); shown as N/A
    }

    return {
        "ref_date": result["ref_date"],
        "n_paths": result.get("n_paths", DEFAULT_N_PATHS),
        "all_cptys": all_cptys,
        "cptys": cptys,          # the scope currently in view
        "ri": ri,
        "profiles": profiles,
        "greeks": greeks_dict,
        "xva": xva_dict,
        "saccr": ri["saccr"],
    }


# ================================================================ TAB 1 -- CPTY base
def cpty_base_df(V):
    ri, P, XVA, SACCR = V["ri"], V["profiles"], V["xva"], V["saccr"]
    cptys = V["cptys"]
    rows = []
    for c in cptys:
        ci = ri["counterparties"][c]
        prof = P[c]
        xv = XVA["xva"].get(c, {})
        s = SACCR[c]
        rows.append({
            "Name": c,
            "Notional": ci["gross_notional"],
            "Trade Count": ci["trade_count"],
            "Maturities": f"{ci['earliest_maturity']} .. {ci['latest_maturity']} ({len(ci['maturities'])})",
            "Average Maturity (y)": round(ci["notional_weighted_ttm"], 3),
            "Credit Spread (bp)": None,
            "Recovery Rate": 0.40,
            "Risk Factors": "/".join(sorted({r.split(":")[0] for r in ci["risk_factors"]})),
            "CVA": xv.get("cva"), "DVA": xv.get("dva"), "FVA": xv.get("fva"), "Net XVA": xv.get("net_xva"),
            "SV01": "N/A",
            "DV01": ci["dv01"],
            "MTM": ci["mtm0"],
            "EE": peak_ee(prof), "EEPE": prof["eepe"], "MPE": prof["mpe_99"],
            "95%": prof["mpe_95"], "99%": prof["mpe_99"],
            "SA-CCR RC": s["RC"], "SA-CCR PFE": s["PFE"], "SA-CCR EAD": s["EAD"],
        })
    if V["cptys"] == V["all_cptys"]:  # only show a BOOK row when viewing every counterparty
        BOOK = P["BOOK_TOTAL"]
        rows.append({
            "Name": "BOOK",
            "Notional": sum(ri["counterparties"][c]["gross_notional"] for c in cptys),
            "Trade Count": sum(ri["counterparties"][c]["trade_count"] for c in cptys),
            "Maturities": f"{min(ri['counterparties'][c]['earliest_maturity'] for c in cptys)} .. "
                          f"{max(ri['counterparties'][c]['latest_maturity'] for c in cptys)}",
            "Average Maturity (y)": None, "Credit Spread (bp)": None, "Recovery Rate": 0.40,
            "Risk Factors": "IR/EQ/FX",
            "CVA": XVA["book_total"]["cva"], "DVA": XVA["book_total"]["dva"],
            "FVA": XVA["book_total"]["fva"], "Net XVA": XVA["book_total"]["net_xva"],
            "SV01": "N/A",
            "DV01": sum(ri["counterparties"][c]["dv01"] for c in cptys),
            "MTM": sum(ri["counterparties"][c]["mtm0"] for c in cptys),
            "EE": peak_ee(BOOK), "EEPE": BOOK["eepe"], "MPE": BOOK["mpe_99"],
            "95%": BOOK["mpe_95"], "99%": BOOK["mpe_99"],
            "SA-CCR RC": sum(SACCR[c]["RC"] for c in cptys),
            "SA-CCR PFE": sum(SACCR[c]["PFE"] for c in cptys),
            "SA-CCR EAD": sum(SACCR[c]["EAD"] for c in cptys),
        })
    return pd.DataFrame(rows)


_MONEY = ["Notional", "CVA", "DVA", "FVA", "Net XVA", "DV01", "MTM", "EE", "EEPE", "MPE",
          "95%", "99%", "SA-CCR RC", "SA-CCR PFE", "SA-CCR EAD"]
_CPTY_COL_ORDER = ["Name", "Notional", "Trade Count", "Maturities", "Average Maturity (y)",
                   "Credit Spread (bp)", "Recovery Rate", "Risk Factors",
                   "CVA", "DVA", "FVA", "Net XVA", "SV01", "DV01", "MTM",
                   "EE", "EEPE", "MPE", "95%", "99%", "SA-CCR RC", "SA-CCR PFE", "SA-CCR EAD"]


def tab1(V):
    raw = cpty_base_df(V)
    df = money_cols(raw[_CPTY_COL_ORDER], _MONEY)
    return html.Div([
        h("Tab 1 -- Counterparty-Level Information: Base Scenario"),
        note(f"{V['n_paths']:,}-path exposure run, valuation date {V['ref_date']}. Scope: "
             f"{'all counterparties' if V['cptys'] == V['all_cptys'] else V['cptys'][0]}. "
             "Notional = sum |notional|. Average Maturity = notional-weighted years to maturity. "
             "CVA / DVA / FVA reported separately (Net XVA = CVA - DVA + FVA). "
             "EE = peak single-date EE; EEPE = Basel effective EPE; MPE = peak 99% PFE; "
             "95%/99% = MPE at that confidence. SV01 = N/A (risk-free bonds -> no credit-spread "
             "factor). DV01 = +1bp parallel P&L. SA-CCR RC/PFE/EAD per Basel CRE52 (uncollateralised, "
             "alpha = 1.4) -- regulatory, shown separately, never added to EE/EEPE/MPE."),
        dtable(df, "t1"),
    ])


# ================================================================ TAB 2 -- trade base
def trade_base_df(V):
    ri = V["ri"]
    dv01 = ri["dv01_by_trade"]
    rows = []
    for tid, t in ri["trades"].items():
        if t["counterparty"] not in V["cptys"]:
            continue
        d = dv01.get(tid)
        rows.append({
            "CPTY": t["counterparty"], "Trade ID": tid,
            "Type": t["type"].replace("Trade", ""),
            "MTM": t["mtm0"], "Notional": t["notional"],
            "Risk Factors": ", ".join(t["risk_factors"]),
            "IR DV01": d if isinstance(d, (int, float)) else "N/A",
        })
    return pd.DataFrame(rows)


def tab2(V):
    raw = trade_base_df(V)
    df = money_cols(raw, ["MTM", "Notional", "IR DV01"]) if not raw.empty else raw
    return html.Div([
        h("Tab 2 -- Trade-Level Information"),
        note(f"Trade-level MTM / Notional / IR DV01 are exact at valuation date {V['ref_date']}. "
             f"Per-trade EE/EEPE/MPE at the SAME {V['n_paths']:,}-path scale as Tab 1 has not been produced in "
             "this run (only counterparty-netted exposure is); showing statics only here."),
        dtable(df, "t2", page=20) if not raw.empty else note("No trades for this counterparty.", RED),
    ])


# ================================================================ TAB 3 -- risk metrics x shock
def _vega_run(greeks, group):
    for r in greeks["vega_runs"]:
        if r["factor_group"] == group:
            return r["vega"]
    return {}


def risk_metric_rows(V):
    ri, greeks = V["ri"], V["greeks"]
    cpty_base = cpty_base_df(V)
    scope_names = list(cpty_base["Name"])
    out = []

    def base_block(shock_type, rf, applies, metric_delta):
        for c in scope_names:
            src = cpty_base[cpty_base["Name"] == c].iloc[0].to_dict()
            row = {"Shock Type": shock_type, "Risk Factor": rf, "Name": c}
            for k in ("Notional", "Trade Count", "Maturities", "Average Maturity (y)",
                      "Credit Spread (bp)", "Recovery Rate", "Risk Factors", "SV01"):
                row[k] = src[k]
            shocked = metric_delta(c) if applies else None
            for k in ("CVA", "DVA", "FVA", "Net XVA", "DV01", "MTM", "EE", "EEPE", "MPE", "95%", "99%"):
                if shocked is not None and k in shocked and shocked[k] is not None:
                    row[k] = shocked[k]
                elif applies:
                    row[k] = src[k]
                else:
                    row[k] = "N/A"
            out.append(row)

    for grp, rf in [("rate", "IR"), ("fx", "FX"), ("equity", "EQ")]:
        v = _vega_run(greeks, grp)

        def md(c, grp=grp, v=v):
            key = "BOOK_TOTAL" if c == "BOOK" else c
            vv = v.get(key, {})
            base = cpty_base[cpty_base["Name"] == c].iloc[0]
            return {
                "EE": base["EE"] + vv.get("EE_max", 0.0),
                "EEPE": base["EEPE"] + vv.get("EEPE", 0.0),
                "95%": base["95%"] + vv.get("MPE_95", 0.0),
                "99%": base["99%"] + vv.get("MPE_99", 0.0),
                "MPE": base["99%"] + vv.get("MPE_99", 0.0),
            }
        base_block("Vega", rf, True, md)

    base_block("Vega", "CD", False, lambda c: None)

    def ir_delta(c):
        base = cpty_base[cpty_base["Name"] == c].iloc[0]
        dv01 = base["DV01"] if isinstance(base["DV01"], (int, float)) else 0.0
        return {"MTM": base["MTM"] + dv01}
    base_block("Delta", "IR", True, ir_delta)

    def fx_delta(c):
        sd = greeks["sa_ccr_delta"].get(c, {})
        base = cpty_base[cpty_base["Name"] == c].iloc[0]
        return {"MTM": base["MTM"], "DV01": sd.get("foreign_exchange", "N/A")}
    base_block("Delta", "FX", True, fx_delta)

    def eq_delta(c):
        sd = greeks["sa_ccr_delta"].get(c, {})
        return {"DV01": sd.get("equity", "N/A")}
    base_block("Delta", "EQ", True, eq_delta)

    base_block("Delta", "CD", False, lambda c: None)
    return pd.DataFrame(out)


_RISK_COLS = ["Shock Type", "Risk Factor", "Name", "Notional", "Trade Count", "Maturities",
              "Average Maturity (y)", "Credit Spread (bp)", "Recovery Rate", "Risk Factors",
              "CVA", "DVA", "FVA", "Net XVA", "SV01", "DV01", "MTM", "EE", "EEPE", "MPE", "95%", "99%"]
_RISK_MONEY = ["Notional", "CVA", "DVA", "FVA", "Net XVA", "DV01", "MTM", "EE", "EEPE", "MPE", "95%", "99%"]


def tab3(V):
    raw = risk_metric_rows(V)
    df = money_cols(raw[_RISK_COLS], _RISK_MONEY)
    return html.Div([
        h("Tab 3 -- Risk Metrics by Shock Type"),
        note("Every counterparty metric under each shock. Shock Type in {Delta, Vega} x Risk "
             f"Factor in {{IR, FX, EQ, CD}}. VEGA (IR/FX/EQ): +1bp vol bump, full {V['n_paths']:,}-path "
             "re-simulation -> shocked EE/EEPE/MPE/95%/99%. DELTA (IR): analytic +1bp parallel "
             "reprice -> shocked MTM (P&L = DV01). DELTA (FX/EQ): SA-CCR supervisory delta "
             "(regulatory effective notional) shown in the DV01 column. CD (delta & vega): "
             "N/A -- bonds modelled risk-free, no credit-spread factor."),
        html.Div([
            html.Label("Quick filter: ", style={"fontWeight": "bold", "marginRight": "8px"}),
            dcc.Dropdown(
                id="t3-shock",
                options=[{"label": "All shocks", "value": "ALL"}] +
                        [{"label": f"{st} / {rf}", "value": f"{st}|{rf}"}
                         for st in ("Delta", "Vega") for rf in ("IR", "FX", "EQ", "CD")],
                value="ALL", clearable=False, style={"width": "260px", "display": "inline-block"}),
        ], style={"margin": "8px 0"}),
        html.Div(dtable(df, "t3tbl", page=40), id="t3-table"),
    ])


# ================================================================ TAB 4 -- base vs shocked
def base_vs_shocked_df(V):
    ri, greeks = V["ri"], V["greeks"]
    cpty_base = cpty_base_df(V)
    scope_names = list(cpty_base["Name"])
    rows = []
    metrics = [("EE", "EE"), ("EEPE", "EEPE"), ("MPE 95%", "95%"), ("MPE 99%", "99%")]
    for grp, rf in [("rate", "IR"), ("equity", "EQ"), ("fx", "FX")]:
        v = _vega_run(greeks, grp)
        for scope in scope_names:
            key = "BOOK_TOTAL" if scope == "BOOK" else scope
            vv = v.get(key, {})
            base = cpty_base[cpty_base["Name"] == scope].iloc[0]
            vega_map = {"EE": vv.get("EE_max", 0.0), "EEPE": vv.get("EEPE", 0.0),
                        "95%": vv.get("MPE_95", 0.0), "99%": vv.get("MPE_99", 0.0)}
            for disp, col in metrics:
                b = base[col]
                delta = vega_map[col]
                rows.append({"Counterparty / Portfolio": scope, "Metric": disp,
                             "Base": b, "Shocked": b + delta, "Absolute Change": delta,
                             "% Change": (delta / b * 100 if b else float("nan")),
                             "Shock Type": "Vega +1bp", "Risk Factor": rf})
    for scope in scope_names:
        base = cpty_base[cpty_base["Name"] == scope].iloc[0]
        dv01 = base["DV01"] if isinstance(base["DV01"], (int, float)) else 0.0
        rows.append({"Counterparty / Portfolio": scope, "Metric": "MTM",
                     "Base": base["MTM"], "Shocked": base["MTM"] + dv01, "Absolute Change": dv01,
                     "% Change": (dv01 / base["MTM"] * 100 if base["MTM"] else float("nan")),
                     "Shock Type": "Delta +1bp", "Risk Factor": "IR"})
    for scope in [c for c in scope_names if c != "BOOK"]:
        sd = greeks["sa_ccr_delta"].get(scope, {})
        for comp, rf in [("interest_rate", "IR"), ("foreign_exchange", "FX"), ("equity", "EQ")]:
            if comp in sd:
                val = sd[comp]
                rows.append({"Counterparty / Portfolio": scope, "Metric": "SA-CCR delta (effective notional)",
                             "Base": 0.0, "Shocked": val, "Absolute Change": val, "% Change": float("nan"),
                             "Shock Type": "Delta (SA-CCR)", "Risk Factor": rf})
    return pd.DataFrame(rows)


_SHOCK_ORDER = {"Delta +1bp": 0, "Delta (SA-CCR)": 1, "Vega +1bp": 2}
_RF_ORDER = {"IR": 0, "FX": 1, "EQ": 2, "CD": 3}


def _fmt_bvs(df):
    df = df.copy()
    df["% Change"] = df["% Change"].map(
        lambda v: (f"{v:+.2f}%" if isinstance(v, (int, float)) and pd.notna(v) else "n/a"))
    return money_cols(df, ["Base", "Shocked", "Absolute Change"])


def tab4(V):
    raw = base_vs_shocked_df(V)
    raw = raw.assign(_s=raw["Shock Type"].map(_SHOCK_ORDER), _r=raw["Risk Factor"].map(_RF_ORDER)) \
              .sort_values(["_s", "_r", "Counterparty / Portfolio", "Metric"]).drop(columns=["_s", "_r"]).reset_index(drop=True)
    num = raw[raw["Absolute Change"].map(lambda v: isinstance(v, (int, float)))]
    worst = num.loc[num["Absolute Change"].abs().idxmax()] if not num.empty else None
    return html.Div([
        h("Tab 4 -- Base vs Shocked Values"),
        note("DELTA rows first, then VEGA. 'Delta +1bp' / IR = analytic +1bp parallel reprice -> "
             "MTM P&L. 'Delta (SA-CCR)' = Basel CRE52 supervisory delta. 'Vega +1bp' = +1bp vol "
             "bump, full 10k re-sim for EE/EEPE/MPE."),
        html.Div(f"Largest single deterioration: {worst['Counterparty / Portfolio']} {worst['Metric']} "
                 f"under {worst['Shock Type']} / {worst['Risk Factor']} -> {usd(worst['Absolute Change'])}."
                 if worst is not None else "No numeric shocks in this scope.",
                 style={"fontWeight": "bold", "color": BLUE, "margin": "6px 0 14px"}),
        dtable(_fmt_bvs(raw), "t4tbl", page=45, height="68vh"),
    ])


# ================================================================ TAB 5 -- graphs
def _bar_fig(x, y, title, colors=None, ytitle="USD"):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=colors or BLUE,
                           text=[usd(v) if abs(v) >= 1 else f"{v:.2f}" for v in y],
                           textposition="outside"))
    fig.update_layout(title=title, template="plotly_white", height=330,
                      margin=dict(l=40, r=20, t=44, b=60), yaxis_title=ytitle,
                      title_font=dict(size=13, color=BLUE))
    return fig


def tab5(V):
    ri, P = V["ri"], V["profiles"]
    cptys = V["cptys"]
    cols = [CPTY_PALETTE[i % len(CPTY_PALETTE)] for i in range(len(cptys))]
    figs = []
    figs.append(_bar_fig(cptys, [ri["counterparties"][c]["gross_notional"] for c in cptys], "Notional by Counterparty", cols))
    figs.append(_bar_fig(cptys, [ri["counterparties"][c]["trade_count"] for c in cptys], "Trade Count", cols, "trades"))
    figs.append(_bar_fig(cptys, [ri["counterparties"][c]["mtm0"] for c in cptys], "MTM by Counterparty", cols))
    figs.append(_bar_fig(cptys, [peak_ee(P[c]) for c in cptys], "Peak EE by Counterparty", cols))
    figs.append(_bar_fig(cptys, [P[c]["eepe"] for c in cptys], "EEPE by Counterparty", cols))
    figs.append(_bar_fig(cptys, [P[c]["mpe_99"] for c in cptys], "MPE 99% by Counterparty", cols))
    if V["xva"]["xva"]:
        figs.append(_bar_fig(cptys, [V["xva"]["xva"].get(c, {}).get("net_xva", 0.0) for c in cptys],
                             "Net XVA by Counterparty", cols))
    figs.append(_bar_fig(cptys, [ri["counterparties"][c]["dv01"] for c in cptys], "IR DV01 by Counterparty", cols))
    figs.append(_bar_fig(cptys, [V["saccr"][c]["EAD"] for c in cptys], "SA-CCR EAD by Counterparty", cols))

    term = go.Figure()
    for i, c in enumerate(cptys):
        color = CPTY_PALETTE[i % len(CPTY_PALETTE)]
        term.add_trace(go.Scatter(x=P[c]["dates"], y=P[c]["ee"], mode="lines+markers", name=f"{c} EE",
                                  line=dict(color=color)))
        term.add_trace(go.Scatter(x=P[c]["dates"], y=P[c]["pfe_99"], mode="lines", name=f"{c} PFE99",
                                  line=dict(color=color, dash="dot")))
    term.update_layout(title="Exposure Term Structure (EE & PFE 99%)", template="plotly_white",
                       height=380, title_font=dict(size=13, color=BLUE))

    grid = [html.Div(dcc.Graph(figure=f), style={"width": "48%", "display": "inline-block",
                                                 "verticalAlign": "top"}) for f in figs]
    return html.Div([
        h("Tab 5 -- Graphs / Plots: Base Scenario"),
        note("Counterparty-level charts, this run's scope."),
        html.Div([dcc.Graph(figure=term)] + grid),
    ])


TABS = {"t1": tab1, "t2": tab2, "t3": tab3, "t4": tab4, "t5": tab5}


# Path-count presets matching every convergence sweep run this project has
# produced (mixed_rv_comparison.py / variance_reduction_comparison.py /
# analytic_vs_mc_convergence.py all used this exact set) -- so a number
# picked here is directly comparable to those already-reported results.
# EE/EEPE-type metrics were found to stabilize by ~5,000-10,000 paths;
# 99th-percentile tail metrics (MPE_99/PFE_99) were still drifting 1-2%
# even at 10,000-20,000 paths in that sweep -- pick accordingly.
PATH_COUNT_OPTIONS = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
DEFAULT_N_PATHS = 10_000


def setup_screen(pricing_date_str, cpty_scope, n_paths, preview_children):
    return html.Div([
        html.H2("Counterparty Credit Risk, XVA & SA-CCR -- Setup", style={"color": BLUE}),
        note("Choose a pricing date, counterparty scope, and path count, review the trade set, "
             "then run the full pipeline (exposure -> Greeks -> xVA). This is a real production-"
             "scale Monte Carlo run -- runtime scales roughly linearly with path count (10,000 "
             "paths took 15-45+ minutes in this project's own production runs; smaller counts are "
             "proportionally faster but noisier, especially for 99th-percentile tail metrics -- "
             "see the note below the path-count selector).", RED),
        html.Div([
            html.Label("Pricing date:", style={"fontWeight": "bold", "marginRight": "8px"}),
            dcc.DatePickerSingle(id="pricing-date-picker",
                                 date=pricing_date_str, display_format="YYYY-MM-DD",
                                 max_date_allowed=date.today().isoformat()),
        ], style={"margin": "12px 0"}),
        html.Div([
            html.Label("Counterparty scope:", style={"fontWeight": "bold", "marginRight": "8px"}),
            dcc.RadioItems(id="cpty-scope-radio",
                           options=[{"label": " All counterparties", "value": "ALL"}] +
                                   [{"label": f" {c}", "value": c} for c in ("CPTY_A", "CPTY_B", "CPTY_C")],
                           value=cpty_scope, inline=True),
        ], style={"margin": "12px 0"}),
        html.Div([
            html.Label("Number of paths:", style={"fontWeight": "bold", "marginRight": "8px"}),
            dcc.Dropdown(id="n-paths-dropdown",
                        options=[{"label": f"{n:,}", "value": n} for n in PATH_COUNT_OPTIONS],
                        value=n_paths, clearable=False,
                        style={"width": "160px", "display": "inline-block", "verticalAlign": "middle"}),
        ], style={"margin": "12px 0"}),
        note("EE/EEPE-type metrics stabilize by ~5,000-10,000 paths in this project's own "
             "convergence testing; MPE_99/PFE_99 (99th-percentile tail metrics) were still "
             "drifting 1-2% run-to-run even at 10,000-20,000 paths -- treat tail metrics from a "
             "smaller run as indicative, not final."),
        html.Button("Load trade set", id="preview-btn", n_clicks=0,
                   style={"padding": "8px 18px", "background": STEEL, "color": "white",
                          "border": "none", "borderRadius": "4px", "cursor": "pointer", "marginRight": "10px"}),
        html.Div(id="preview-area", children=preview_children, style={"margin": "16px 0"}),
        html.Button("Run pipeline (fresh)", id="run-btn", n_clicks=0,
                   style={"padding": "10px 22px", "background": GREEN, "color": "white",
                          "border": "none", "borderRadius": "4px", "cursor": "pointer", "fontWeight": "bold"}),
        html.Hr(style={"margin": "28px 0"}),
        saved_runs_panel(),
    ], style={"padding": "24px", "maxWidth": "900px", "margin": "0 auto"})


def saved_runs_panel():
    runs = list_saved_runs()
    if not runs:
        return html.Div([
            h("Saved Results"),
            note("No saved runs yet -- completed pipeline runs are saved automatically and will "
                 "appear here."),
        ])
    rows = []
    for r in runs:
        saved_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["saved_at"]))
        rows.append(html.Div([
            html.Span(f"{r['pricing_date']}  |  {r['n_paths']:,} paths  |  saved {saved_str}",
                      style={"marginRight": "16px"}),
            html.Button("View", id={"type": "load-saved-run-btn", "run_id": r["run_id"]}, n_clicks=0,
                       style={"padding": "4px 14px", "background": STEEL, "color": "white",
                              "border": "none", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px"}),
        ], style={"padding": "8px 0", "borderBottom": "1px solid #e0e6ee",
                  "display": "flex", "alignItems": "center"}))
    return html.Div([
        h("Saved Results"),
        note(f"{len(runs)} completed run(s) saved on disk -- click View to load one without "
             "re-running the pipeline (counterparty scope can still be changed after loading)."),
        html.Div(rows),
    ])


def progress_screen(stage, pct, n_paths):
    return html.Div([
        html.H2("Running pipeline...", style={"color": BLUE}),
        html.Div(id="progress-stage-text", children=stage, style={"fontSize": "14px", "margin": "12px 0 6px"}),
        html.Div([
            html.Div(id="progress-bar-fill", style={"width": f"{pct}%", "background": GREEN, "height": "22px",
                            "borderRadius": "4px", "transition": "width 0.5s"}),
        ], style={"width": "100%", "background": "#e0e6ee", "borderRadius": "4px", "overflow": "hidden"}),
        html.Div(id="progress-pct-text", children=f"{pct:.0f}%", style={"marginTop": "6px", "fontWeight": "bold"}),
        note(f"Exposure -> Greeks -> xVA -> statics, sequentially, at {n_paths:,} paths. This page "
             "auto-refreshes every 3 seconds."),
        dcc.Interval(id="progress-interval", interval=3000, n_intervals=0),
    ], style={"padding": "24px", "maxWidth": "700px", "margin": "80px auto", "textAlign": "center"})


def results_screen(V):
    return html.Div([
        html.Div([
            html.H2("Counterparty Credit Risk, XVA & SA-CCR -- Results", style={"color": BLUE, "margin": "0"}),
            html.Div(f"Valuation date {V['ref_date']}  |  Scope: "
                     f"{'ALL' if V['cptys'] == V['all_cptys'] else V['cptys'][0]}  |  "
                     f"{V['n_paths']:,}-path production run  |  Basel CRE52 SA-CCR",
                     style={"color": "#666", "fontSize": "12px", "marginTop": "4px"}),
            html.Button("New run", id="new-run-btn", n_clicks=0,
                       style={"marginTop": "8px", "padding": "4px 12px", "fontSize": "12px"}),
        ], style={"padding": "16px 24px", "borderBottom": f"3px solid {BLUE}"}),
        dcc.Tabs(id="tabs", value="t1", children=[
            dcc.Tab(label="1. Counterparty Base", value="t1"),
            dcc.Tab(label="2. Trade Base", value="t2"),
            dcc.Tab(label="3. Risk Metrics (Shocks)", value="t3"),
            dcc.Tab(label="4. Base vs Shocked", value="t4"),
            dcc.Tab(label="5. Graphs", value="t5"),
        ]),
        html.Div(id="tab-content", style={"padding": "8px 24px 40px"}),
    ], style={"maxWidth": "1500px", "margin": "0 auto", "fontFamily": "system-ui, sans-serif"})


# ================================================================ APP
app = Dash(__name__, title="Capitolis CCR / XVA / SA-CCR Dashboard", suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Store(id="app-state", data={"screen": "setup", "run_id": None,
                                    "pricing_date": date.today().isoformat(), "cpty_scope": "ALL",
                                    "n_paths": DEFAULT_N_PATHS}),
    html.Div(id="page-root"),
], style={"fontFamily": "system-ui, sans-serif"})


@app.callback(Output("page-root", "children"), Input("app-state", "data"))
def render_page(state):
    screen = state["screen"]
    if screen == "setup":
        return setup_screen(state["pricing_date"], state["cpty_scope"],
                            state.get("n_paths", DEFAULT_N_PATHS), state.get("preview_children"))
    if screen == "progress":
        run = _RUNS.get(state["run_id"], {})
        return progress_screen(run.get("stage", "Starting..."), run.get("pct", 0.0),
                               state.get("n_paths", DEFAULT_N_PATHS))
    if screen == "error":
        run = _RUNS.get(state["run_id"], {})
        return html.Div([
            html.H2("Run failed", style={"color": RED}),
            html.Pre(run.get("error", "Unknown error"), style={"whiteSpace": "pre-wrap", "fontSize": "11px"}),
            html.Button("Back to setup", id="new-run-btn", n_clicks=0),
        ], style={"padding": "24px"})
    if screen == "results":
        run = _RUNS.get(state["run_id"])
        V = _get_view(run, state["cpty_scope"]) if run else None
        if V is None:
            return setup_screen(state["pricing_date"], state["cpty_scope"], state.get("n_paths", DEFAULT_N_PATHS), None)
        return results_screen(V)
    return setup_screen(state["pricing_date"], state["cpty_scope"], state.get("n_paths", DEFAULT_N_PATHS), None)


@app.callback(Output("app-state", "data"), Input("preview-btn", "n_clicks"),
              State("pricing-date-picker", "date"), State("cpty-scope-radio", "value"),
              State("n-paths-dropdown", "value"), State("app-state", "data"), prevent_initial_call=True)
def do_preview(n, pricing_date_str, cpty_scope, n_paths, state):
    if not n:
        return no_update
    pd_date = date.fromisoformat(pricing_date_str)
    trades, dropped = _preview_trades(pd_date)
    scoped = {tid: t for tid, t in trades.items() if cpty_scope == "ALL" or t["counterparty"] == cpty_scope}
    df = pd.DataFrame([{"Trade ID": tid, **t} for tid, t in scoped.items()])
    df = money_cols(df, ["notional"]) if not df.empty else df
    table = dtable(df, "preview-tbl", page=20) if not df.empty else note("No trades in this scope.", RED)
    warn = note(f"Skipped (market data unavailable): {', '.join(dropped)}", RED) if dropped else None
    children = html.Div([
        html.Div(f"{len(scoped)} trade(s) in scope.", style={"fontWeight": "bold", "margin": "6px 0"}),
        warn, table,
    ] if warn else [html.Div(f"{len(scoped)} trade(s) in scope.", style={"fontWeight": "bold", "margin": "6px 0"}), table])
    new_state = dict(state, pricing_date=pricing_date_str, cpty_scope=cpty_scope, n_paths=n_paths,
                     preview_children=children)
    return new_state


@app.callback(Output("app-state", "data", allow_duplicate=True), Input("run-btn", "n_clicks"),
              State("pricing-date-picker", "date"), State("cpty-scope-radio", "value"),
              State("n-paths-dropdown", "value"), State("app-state", "data"), prevent_initial_call=True)
def do_run(n, pricing_date_str, cpty_scope, n_paths, state):
    if not n:
        return no_update
    pd_date = date.fromisoformat(pricing_date_str)
    run_id = _start_run(pd_date, n_paths=n_paths or DEFAULT_N_PATHS)
    return dict(state, screen="progress", run_id=run_id, pricing_date=pricing_date_str,
               cpty_scope=cpty_scope, n_paths=n_paths)


@app.callback(Output("app-state", "data", allow_duplicate=True), Input("progress-interval", "n_intervals"),
              State("app-state", "data"), prevent_initial_call=True)
def poll_progress(n_intervals, state):
    """Handles the RUNNING -> DONE / RUNNING -> ERROR screen transition
    only. Live stage/percent text WHILE running is updated by
    poll_progress_display below instead -- routing that through app-state
    would only update the visible bar on a state CHANGE, but "still
    running" is not a state change (see poll_progress_display's docstring
    for the bug this fixes)."""
    run = _RUNS.get(state["run_id"])
    if run is None:
        return no_update
    if run["status"] == "done":
        return dict(state, screen="results")
    if run["status"] == "error":
        return dict(state, screen="error")
    return no_update


@app.callback(Output("progress-stage-text", "children"), Output("progress-bar-fill", "style"),
              Output("progress-pct-text", "children"),
              Input("progress-interval", "n_intervals"), State("app-state", "data"),
              prevent_initial_call=True)
def poll_progress_display(n_intervals, state):
    """Updates the progress bar/text on EVERY interval tick, independent of
    app-state -- app-state (and therefore the page-root re-render) only
    changes on a SCREEN transition (poll_progress above), so routing live
    percent/stage text through app-state would freeze the bar at its
    initial render: "still running" produces no state change, so
    render_page would never re-fire and _RUNS[run_id]['pct'] advancing
    server-side would never reach the DOM. This callback targets the
    progress screen's own elements directly instead, so it updates every
    3s regardless of whether the overall screen state has changed."""
    run = _RUNS.get(state.get("run_id"))
    if run is None or run["status"] != "running":
        return no_update, no_update, no_update
    pct = run.get("pct", 0.0)
    style = {"width": f"{pct}%", "background": GREEN, "height": "22px",
             "borderRadius": "4px", "transition": "width 0.5s"}
    return run.get("stage", ""), style, f"{pct:.0f}%"


@app.callback(Output("app-state", "data", allow_duplicate=True), Input("new-run-btn", "n_clicks"),
              State("app-state", "data"), prevent_initial_call=True)
def back_to_setup(n, state):
    if not n:
        return no_update
    return dict(state, screen="setup", run_id=None, preview_children=None)


@app.callback(Output("app-state", "data", allow_duplicate=True),
              Input({"type": "load-saved-run-btn", "run_id": ALL}, "n_clicks"),
              State("app-state", "data"), prevent_initial_call=True)
def load_saved_run(n_clicks_list, state):
    """One pattern-matching callback handles every 'View' button on the
    saved-runs panel (their number varies with how many runs are saved, so
    a fixed Input per button isn't possible -- see saved_runs_panel's
    dict-shaped component ids). ctx.triggered_id identifies WHICH button
    fired; a fresh page load or a re-render of the panel (e.g. after
    another run completes) also fires this callback with all-None clicks,
    which must be a no-op, not a spurious load of run_id=None."""
    triggered = ctx.triggered_id
    if not triggered or not any(n_clicks_list):
        return no_update
    run_id = triggered["run_id"]
    if not _load_saved_run(run_id):
        return no_update
    run = _RUNS[run_id]
    view = run["saved_view_all_scope"]
    return dict(state, screen="results", run_id=run_id,
               pricing_date=view["ref_date"], cpty_scope="ALL", n_paths=view["n_paths"])


@app.callback(Output("tab-content", "children"), Input("tabs", "value"), State("app-state", "data"))
def render_tab(which, state):
    run = _RUNS.get(state["run_id"])
    V = _get_view(run, state["cpty_scope"]) if run else None
    if V is None:
        return note("No results loaded.", RED)
    return TABS[which](V)


@app.callback(Output("t3-table", "children"), Input("t3-shock", "value"), State("app-state", "data"))
def t3_filter(sel, state):
    run = _RUNS.get(state["run_id"])
    V = _get_view(run, state["cpty_scope"]) if run else None
    if V is None:
        return no_update
    df = risk_metric_rows(V)[_RISK_COLS]
    if sel != "ALL":
        st, rf = sel.split("|")
        df = df[(df["Shock Type"] == st) & (df["Risk Factor"] == rf)]
    return dtable(money_cols(df, _RISK_MONEY), "t3tbl2", page=40)


server = app.server  # WSGI entry point (gunicorn/waitress: `dashboard:server`)


def _lan_ip():
    """Best-effort primary LAN IPv4 (no traffic actually sent)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CCR / XVA / SA-CCR dashboard")
    ap.add_argument("--host", default=os.environ.get("DASH_HOST", "0.0.0.0"),
                    help="bind address (default 0.0.0.0 = reachable from the LAN; "
                         "use 127.0.0.1 for local-only)")
    ap.add_argument("--port", type=int, default=int(os.environ.get("DASH_PORT", "8050")))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.host in ("0.0.0.0", "::"):
        ip = _lan_ip()
        print("Dashboard reachable on this LAN at:")
        print(f"    http://{ip}:{args.port}      <- share this with colleagues on the same network")
        print(f"    http://127.0.0.1:{args.port}   (this machine)")
        print("No authentication -- anyone on the network who has the URL can view the data.")
    else:
        print(f"Dashboard at http://{args.host}:{args.port}")
    print("Ctrl+C to stop.\n")
    app.run(debug=args.debug, host=args.host, port=args.port)
