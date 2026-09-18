"""
Wraps the production pipeline scripts (exposure_profile.py, greeks_report.py,
xva_report.py, report_inputs.py) into ONE function the interactive dashboard
calls for a given pricing date -- this module does not duplicate any of
their pipeline logic, it calls their existing main()/compute_*() entry
points in sequence and collects what they return.

This is intentionally SEQUENTIAL (book load -> exposure -> greeks -> xva ->
statics), not parallel -- matching this project's own established practice
of not running multiple full-book Monte Carlo pipelines concurrently (CPU
contention distorted timing badly earlier in this project whenever that
was tried).

Progress is reported via a callback: progress_cb(stage: str, pct: float)
so the dashboard's background thread can update a shared status dict the
UI polls -- no Dash dependency in this module, so it stays usable
standalone (`python dashboard_runner.py --pricing-date ... --n-paths ...`).
"""
import os
import sys
from datetime import date

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRICERS_ROOT = os.path.join(ROOT, "capitolis_pricers", "capitolis_pricers")
for p in (os.path.join(ROOT, "risk_engine"), PRICERS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def _noop_progress(stage, pct):
    pass


def run_full_pipeline(pricing_date: date, n_paths: int = 10_000, n_workers=None, progress_cb=None):
    """Runs exposure -> greeks -> xva -> report-statics sequentially for
    pricing_date, at n_paths. Returns a dict:

        ref_date            isoformat string
        report_inputs       report_inputs.compute_report_inputs()'s payload
                             (per-trade/per-counterparty statics, SA-CCR
                             RC/PFE/EAD, DV01) -- SAME function
                             report_inputs.py's CLI uses, not reimplemented
        profiles            {counterparty_id: ExposureProfile}, +BOOK_TOTAL
        greeks_report        risk_engine.greeks.report.GreeksReport
        xva_by_cpty          {counterparty_id: XVAResult}
        trades               {trade_id: Pricer} -- the exact trade set priced
                             (already excludes any name whose market data
                             failed to fetch, per _sourced_book's own
                             graceful-degradation logic)

    progress_cb(stage: str, pct: float in [0,100]): called as each stage
    starts, so a caller (e.g. a background thread) can report live status
    to a polling UI. None (default) -> no-op.
    """
    progress_cb = progress_cb or _noop_progress

    progress_cb("Loading book & sourcing market data", 2)
    import risk_engine.examples.exposure_profile as exposure_profile
    import risk_engine.examples.greeks_report as greeks_report
    import risk_engine.examples.xva_report as xva_report
    import risk_engine.examples.report_inputs as report_inputs
    from risk_engine.examples._sourced_book import build_sourced_book

    # exposure_profile.main()/greeks_report.main()/xva_report.main() each
    # call build_sourced_book(ref_date=pricing_date) independently (their
    # existing, established design -- not changed here). Market data is
    # sourced live per call (see _sourced_book's REF_DATE handling), so
    # these three loads are not guaranteed byte-identical if run across a
    # day boundary, but are otherwise the same book each time.
    progress_cb("Running exposure profile (EE/PFE/MPE/EEPE)", 10)
    profiles, exposure_png_path = exposure_profile.main(n_paths=n_paths, n_workers=n_workers, pricing_date=pricing_date)

    progress_cb("Running Greeks (SA-CCR delta + vega)", 45)
    greeks_report_obj = greeks_report.main(n_paths=n_paths, n_workers=n_workers, pricing_date=pricing_date)

    progress_cb("Running xVA (CVA/DVA/FVA)", 80)
    xva_by_cpty = xva_report.main(n_paths=n_paths, n_workers=n_workers, pricing_date=pricing_date)

    progress_cb("Building counterparty/trade statics (report_inputs)", 92)
    # Fresh book load for statics -- report_inputs.compute_report_inputs
    # needs its OWN book object (grid/market_data/etc.), and reusing one of
    # the three above would couple this function to their internals; the
    # extra load is cheap relative to the three Monte Carlo runs already done.
    # No restrict_to_trade_ids filter needed: this fresh load already
    # reflects whatever names _sourced_book's own market-data-availability
    # check dropped for THIS pricing_date, consistent with what the three
    # runs above just priced.
    statics_book = build_sourced_book(ref_date=pricing_date)
    report_inputs_payload = report_inputs.compute_report_inputs(statics_book)

    progress_cb("Done", 100)

    return {
        "ref_date": pricing_date.isoformat(),
        "n_paths": n_paths,
        "report_inputs": report_inputs_payload,
        "profiles": profiles,
        "greeks_report": greeks_report_obj,
        "xva_by_cpty": xva_by_cpty,
        "trades": statics_book["trades"],
    }


if __name__ == "__main__":
    import argparse
    import uuid
    ap = argparse.ArgumentParser()
    ap.add_argument("--pricing-date", type=date.fromisoformat, default=None)
    ap.add_argument("--n-paths", type=int, default=10_000)
    ap.add_argument("--no-save", action="store_true", help="skip saving the result for the dashboard's 'view saved results' picker")
    args = ap.parse_args()
    pricing_date = args.pricing_date or date.today()

    def _print_progress(stage, pct):
        print(f"[{pct:5.1f}%] {stage}")

    result = run_full_pipeline(pricing_date, n_paths=args.n_paths, progress_cb=_print_progress)
    print("Pipeline complete.")

    if not args.no_save:
        # Save this terminal-launched run the SAME way dashboard.py's own
        # "Run pipeline" button does, so it shows up in the dashboard's
        # "Saved Results" picker too -- a run started from the terminal and
        # one started from the UI are otherwise identical artifacts.
        import sys
        sys.path.insert(0, os.path.dirname(__file__))
        import dashboard
        run_id = uuid.uuid4().hex
        view_all = dashboard.build_view(result, "ALL")
        dashboard._save_run_to_disk(run_id, pricing_date.isoformat(), args.n_paths, view_all)
        print(f"Saved run {run_id} -- open the dashboard and click 'View' under Saved Results to see it, "
              f"or read {dashboard._saved_run_path(run_id)} directly.")
