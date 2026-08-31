"""
price_curves: for every trade, every path, every regression date d, produce
two NPVs -- NPV0 (value AT d) and NPV10 (the MPoR curve: value at d + 10
business days) -- both read off ONE precached simulation (the fixed
simulation grid) via each model's state_at() bridge interpolation
(risk_engine.simulation.interpolate), never a fresh Monte Carlo draw at d or
d+10bd. This is the "2 curves" of the 3-stage design: NPV0 and NPV10, built
from one precache rather than two separately-simulated grids.

capitolis_pricers has no batch/vectorized pricing API, so pricing a single
node is a plain Python loop -- price_curves parallelizes across PATHS via
multiprocessing.Pool, same mechanism as before: paths are fully independent
(each is its own Monte Carlo draw with no cross-path state), so splitting
work by path chunk is exact, not an approximation.

Maturity guard: none of the three provided pricers check market.ref_date
against their own maturity (EquityTRS.end_date, BondForwardTrade.forward_date,
BondTRS.end_date) -- they'll happily discount a settled contract as if it
were still live on any later valuation date (confirmed directly: pricing
BF_0003, forward_date 2026-12-06, on a 2028-06-15 MarketState returns a large
nonzero NPV). Left unguarded, a trade keeps contributing full exposure for
every regression date past its own maturity, which silently inflates EE/PFE
long after the trade has rolled off. price_curves therefore excludes a trade
from every regression date whose date is past that trade's own maturity --
this is the exposure engine's responsibility (the provided pricers are
correctly scoped to "reprice on one snapshot", not "know whether that
snapshot postdates settlement").
"""
import os
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Dict, List

from .simulation.grid import add_business_days
from .simulation.scenario_market import build_market_states_at


def trade_maturity(trade):
    for attr in ("end_date", "forward_date"):
        if hasattr(trade, attr):
            return getattr(trade, attr)
    raise AttributeError(f"{type(trade).__name__} has no recognized maturity attribute "
                          f"(end_date or forward_date)")


@dataclass
class CurveResult:
    npv0: Dict    # {(path_index, date, trade_id): npv}  -- value AT `date`
    npv10: Dict   # {(path_index, date, trade_id): npv}  -- value AT `date + mpor_days` business days
    n_paths: int
    regression_dates: List
    trade_ids: List[str]
    trade_maturities: Dict[str, object]
    mpor_days: int

    def mean_npv0_by_trade(self, at_date):
        out = {}
        for tid in self.trade_ids:
            out[tid] = sum(self.npv0[(p, at_date, tid)] for p in range(self.n_paths)) / self.n_paths
        return out


def _price_path_chunk(args):
    (trades, joint_result_chunk, global_path_ids, regression_dates, mpor_days,
     equity_dividend_rates, reporting, reporting_ccy) = args
    maturities = {tid: trade_maturity(trade) for tid, trade in trades.items()}
    sim_times = joint_result_chunk.sim_times

    npv0, npv10 = {}, {}
    for d in regression_dates:
        d10 = add_business_days(d, mpor_days)
        live0 = {tid: t for tid, t in trades.items() if d <= maturities[tid]}
        live10 = {tid: t for tid, t in trades.items() if d10 <= maturities[tid]}

        markets0 = (build_market_states_at(joint_result_chunk, sim_times, d, equity_dividend_rates, reporting_ccy)
                    if live0 else None)
        markets10 = (build_market_states_at(joint_result_chunk, sim_times, d10, equity_dividend_rates, reporting_ccy)
                     if live10 else None)

        for local_p, global_p in enumerate(global_path_ids):
            for tid, trade in trades.items():
                npv0[(global_p, d, tid)] = trade.npv(markets0[local_p], reporting=reporting) if tid in live0 else 0.0
                npv10[(global_p, d, tid)] = trade.npv(markets10[local_p], reporting=reporting) if tid in live10 else 0.0
    return npv0, npv10


def _chunk(seq, n_chunks):
    n_chunks = max(1, min(n_chunks, len(seq)))
    size = -(-len(seq) // n_chunks)  # ceil division
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def price_curves(trades: Dict[str, object], precache, regression_dates: List, equity_dividend_rates: dict,
                  mpor_days: int = 10, reporting: bool = True, reporting_ccy: str = "USD",
                  n_workers: int = None) -> CurveResult:
    """trades: merged {trade_id: Pricer} dict across all instrument types.
    precache: risk_engine.simulation.joint.JointSimResult from ONE simulate()
        call on the FIXED simulation grid (risk_engine.simulation.grid.build_simulation_grid) --
        never re-simulated per regression date.
    regression_dates: risk_engine.simulation.grid.collect_regression_dates(...) output --
        every date pricing actually needs a value at (trade cashflow dates +
        MPoR window endpoints for every reporting anchor).

    n_workers: number of processes to price path-chunks in parallel.
        None (default) -> os.cpu_count(). 1 -> single-process, no pool
        spun up at all (used by the test suite for fast, deterministic,
        low-overhead runs on small n_paths).
    """
    n_paths = next(iter(precache.rate_states.values())).shape[0]
    path_ids = list(range(n_paths))
    trade_ids = list(trades.keys())

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    if n_workers <= 1:
        npv0, npv10 = _price_path_chunk(
            (trades, precache, path_ids, regression_dates, mpor_days, equity_dividend_rates, reporting, reporting_ccy))
    else:
        chunks = _chunk(path_ids, n_workers)
        # slice_paths keeps each worker's IPC payload to just its own path
        # chunk -- see JointSimResult.slice_paths docstring (Windows named-pipe
        # resource limit at full path counts otherwise).
        args = [(trades, precache.slice_paths(chunk), chunk, regression_dates, mpor_days,
                  equity_dividend_rates, reporting, reporting_ccy) for chunk in chunks]
        ctx = get_context("spawn")
        with ctx.Pool(processes=len(chunks)) as pool:
            results = pool.map(_price_path_chunk, args)
        npv0, npv10 = {}, {}
        for r0, r10 in results:
            npv0.update(r0)
            npv10.update(r10)

    trade_maturities = {tid: trade_maturity(trade) for tid, trade in trades.items()}
    return CurveResult(npv0=npv0, npv10=npv10, n_paths=n_paths, regression_dates=list(regression_dates),
                        trade_ids=trade_ids, trade_maturities=trade_maturities, mpor_days=mpor_days)


def price_curves_per_trade(trades: Dict[str, object], precache_by_trade_id: Dict[str, object],
                            regression_dates: List, equity_dividend_rates: dict, mpor_days: int = 10,
                            reporting: bool = True, reporting_ccy: str = "USD",
                            n_workers: int = None) -> Dict[str, CurveResult]:
    """Per-trade analogue of price_curves, for risk_engine.simulation.per_trade's
    INDEPENDENT (uncorrelated across trades) simulation mode.

    precache_by_trade_id: {trade_id: JointSimResult} -- one INDEPENDENT
        simulation per trade (risk_engine.simulation.per_trade.
        simulate_all_trades_independently), each with its own path-index
        space unrelated to any other trade's.

    Returns {trade_id: CurveResult} -- one CurveResult PER TRADE, each with
    n_paths of its own, rather than one merged CurveResult with a shared
    path index across trades. This is deliberate: price_curves' single-
    CurveResult design assumes every trade shares one path-index space (see
    that function's docstring and netting.NettingSet.netted_npv, which sums
    across trades at a fixed path index) -- merging independently-simulated
    trades into one CurveResult would silently misrepresent uncorrelated
    scenarios as if they were the same joint scenario. Each trade's
    CurveResult here has n_paths=1 trade in its trade_ids, so calling
    plain compute_exposure_profile on a single-trade NettingSet against
    each of these independently is safe (see exposure.compute_per_trade_profiles).
    """
    return {
        tid: price_curves({tid: trade}, precache_by_trade_id[tid], regression_dates,
                           equity_dividend_rates, mpor_days, reporting, reporting_ccy, n_workers)
        for tid, trade in trades.items()
    }
