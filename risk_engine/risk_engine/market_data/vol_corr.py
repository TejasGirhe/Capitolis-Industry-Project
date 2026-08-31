"""
Realized volatility and correlation from historical daily price history --
the free-data substitute for implied vol surfaces and cross-asset
correlations, which are paid data products (option chains / a vol-surface
vendor) not available from any free source. MARKET_DATA.md Sec.5 itself
says of dividend yield "source it as you like (implied, put-call
parity...)"; realized vol/correlation is the analogous free-data choice for
the volatility and correlation inputs.

This gives REALIZED (backward-looking, historical) vol/correlation, not
IMPLIED (forward-looking, market-priced) vol/correlation -- a real,
documented difference, not just a data-quality footnote: implied vol prices
in the market's expectation of future risk (including event risk, skew
demand), realized vol only reflects what already happened. Prefer implied
data if you have access to it.

Where this plugs in:
    realized_vol(...)               -> risk_engine.calibration.market_surface.
                                        flat_vol_surface(factor_key, flat_vol=...)
                                        (a single-point term structure -- realized
                                        vol from one price history has no natural
                                        tenor/strike axis, so it feeds the flat/ATM
                                        entry point, not the extended smile surface)
    realized_correlation_matrix(...) -> capitolis_pricers.market.MarketState(
                                        correlations={...}), keyed by
                                        str(RiskFactor) pairs exactly as
                                        risk_engine.simulation.joint.JointSimulator
                                        already expects (see joint.py's
                                        MarketState.correlation(str(a), str(b)) lookup)
"""
import math
from typing import Dict, List

TRADING_DAYS_PER_YEAR = 252


def _log_returns(closes: List[float]) -> List[float]:
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def realized_vol(closes: List[float], annualize: bool = True) -> float:
    """Annualized realized volatility from a daily close-price history
    (population stdev of log returns, scaled by sqrt(252) if annualize)."""
    rets = _log_returns(closes)
    if len(rets) < 2:
        raise ValueError("need at least 2 return observations to compute realized vol")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    vol = math.sqrt(var)
    return vol * math.sqrt(TRADING_DAYS_PER_YEAR) if annualize else vol


def realized_correlation_matrix(price_histories: Dict[str, List[float]]) -> Dict:
    """price_histories: {factor_key: [close, close, ...]} -- histories should
    share the same trading-day calendar (same length/dates); this function
    truncates every series to the shortest one and aligns from the end
    (most recent N observations) rather than requiring the caller to
    pre-align, since real fetches can return slightly different bar counts
    per ticker (holidays, listing gaps).

    Returns {(factor_key_a, factor_key_b): correlation} for every pair,
    a-then-b in the SAME iteration order as price_histories.keys() --
    directly usable as capitolis_pricers.market.MarketState(correlations=...).
    """
    keys = list(price_histories.keys())
    min_len = min(len(v) for v in price_histories.values())
    if min_len < 3:
        raise ValueError("need at least 3 overlapping observations to compute correlation")

    returns = {}
    for k in keys:
        closes = price_histories[k][-min_len:]
        returns[k] = _log_returns(closes)

    out = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            out[(a, b)] = _pearson(returns[a], returns[b])
    return out


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(vx * vy)
    return cov / denom if denom > 0 else 0.0
