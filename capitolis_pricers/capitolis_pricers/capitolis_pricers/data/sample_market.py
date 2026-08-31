"""
Illustrative MarketState built in code (sample values) so the example runs.
Students build their own MarketState from collected data -- see MARKET_DATA.md
for the exact fields. Only a USD rate curve and the JPYUSD FX spot are needed to
PRICE (the compo JPY leg grows on the USD curve; JPY discounting is implied via
FX forwards, so no separate JPY rate curve).
"""
from ..market import MarketState
from ..curves import zero_curve, FxCurve

REF_DATE = "2026-01-15"


def sample_market(ref_date=REF_DATE):
    usd = zero_curve(ref_date, [0.25, 0.5, 1, 2, 3, 5, 7, 10],
                     [0.0435, 0.0430, 0.0420, 0.0405, 0.0398, 0.0395, 0.0400, 0.0410])
    fx = FxCurve("USD", "JPY", 150.0, usd)          # spot only needed to price
    return MarketState(
        ref_date=ref_date, reporting_ccy="USD",
        discount_curves={"USD": usd},
        equity_spots={"US88579Y1010": 100.0, "JP3633400001": 3000.0, "JP3435000009": 1500.0},
        equity_dividend_rates={"US88579Y1010": 0.015, "JP3633400001": 0.020, "JP3435000009": 0.020},
        fx_curves={("USD", "JPY"): fx},
        equity_vols={"USEQ": 0.22, "JPEQ": 0.20, "JPEQ2": 0.24},   # engine input
        correlations={("JP3633400001", "USDJPY"): -0.30},                  # engine input
    )
