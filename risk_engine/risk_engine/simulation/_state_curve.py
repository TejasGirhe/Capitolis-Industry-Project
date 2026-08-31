"""
A Curve-compatible object backed by a simulated rate state, so pricers can
call `.discount(d)` on a scenario exactly as they do on a real
capitolis_pricers.curves.Curve, without knowing anything is simulated.
"""
from datetime import timedelta


class StateImpliedCurve:
    """discount(d) = CalibratedRateModel.discount_factor(state, t_node, T),
    i.e. the discount factor from the scenario's valuation time t_node out to
    d, read off the model's own closed-form DF formula on this path's state.
    ref_date is the SCENARIO date (t_node), not the original t=0 valuation
    date -- matches how a real Curve is rebuilt "as of" the pricing date in a
    Monte Carlo engine (README Sec.6: "rebuild a Curve per simulated node").
    """

    def __init__(self, calibrated_rate_model, state, t_node: float, ref_date):
        self._model = calibrated_rate_model
        self._state = state
        self._t_node = t_node
        self.ref_date = ref_date

    def discount(self, d):
        T = self._t_node + _year_frac(self.ref_date, d)
        # Pass the real dates through directly for both T and t_node --
        # avoids discount_factor's shift_date() round-trip (year_frac then
        # back to a date), which loses up to 0.5 days of precision and
        # showed up as ~$1 noise on ~$300k+ notionals in end-to-end pricing
        # tests. self.ref_date IS the real date corresponding to t_node
        # (both were derived from the same scenario valuation date in
        # build_market_states_at).
        return self._model.discount_factor(self._state, self._t_node, T, T_date=d, t_date=self.ref_date)

    def zero_rate(self, d, comp="continuous"):
        import math
        t = _year_frac(self.ref_date, d)
        if t <= 0:
            return 0.0
        return -math.log(self.discount(d)) / t


def _year_frac(ref_date, d):
    return (d - ref_date).days / 365.0
