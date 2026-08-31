"""
Dates, day-count fractions, and schedules -- Python standard library only.

Deliberately minimal: no holiday calendars (weekend/holiday adjustment is an
extension). Dates are `datetime.date`. This is enough to reproduce the
discounted-cash-flow maths of the four products to the sixth decimal against a
reference implementation.

Methodology mirrors QuantLib / ORE (the Open Source Risk Engine); this is a
standalone re-implementation, not a wrapper -- there is no third-party
dependency.
"""

from datetime import date
import calendar

# ------------------------------------------------------------------ parsing
def to_date(x):
    """Accept a date, ISO 'YYYY-MM-DD', or US 'MM/DD/YYYY' (also 'M/D/YYYY') string."""
    if isinstance(x, date):
        return x
    s = str(x).strip()
    if "/" in s:                       # US format: month/day/year
        m, d, y = (int(p) for p in s.split("/"))
        if y < 100:                    # 2-digit year -> 2000s
            y += 2000
        return date(y, m, d)
    return date.fromisoformat(s)       # ISO year-month-day


# ------------------------------------------------------------- day counts
def year_fraction(d1, d2, basis="ACT/365F"):
    d1, d2 = to_date(d1), to_date(d2)
    b = basis.upper().replace(" ", "")
    if b in ("ACT/365F", "ACT/365", "A365F"):
        return (d2 - d1).days / 365.0
    if b in ("ACT/360", "A360"):
        return (d2 - d1).days / 360.0
    if b in ("30/360", "30U/360", "30/360US", "BONDBASIS", "30/360BONDBASIS"):
        dd1, dd2 = d1.day, d2.day
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31 and dd1 == 30:
            dd2 = 30
        return (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)) / 360.0
    if b in ("30E/360", "30E/360ISDA"):
        dd1 = 30 if d1.day == 31 else d1.day
        dd2 = 30 if d2.day == 31 else d2.day
        return (360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (dd2 - dd1)) / 360.0
    raise ValueError(f"Unknown day-count basis: {basis}")


# ------------------------------------------------------------- date maths
def add_months(d, n):
    """Add n calendar months, clamping the day to the target month's length."""
    d = to_date(d)
    m0 = d.month - 1 + n
    y = d.year + m0 // 12
    m = m0 % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def schedule_forward(start, end, months):
    """Reset/coupon dates from `start` forward to `end` (short final stub).
    months <= 0 means a single period (no intermediate resets): [start, end]."""
    start, end = to_date(start), to_date(end)
    if months <= 0:
        return [start, end]
    dates, d, k = [start], start, 1
    while True:
        nxt = add_months(start, months * k)
        if nxt >= end:
            break
        dates.append(nxt)
        k += 1
    dates.append(end)
    return dates


def schedule_backward(issue, maturity, months):
    """Bond coupon period boundaries from `maturity` backward to `issue`."""
    issue, maturity = to_date(issue), to_date(maturity)
    dates, k = [maturity], 1
    while True:
        prev = add_months(maturity, -months * k)
        if prev <= issue:
            break
        dates.append(prev)
        k += 1
    dates.append(issue)
    return sorted(dates)
