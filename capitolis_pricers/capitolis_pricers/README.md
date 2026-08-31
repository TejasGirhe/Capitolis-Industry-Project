# Capitolis Pricing Models

Starter **pricing models** and their inputs for four instruments:

- USD Equity Total Return Swap
- JPY Equity Total Return Swap (compo)
- Bond Forward
- Bond Total Return Swap

**Standalone, no dependencies.** Self-contained re-implementation of the
discounted-cash-flow methodology used by QuantLib / ORE (the Open Source Risk
Engine); Python standard library only.

---

## 1. Three data layers (who owns what)

| Layer | Folder | Owner | What it is |
|---|---|---|---|
| **Trades** | `trade_data/` | Capitolis | the deals (`equity_trs.csv`, `bond_forward.csv`, `bond_trs.csv`) |
| **Underlyings** | `trade_data/underlyings/` | Capitolis | the reference legs: `equities.csv` (per-trade basket + identity, by `trade_id`) and `bonds.csv` (bond definitions, by `bond_id`) |
| **Market data** | (spec) | **students collect** | the risk-factor inputs — curves, spots, FX, vols, correlations. Not templated; the fields are specified in `MARKET_DATA.md`. |

```bash
python -m capitolis_pricers.examples.price_all
```

```python
from capitolis_pricers.data.sample_market import sample_market
from capitolis_pricers.underlyings_loader import load_equities
from capitolis_pricers.trade_loader import load_equity_trs

market  = sample_market()                                    # students build theirs (MARKET_DATA.md)
baskets = load_equities("trade_data/underlyings/equities.csv")  # basket by trade_id
trades  = load_equity_trs("trade_data/equity_trs.csv", baskets)
for tid, p in trades.items():
    print(tid, p.npv(market, reporting=True))
```

## 2. The pricing contract

Every pricer is a pure function of a `MarketState`:

```python
npv = pricer.npv(market)                  # NPV in the trade's currency
npv = pricer.npv(market, reporting=True)  # NPV in market.reporting_ccy (spot FX)
```

To build exposure, your engine simulates risk factors, assembles a
`MarketState` per (scenario, horizon) node, and calls `pricer.npv(node_market)`.

## 3. What each pricer needs -> where it comes from

### Equity TRS

```
equity leg  = sum_i shares_i * DF(end) * (F_i(end) - basis_i)
F(t)        = S_ccy / DF(end)                     (total-return fwd, grows at r)
funding leg = sum rate_i * N * tau * DF(t_{i+1})
    rate_i  = SOFR_fwd + spread                   (fixed_rate blank -> floating)
            = fixed_rate                          (fixed_rate given -> fixed)
NPV(receiver) = equity leg - funding leg
```

Position is **shares + basis** per name (notional weight is derived). Funding is
**floating** (`SOFR_fwd + spread`) unless a `fixed_rate` is given. **Dividends:**
100% pass-through assumed, so the equity leg is the total return and the forward
grows at `r`; the dividend rate is a market input for the *simulation* drift
`(r - q)`, not used by the pricer.

### Bond Forward
`NPV = sign * (notional/par) * DF(T) * (forward_clean(T) - strike_clean)`.
The forward clean price is the spot financed at **`repo_rate`** (ACT/360, less
coupons in the window) when given, else curve-derived (used in simulation).

### Bond TRS
Return leg (bond price change + coupons) + funding leg over the reset schedule,
scaled by `notional`/par. Funding is floating (`SOFR_fwd + spread`) or
`fixed_rate`. Bonds are **risk-free** (government / agency), discounted on the
USD curve (issuer credit is the optional extension, §8).

## 4. The JPY trade — compo

The JPY Equity TRS is **compo**: both legs pay USD and the equity leg pays the
return of the USD value of the JPY share (`S_JPY x FX`), so equity and FX both
flow through. Set `trade_ccy = USD` on a JPY-quoted position; the pricer
converts the spot at FX and grows it on the USD curve. `basis` for a compo name
is the locked **USD** level at inception (`S_JPY x FX` at that date).

## 5. Input data — every field explained

The trade inputs live in `trade_data/` (Capitolis supplies them). Below is every
column, its format, whether it's required, and what it's for. (Market data —
spots, curves, FX — is collected separately per `MARKET_DATA.md`.)

### Formatting rules (read first)

- **Plain numbers only** — no `$`, no thousands commas, no `%`. Write `126.22`
  (not `$126.22`) and `50000000` (not `50,000,000`).
- **Rates/spreads/coupons/dividends/repo are decimals**, not basis points or
  percent: `0.0050` = 50 bp, `0.042` = 4.2%.
- **Dates:** `MM/DD/YYYY` or ISO `YYYY-MM-DD` (both accepted; slash dates are
  read US month-first).
- **Notionals and shares are positive.** The payer/receiver (or long/short) side
  is set by `direction`/`position` — a negative notional is rejected. Map a
  report's negative short quantity to a **positive** notional plus the right
  `direction` (`pay_*`) or `position` (`short`).
- **Blank means "use the default" — don't type `0`.** In particular, leave
  `fixed_rate` **blank** on a floating leg; a literal `0` books a 0% fixed leg.
- `direction` and `position` are **case-insensitive**; an invalid value errors
  (it won't silently flip the side). `trade_id` is trimmed, case-sensitive,
  **unique**, and must **match exactly** between `equity_trs.csv` and
  `equities.csv`.

### `equity_trs.csv` — the swap deal (one row per trade)

| Field | Req? | Format | Purpose |
|---|---|---|---|
| `trade_id` | required | text, unique | the swap id; links to its basket in `equities.csv` |
| `direction` | required | `pay_equity` / `receive_equity` | side; `pay_equity` = CLGM pays the total return, receives funding |
| `trade_ccy` | required | ccy code (`USD`) | settlement currency (USD, incl. compo JPY) |
| `start_date` | required | date | financing / effective start (≈ settlement) |
| `end_date` | required | date | financing end (swap maturity) |
| `reset_frequency_months` | optional | integer; `0`/blank = single period | **funding-leg (SOFR) reset / payment** frequency |
| `funding_index` | optional | label (`SOFR`); blank ⇒ fixed | floating index label (not used in pricing) |
| `spread` | floating leg | decimal | spread over SOFR; blank on a fixed leg |
| `fixed_rate` | fixed leg | decimal | fixed funding rate; **filling it makes the leg fixed**; blank ⇒ floating |
| `funding_notional` | optional | positive number; blank ⇒ Σ shares×basis | financed amount |
| `counterparty` | required | text | netting-set key (groups trades by counterparty in the simulation) |

### `trade_data/underlyings/equities.csv` — the basket (rows keyed by `trade_id`)

One row per name; single-name TRS = one row, basket = several.

| Field | Req? | Format | Purpose |
|---|---|---|---|
| `trade_id` | required | text | links this basket row to its deal in `equity_trs.csv` |
| `ticker` | optional | text | identifier (mapping only) |
| `isin` | required | text | **security identifier that keys the market data** (spot, dividend) |
| `cusip` | optional | text | identifier (mapping only) |
| `currency` | required | ccy code | the name's **native** quote ccy (`JPY` for compo); drives the compo FX conversion |
| `shares` | required | number | position size per name; positive (side set by `direction`). May be signed to mix long/short names in one basket |
| `basis` | optional | price per share in **`trade_ccy`** (USD) | cost basis / strike. Compo: the **USD** level at inception. Blank ⇒ struck at today's spot |

### `bond_forward.csv`

| Field | Req? | Format | Purpose |
|---|---|---|---|
| `trade_id` | required | text, unique | the forward's id |
| `trade_ccy` | required | ccy code | settlement currency |
| `bond_id` | required | text | references a row in `bonds.csv` |
| `position` | required | `long` / `short` | side |
| `forward_date` | required | date | settlement / delivery date |
| `strike_clean` | required | price per 100 (e.g. `100`) | agreed forward clean price |
| `notional` | required | positive number (face amount) | face amount of the bond |
| `repo_rate` | optional | decimal, ACT/360; blank ⇒ curve carry | financing rate for the forward |
| `counterparty` | required | text | netting-set key |

### `bond_trs.csv`

| Field | Req? | Format | Purpose |
|---|---|---|---|
| `trade_id` | required | text, unique | the swap id |
| `trade_ccy` | required | ccy code | settlement currency |
| `bond_id` | required | text | references a row in `bonds.csv` |
| `direction` | required | `pay_tr` / `receive_tr` | side; `pay_tr` = CLGM pays the total return, receives funding |
| `notional` | required | positive number (face amount) | face amount |
| `start_date` / `end_date` | required | date | financing start / end |
| `reset_frequency_months` | optional | integer; `0`/blank = single period | funding-leg reset / payment frequency |
| `funding_index` | optional | label; blank ⇒ fixed | floating index label (not priced) |
| `spread` | floating leg | decimal | spread over SOFR; blank on a fixed leg |
| `fixed_rate` | fixed leg | decimal | fixed funding rate; blank ⇒ floating |
| `funding_notional` | optional | positive; blank ⇒ initial dirty market value | financed amount |
| `counterparty` | required | text | netting-set key |

### `trade_data/underlyings/bonds.csv` — bond definitions (keyed by `bond_id`)

| Field | Req? | Format | Purpose |
|---|---|---|---|
| `bond_id` | required | text, unique | key referenced by the bond trades |
| `isin` | optional | text | identifier (mapping) |
| `cusip` | optional | text | identifier (mapping) |
| `name` | optional | text | description |
| `issuer` | optional | text | credit key for the **optional** risky-bond extension (§8); blank/unmatched ⇒ risk-free |
| `currency` | optional | ccy code (default `USD`) | bond currency |
| `issue_date` / `maturity_date` | required | date | bond schedule bounds |
| `coupon` | required | decimal (`0.0388` = 3.88%) | annual coupon rate |
| `coupon_frequency_months` | required | integer (`6` = semiannual) | coupon frequency |
| `day_count` | required | `30/360`, `30E/360`, `ACT/365F`, `ACT/360` | accrual convention |
| `par` | optional | number (default `100`) | price convention |

**Mapping.** Bonds map *up* (a bond trade references one `bond_id`); equities map
*down* (a TRS `trade_id` fans out to its basket in `equities.csv`).

## 6. Curves, FX, and MarketState

Curve builders are pure Python and take plain numbers, so you rebuild them per
simulated node (or shift the nodes) inside your engine.

```python
from capitolis_pricers import flat_curve, zero_curve, FxCurve
usd = zero_curve("2026-01-15", [0.5,1,2,5,10], [0.0430,0.0420,0.0405,0.0395,0.0410])
usd.discount("2027-01-15")
fx  = FxCurve("USD","JPY",150.0, usd)      # 150 JPY per USD (spot only needed to price)
```

`Curve` is a discount-factor curve (log-linear on DFs; ACT/365F time). Build the
`MarketState` from your collected snapshot (see `sample_market.py`). To reprice
on a scenario, build a new `MarketState` and call `pricer.npv(...)` again. To
**price** you need only the USD curve, equity spots, and the FX spot; to
**simulate** you need the full pillars, dividend rates, FX forwards,
volatilities, and correlations (see `MARKET_DATA.md`).

## 7. Risk factors per trade (simulation scope)

| Trade | Rates | Equity | FX |
|---|---|---|---|
| USD Equity TRS | USD | equity spot (drift uses dividend rate) | — |
| JPY Equity TRS (compo) | USD | equity spot (JPY) | USDJPY |
| Bond Forward | USD | — | — |
| Bond TRS | USD (bond + funding) | — | — |

Bonds are risk-free (government / agency), discounted on the USD curve. Issuer
credit is an optional factor **only if extended to risky bonds** (§8).

## 8. Credit / risky bonds (optional — build for it)

Bonds ship **risk-free** (govie/agency). If the project is extended to **risky
bonds**, the bond's **issuer** credit is an optional add-on — it is issuer credit
on the underlying, not counterparty credit.

**Where it enters.** The bond's discounting. With an issuer `CreditCurve`
supplied, the bond's cash flows are survival-weighted and a recovery is paid on
default, so the bond (and any forward / TRS on it) prices credit-risky:

```
credit charge ("CVA") = risk-free price - credit-risky price
```

**How to set up the data.** Give the bond an `issuer` (`bonds.csv`); collect the
issuer's CDS curve + recovery (`MARKET_DATA.md` §4) and build a `CreditCurve`;
put it in `MarketState.credit_curves` keyed by that `issuer`. The bond pricers
pick it up automatically; with none, bonds price risk-free.
`capitolis_pricers/credit.py` + `examples/credit_risky_bond.py` show it. Build
your engine so this can be switched on.

**Counterparty (separate, required).** Every trade carries a `counterparty` — not
credit, but the **netting-set key** your simulation groups by to net exposures.

## 9. Validation and hardening

- **Checks to reproduce:** equity and bond forwards match a QuantLib/ORE
  reference; a total-return equity TRS struck at market is ~0 at zero spread; a
  basket equals the sum of its single-name positions.
- **Extensions:** holiday calendars / business-day adjustment; discrete dividend
  schedule; periodic (vs single-bullet) equity resets; caching for speed.
