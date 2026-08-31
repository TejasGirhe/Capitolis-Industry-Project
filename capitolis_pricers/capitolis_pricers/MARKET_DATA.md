# Market Data Specification

This is the market data **you collect** to drive the exposure simulation. It is a
specification, not fill-in templates: use the field titles and definitions below
to assemble the risk-factor inputs and to build the `MarketState` the pricers
consume. (Trades and underlyings are supplied separately by Capitolis.)

Everything is quoted **as of a single valuation date**; the same snapshot prices
the whole book, and the same risk factors are evolved forward in the simulation.

Bonds are **risk-free** (government and agency), discounted on the USD risk-free
curve. If the project is extended to **risky bonds**, the bond ISSUER's credit
(CDS) is an **optional** input (§4).

---

## 1. Risk factors

The simulation evolves a joint set of correlated risk factors and reprices the
book along each path. Identify, calibrate, and **correlate** all of them:

| Risk factor | What it drives | Market data needed |
|---|---|---|
| **Short rate (USD)** | discounting, floating-funding projection, bond prices | USD OIS (SOFR) curve |
| **Equity** | equity TRS performance leg | equity spot + dividend rate (per name) |
| **FX (USDJPY)** | compo JPY conversion; FX exposure | JPYUSD FX forward curve |

**Correlation.** Collect the pairwise correlations among all factors — the short
rate, each equity, and FX — and simulate them jointly.

---

## 2. Curves — conventions and pillars

Only two curves are needed:

- the **USD** rate curve (all discounting and funding is USD; the compo JPY leg
  grows on the USD curve; risk-free bonds discount on it), and
- the **JPYUSD FX forward** curve.

There is **no JPY rate curve**. The compo JPY trade has no JPY-denominated
cashflows — both legs are USD and the equity leg grows on the USD curve — so
nothing JPY is ever discounted. The JPYUSD FX forward curve is needed only to
simulate **FX as a risk factor** (its drift is the FX forward); it is collected
directly as FX spot + swap points, and the JPY rate is embedded in those
forwards but never separately built.

Both curves start at **spot (T+2)**: the O/N and T/N pillars roll from today to
spot, and the swap pillars extend the curve from spot outward. These pillar
quotes are the inputs into the simulation (they build the initial curve that is
then evolved).

### 2.1 USD OIS (SOFR) curve

| Field | Title | Description |
|---|---|---|
| `instrument` | Instrument | `DEPO` for O/N and T/N; `OIS` for the swap pillars |
| `tenor` | Tenor | `O/N`, `T/N`, then `1W, 2W, 1M, 2M, 3M, 6M, 9M, 1Y, 18M, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y` (extend as needed) |
| `rate` | Rate | quoted rate, decimal (e.g. `0.0432`) |
| `day_count` | Day count | `ACT/360` (USD money market / SOFR OIS) |
| `spot_lag` | Spot lag | settlement lag to spot; `T+2` |

- **O/N** (overnight): today → T+1. **T/N** (tom-next): T+1 → spot (T+2). Together
  they establish the curve to spot.
- **OIS swaps** (par SOFR OIS rates) define the curve beyond spot.
- One SOFR curve is used for **both** discounting and floating-funding projection
  (OIS discounting).

### 2.2 JPYUSD FX forward curve

Convention: **USDJPY = JPY per USD** (≈ 150). Forward = spot + points.

| Field | Title | Description |
|---|---|---|
| `instrument` | Instrument | `FX_SPOT`; `FX_SWAP` for the point pillars |
| `tenor` | Tenor | `SPOT` (T+2); `O/N`, `T/N`; then `1W, 1M, 2M, 3M, 6M, 9M, 1Y, 2Y` (extend as needed) |
| `quote` | Quote | spot rate for `FX_SPOT`; forward/swap **points** for `FX_SWAP` |
| `spot_lag` | Spot lag | `T+2` |

- **FX spot** settles T+2. **O/N** and **T/N** FX swap points roll today → spot.
- **FX swap points** at each tenor give the forward curve, which sets the FX
  drift for the simulation. (No JPY discount curve is built — compo has no JPY
  cashflows.)

---

## 3. Equity

| Field | Title | Description |
|---|---|---|
| `isin` | ISIN | security identifier; the key that links to the equity in the basket |
| `spot` | Spot | current price in the equity's own currency |
| `dividend_rate` | Dividend rate | projected continuous dividend yield (decimal); sets the equity drift `r − q` in the simulation. Source it as you like (implied, put-call parity). |

(Currency and descriptive fields live with the underlying, not here.)

---

## 4. Credit / CDS (optional — risky bonds only)

Only if you extend to **risky bonds**. Collected per bond **issuer**; the bond is
then priced credit-risky (survival-weighted cash flows + recovery on default).
Not needed for the risk-free (govie/agency) bonds shipped.

| Field | Title | Description |
|---|---|---|
| `issuer` | Issuer | id matching the bond's `issuer` field in `bonds.csv` |
| `tenor` | Tenor | `6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y` |
| `cds_spread` | CDS spread | par CDS spread, decimal (e.g. `0.0150`) |
| `recovery` | Recovery | assumed recovery; `LGD = 1 − recovery` |

CDS spreads give the issuer's survival probabilities (credit triangle) used in
the risky bond price. See `capitolis_pricers/credit.py`.

---

## 5. Volatilities (simulation calibration)

| Field | Title | Description |
|---|---|---|
| `factor` | Factor | `RATE_USD`, an equity `isin`, or `FX_USDJPY` |
| `type` | Type | vol type (e.g. lognormal for equity/FX, normal or lognormal for rates) |
| `tenor` | Tenor / expiry | the vol's tenor or option expiry |
| `volatility` | Volatility | decimal |

Volatilities calibrate each factor's diffusion; they are not used by the linear
pricers.

---

## 6. Correlations

| Field | Title | Description |
|---|---|---|
| `factor_a` | Factor A | one risk factor |
| `factor_b` | Factor B | the other risk factor |
| `correlation` | Correlation | pairwise correlation in `[−1, 1]` |

Provide the pairwise correlations among the short rate, each equity, and FX —
enough to form a positive-semidefinite matrix for the joint simulation.

---

## 7. What the pricer needs vs what the engine needs

- **To price a trade on one snapshot**, the pricers use only: the **USD** discount
  curve, equity **spots**, and the **FX spot**. That is the minimum to build a
  `MarketState`.
- **To simulate**, the engine additionally needs the full curve pillars,
  dividend rates, FX forwards, volatilities, and the correlation matrix above.

Build the `MarketState` from your collected snapshot; see
`capitolis_pricers/data/sample_market.py` for an in-code example.
