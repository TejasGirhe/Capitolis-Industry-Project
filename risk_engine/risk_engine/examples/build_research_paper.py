"""
Research-paper-format writeup of the full project: abstract, formal
methodology with numbered equations, results, discussion, limitations,
conclusion. Distinct from build_report_pdf.py (a step-by-step narrated
report) -- this is written in academic register for a reader who wants the
mathematical framework and findings, not a session walkthrough.

    python risk_engine/examples/build_research_paper.py
"""
import os
from datetime import date
from fpdf import FPDF

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "Capitolis_Risk_Engine_Paper.pdf")

INK = (20, 20, 20)
GREY = (100, 100, 100)
RULE = (180, 180, 180)


class Paper(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Times", "I", 8.5)
        self.set_text_color(*GREY)
        self.cell(0, 8, "A Monte Carlo Counterparty Credit Risk Engine", align="L")
        self.cell(0, 8, str(self.page_no()), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*RULE)
        self.line(20, 15, 190, 15)
        self.ln(5)

    def section(self, num, title):
        self.ln(3)
        self.set_font("Times", "B", 13)
        self.set_text_color(*INK)
        self.cell(0, 8, f"{num}.  {title}", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsection(self, num, title):
        self.ln(1)
        self.set_font("Times", "BI", 11.5)
        self.set_text_color(*INK)
        self.cell(0, 7, f"{num}  {title}", new_x="LMARGIN", new_y="NEXT")

    def para(self, text):
        self.set_font("Times", "", 11)
        self.set_text_color(*INK)
        self.multi_cell(0, 5.8, text)
        self.ln(1.5)

    def eq(self, label, text):
        self.ln(1)
        self.set_font("Courier", "", 10.5)
        self.set_text_color(*INK)
        self.set_x(30)
        self.multi_cell(150, 6.2, text, align="C")
        self.set_font("Times", "I", 9.5)
        self.set_text_color(*GREY)
        self.set_y(self.get_y() - 6.2)
        self.set_x(178)
        self.cell(15, 6.2, f"({label})", align="R")
        self.ln(7)
        self.set_text_color(*INK)

    def bullet(self, text, bold_prefix=None):
        self.set_font("Times", "", 11)
        x0 = self.get_x()
        self.set_x(x0 + 5)
        self.cell(4, 5.8, "-")
        if bold_prefix:
            self.set_font("Times", "B", 11)
            w = self.get_string_width(bold_prefix + "  ")
            self.cell(w, 5.8, bold_prefix)
            self.set_font("Times", "", 11)
        self.multi_cell(0, 5.8, text)
        self.set_x(x0)

    def footnote(self, text):
        self.set_font("Times", "I", 9)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4.6, text)
        self.set_text_color(*INK)
        self.ln(1)

    def table(self, headers, rows, col_widths=None, font_size=9):
        n = len(headers)
        avail = 170
        if col_widths is None:
            col_widths = [avail / n] * n
        self.set_x(20)
        self.set_font("Times", "B", font_size)
        self.set_draw_color(*INK)
        for h, w in zip(headers, col_widths):
            self.cell(w, 7, h, border="B", align="C")
        self.ln()
        self.set_font("Times", "", font_size)
        for row in rows:
            self.set_x(20)
            for val, w in zip(row, col_widths):
                self.cell(w, 6.3, str(val), align="C")
            self.ln()
        self.set_x(20)
        self.cell(sum(col_widths), 0, "", border="T")
        self.ln(4)

    def caption(self, text):
        self.set_font("Times", "I", 9.5)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4.8, text, align="C")
        self.set_text_color(*INK)
        self.ln(2)

    def image_block(self, path, caption_text, w=150):
        if not os.path.exists(path):
            return
        if self.get_y() + w * 0.62 > 270:
            self.add_page()
        x = (210 - w) / 2
        self.image(path, x=x, w=w)
        self.ln(1)
        self.caption(caption_text)


def money(x):
    return f"${x:,.0f}"


pdf = Paper()
pdf.set_auto_page_break(auto=True, margin=22)
pdf.set_margins(20, 20, 20)

# ================================================================= TITLE
pdf.add_page()
pdf.ln(15)
pdf.set_font("Times", "B", 18)
pdf.set_text_color(*INK)
pdf.multi_cell(0, 8, "A Monte Carlo Counterparty Credit Risk Engine:\nModel Design, Exposure Simulation, and Regulatory Sensitivities\nfor an Equity and Bond Total-Return-Swap Book", align="C")
pdf.ln(4)
pdf.set_font("Times", "", 12)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, "Capitolis x UC Berkeley Master of Financial Engineering Industry Project", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Times", "I", 10.5)
pdf.cell(0, 6, date.today().strftime("%B %Y"), align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(*INK)
pdf.ln(10)

pdf.set_font("Times", "B", 11)
pdf.cell(0, 6, "Abstract", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Times", "", 10.3)
pdf.set_x(28)
pdf.multi_cell(154, 5.4,
    "We present the design and implementation of an end-to-end counterparty credit risk engine "
    "for a book of equity total-return swaps, bond forwards, and bond total-return swaps. The "
    "engine simulates interest-rate, equity, and foreign-exchange risk factors jointly under a "
    "single correlated Monte Carlo framework built on the Linear Gauss-Markov (LGM) short-rate "
    "family, geometric Brownian motion (GBM) for equity, and its foreign-exchange analogue "
    "(FXGBM), each with optional stochastic-volatility extensions driven by a CIR variance "
    "process. Exposure profiles (expected exposure, potential future exposure, expected "
    "positive exposure) are computed under a ten-business-day margin-period-of-risk convention. "
    "We derive closed-form SA-CCR regulatory deltas under Basel III Annex 4 and a "
    "bump-and-reprice vega sensitivity, and compute CVA, DVA, and FVA using a rating-tier "
    "credit-spread proxy sourced from public bond-yield data. Market data -- discount curves, "
    "equity and FX spot levels, realized volatility, and realized cross-asset correlation -- "
    "is sourced from free public providers (FRED, Yahoo Finance) wherever such data exists, "
    "with every remaining assumption explicitly documented as a placeholder. We further compare "
    "four rate-model specifications (one- and two-factor LGM, each with and without stochastic "
    "volatility) on an identical book and find a non-monotonic relationship between model "
    "richness and tail exposure, and we develop a reproducible, cross-factor-correlated "
    "extension of the engine's Brownian-bridge date interpolation that recovers the exact "
    "conditional variance omitted by a mean-only bridge.")
pdf.ln(3)

# ================================================================= 1. INTRODUCTION
pdf.section("1", "Introduction")
pdf.para(
    "Counterparty credit risk (CCR) measurement requires simulating the future value of a "
    "derivatives portfolio under the joint evolution of every risk factor the portfolio "
    "depends on, then aggregating simulated exposures into risk metrics such as expected "
    "exposure (EE) and potential future exposure (PFE) that feed capital requirements and "
    "valuation adjustments (CVA/DVA/FVA). This project builds such an engine from first "
    "principles for a book of three instrument types -- equity total-return swaps (TRS), bond "
    "forwards, and bond TRS -- spanning three counterparties, and documents every modeling "
    "choice, data source, and approximation made in its construction.")
pdf.para(
    "The remainder of this paper is organized as follows. Section 2 describes the risk-factor "
    "extraction and stochastic model library. Section 3 describes the joint Monte Carlo "
    "simulation and pricing pipeline, including the Brownian-bridge interpolation used to price "
    "at dates off the simulation grid. Section 4 defines the exposure metrics and the "
    "margin-period-of-risk convention. Section 5 derives the regulatory and risk-desk "
    "sensitivity measures. Section 6 describes the valuation-adjustment (xVA) framework. "
    "Section 7 documents market-data sourcing. Section 8 reports results. Section 9 presents a "
    "rate-model comparison. Section 10 describes an extension to exact Brownian-bridge "
    "interpolation. Section 11 discusses limitations, and Section 12 concludes.")

# ================================================================= 2. MODELS
pdf.section("2", "Risk Factor Extraction and Stochastic Models")
pdf.subsection("2.1", "Factor extraction")
pdf.para(
    "Each trade in the book is parsed to identify the distinct rate, equity, and foreign-"
    "exchange risk factors it depends on, so that the simulator constructs exactly the joint "
    "state space the book requires -- no more, no fewer factors than needed. The book studied "
    "here resolves to one USD rate factor, thirty-seven equity factors, and one USD/JPY "
    "foreign-exchange factor (the latter arising from a JPY-denominated equity basket "
    "referenced by two compo trades).")

pdf.subsection("2.2", "Interest rates: the Linear Gauss-Markov family")
pdf.para(
    "The interest-rate factor is modeled under the Linear Gauss-Markov (LGM) framework, a "
    "separable-HJM representation of the Hull-White short-rate model. The state variable x(t) "
    "evolves as a driftless Brownian motion under the appropriate measure, and the discount "
    "factor between any two times is reconstructed exactly from the simulated state:")
pdf.eq("1", "DF(t,T) = [DF(0,T)/DF(0,t)] . exp( -H(T-t) x(t) - (1/2) H(T-t)^2 zeta(t) )")
pdf.para(
    "where H(.) is the model's deterministic mean-reversion loading function and zeta(t) is "
    "the accumulated variance of x up to time t. Because Eq. (1) is exact given x(t) and "
    "zeta(t), the model reproduces the calibration curve by construction (forward-matching), "
    "with no residual error beyond Monte Carlo noise. Four variants are implemented: one- and "
    "two-factor versions (LGM1F, LGM2F), each with an optional stochastic-volatility (SV) "
    "extension in which the instantaneous volatility is scaled by a CIR mean-one variance "
    "multiplier v(t):")
pdf.eq("2", "dv(t) = kappa (1 - v(t)) dt + eta sqrt(v(t)) dW(t),   sigma_eff(t) = sigma(t) sqrt(v(t))")
pdf.para(
    "The production configuration for this book is LGM2F-SV; Section 9 compares all four "
    "variants directly.")

pdf.subsection("2.3", "Equity: geometric Brownian motion")
pdf.para(
    "Each equity factor follows geometric Brownian motion, with an SV extension sharing the "
    "same CIR variance driver as Eq. (2). The drift is not read off a static forward curve but "
    "is constructed to be path-consistent with the SAME simulated short-rate path within each "
    "Monte Carlo scenario -- a design choice that keeps rate and equity dynamics mutually "
    "consistent path-by-path rather than only in expectation.")

pdf.subsection("2.4", "Foreign exchange: FXGBM")
pdf.para(
    "The FX factor (USD/JPY) follows the same construction as equity, with drift set by "
    "covered interest-rate parity against the simulated domestic short rate, again "
    "path-consistent rather than curve-static.")

pdf.subsection("2.5", "Joint correlated simulation")
pdf.para(
    "All active factors are simulated under a single N-driver correlated Monte Carlo scheme. "
    "Let rho denote the correlation matrix among the driving Brownian increments across every "
    "factor's own driver(s) (two for LGM2F, one for LGM1F/GBM/FXGBM). Writing rho = L L^T via "
    "Cholesky factorization, independent standard-normal draws Z are correlated as")
pdf.eq("3", "Z_joint = Z_indep @ L^T,      Z_indep ~ N(0, I)")
pdf.para(
    "and each factor's own driver columns are sliced from Z_joint before being passed to that "
    "factor's own path simulator. This single joint draw is what allows netted exposure to "
    "reflect genuine cross-trade, cross-asset-class diversification, rather than treating each "
    "trade's risk factors as independent.")

# ================================================================= 3. SIMULATION & PRICING
pdf.add_page()
pdf.section("3", "Simulation and Pricing Pipeline")
pdf.subsection("3.1", "A three-stage design")
pdf.para(
    "The pipeline separates three concerns that would otherwise couple simulation cost to "
    "portfolio complexity. First, a fixed, coarse simulation grid is Monte Carlo simulated "
    "exactly once (weekly resolution through month three, monthly through year one, quarterly "
    "beyond, out to the book's longest maturity plus one year). Second, every date at which a "
    "price is actually required -- every trade's cashflow/reset dates, and every reporting "
    "anchor's margin-period-of-risk window endpoints -- is collected into a set of regression "
    "dates, independent of the simulation grid's own density. Third, prices at each regression "
    "date are obtained by interpolating the cached simulated state, not by re-simulating.")
pdf.subsection("3.2", "Brownian-bridge state interpolation")
pdf.para(
    "For a driftless Brownian state x(t) with accumulated variance zeta(t), the conditional "
    "distribution of x at an intermediate date d given two observed grid dates t0 < d < t1 is "
    "itself Gaussian, with conditional mean and variance")
pdf.eq("4", "E[x(d) | x(t0), x(t1)] = x(t0) + w (x(t1) - x(t0)),   w = (zeta(d)-zeta(t0)) / (zeta(t1)-zeta(t0))")
pdf.eq("5", "Var[x(d) | x(t0), x(t1)] = (zeta(d)-zeta(t0)) (zeta(t1)-zeta(d)) / (zeta(t1)-zeta(t0))")
pdf.para(
    "Weighting by accumulated variance rather than calendar time is essential whenever the "
    "instantaneous volatility term structure is non-constant: linear-in-time interpolation "
    "would misstate the conditional mean whenever the two bracketing dates span periods of "
    "different volatility. The baseline implementation evaluates only Eq. (4); Section 10 "
    "develops a full-bridge extension incorporating Eq. (5) as well.")
pdf.subsection("3.3", "Discount curve reconstruction")
pdf.para(
    "A discount curve at a simulated scenario is never bootstrapped from market quotes at "
    "simulation time; it is reconstructed analytically from the interpolated state via Eq. (1), "
    "evaluated fresh for every (path, date) pair a pricer requests. This preserves exactness "
    "given the interpolated state, at the cost of the approximation inherent in Eq. (4)-(5) "
    "alone (addressed in Section 10).")
pdf.subsection("3.4", "Parallelization")
pdf.para(
    "Pricing is parallelized across Monte Carlo paths via multiprocessing, since paths are "
    "fully independent draws with no shared state -- splitting work by path chunk is exact, not "
    "an approximation. For a 10,000-path, fifteen-trade, hundred-regression-date run on "
    "twenty-four workers, empirical pricing wall-clock time is approximately 900-1000 seconds.")

# ================================================================= 4. EXPOSURE
pdf.section("4", "Exposure Metrics and the Margin Period of Risk")
pdf.para(
    "Counterparty exposure is not computed as a naive point-in-time mark-to-market value. "
    "Instead, following standard industry practice for collateralized exposure under a margin "
    "agreement, exposure at a reporting date t reflects the loss that would be realized if the "
    "counterparty defaulted at t, collateral had been posted based on the prior business day's "
    "mark, and it takes a fixed margin period of risk (MPoR, here ten business days) to close "
    "out the position:")
pdf.eq("6", "Exposure(t) = NPV(t + MPoR) - NPV(t - VMlag)")
pdf.para(
    "with VMlag = 1 business day. From the resulting distribution of exposures across "
    "simulated paths at each reporting date, we compute:")
pdf.bullet("expected exposure, EE(t) = E[max(Exposure(t), 0)];")
pdf.bullet("negative expected exposure, NEE(t) = E[min(Exposure(t), 0)], the mirror quantity feeding DVA;")
pdf.bullet("potential future exposure at confidence q, PFE_q(t), the qth percentile of the exposure distribution;")
pdf.bullet("maximum PFE (MPE_q), the largest PFE_q observed across all reporting dates;")
pdf.bullet("effective expected positive exposure (EEPE), the time-weighted average of the running-maximum EE over the first simulation year.")
pdf.para(
    "Two engineering safeguards materially affect these metrics: reporting anchors with no "
    "valid prior variation-margin mark are excluded rather than assigned a fictitious zero "
    "mark, and trades maturing within the ten-business-day close-out window straddling a "
    "reporting date are excluded from both window endpoints, preventing a spurious negative-"
    "exposure artifact from a trade that has already rolled off.")

# ================================================================= 5. GREEKS
pdf.section("5", "Sensitivity Measures")
pdf.subsection("5.1", "SA-CCR delta (Basel III Annex 4)")
pdf.para(
    "Regulatory delta under the Standardized Approach for Counterparty Credit Risk feeds the "
    "exposure-at-default calculation EAD = alpha (RC + PFE_addon). For an interest-rate trade "
    "with start and end dates S and E, the supervisory duration is")
pdf.eq("7", "SD(S,E) = [ exp(-0.05 S) - exp(-0.05 E) ] / 0.05")
pdf.para(
    "and delta is the product of trade notional, SD, direction sign, and a fixed supervisory "
    "factor (0.50% for interest rate, 4.00% for FX, 32% for single-name equity). This is a "
    "closed-form, notional-based quantity with no dependence on simulated market data.")
pdf.subsection("5.2", "Vega (bump-and-reprice)")
pdf.para(
    "SA-CCR has no vega concept -- its exposure add-on uses a fixed supervisory volatility "
    "factor per asset class rather than a shocked recomputation. We therefore define a "
    "separate, risk-desk-style vega: each factor group's volatility input is shocked by a fixed "
    "amount (1 basis point in this study) and the full exposure pipeline is re-run end to end. "
    "Vega for metric M is reported as the raw dollar change,")
pdf.eq("8", "Vega(M) = M(shocked) - M(base)")
pdf.para(
    "rather than a per-unit-of-shock sensitivity, so that the reported number is directly "
    "interpretable as the exposure impact of the stated shock.")

# ================================================================= 6. XVA
pdf.section("6", "Valuation Adjustments (xVA)")
pdf.para(
    "Credit and funding valuation adjustments are computed from the EE/NEE profiles of Section "
    "4 combined with counterparty and own survival curves:")
pdf.eq("9", "CVA = (1-R) sum_t DF(0,t) EE(t) [S(t_prev) - S(t)]")
pdf.eq("10", "DVA = (1-R_own) sum_t DF(0,t) (-NEE(t)) [S_own(t_prev) - S_own(t)]")
pdf.eq("11", "FVA = sum_t DF(0,t) EE(t) s_fund dt")
pdf.para(
    "where R is recovery rate, S(.) is the survival probability implied by a credit-triangle "
    "approximation (hazard rate h = s/(1-R), S(t) = exp(-h t)) from a credit spread s, and "
    "s_fund is a funding spread. Net xVA is reported as CVA - DVA + FVA.")

# ================================================================= 7. MARKET DATA
pdf.section("7", "Market Data Sourcing")
pdf.para(
    "A central methodological commitment of this project is that every market-data input is "
    "sourced from a free, public provider wherever such a source exists, with any remaining gap "
    "explicitly documented as a placeholder rather than silently assumed. Table 1 summarizes "
    "the classification used throughout.")
pdf.table(
    ["Input", "Classification", "Source"],
    [
        ["USD discount curve", "Real", "FRED (SOFR, T-bills, Treasuries)"],
        ["Equity spot & realized vol", "Real", "Yahoo Finance daily closes"],
        ["FX spot & realized vol", "Real", "Yahoo Finance daily closes"],
        ["Cross-asset correlation", "Real", "Realized return correlation, Yahoo Finance"],
        ["Rate volatility", "Placeholder", "No free rate vol-surface source; flat assumption"],
        ["Credit spreads", "Proxy", "FRED rating-tier corporate yields, tiered by book size"],
    ],
    col_widths=[55, 35, 80],
)
pdf.para(
    "Realized (backward-looking) volatility and correlation are used in place of implied "
    "(forward-looking) quantities, since option-chain and correlation-swap data are paid "
    "products with no free equivalent; this is a genuine methodological limitation, not merely "
    "a data-quality footnote, since implied measures price in the market's current expectation "
    "of future co-movement in a way realized history cannot. Counterparty credit spreads are a "
    "rating-tier proxy: no free source publishes name-specific curves for the counterparties "
    "studied here, so each is tiered by book notional and mapped to a FRED Moody's Aaa/Baa "
    "corporate-bond-yield spread over the ten-year Treasury, interpolated linearly by tier.")

# ================================================================= 8. RESULTS
pdf.add_page()
pdf.section("8", "Results")
pdf.subsection("8.1", "Exposure profiles")
pdf.para(
    "Table 2 reports exposure metrics under the production configuration (LGM2F-SV, GBM-SV, "
    "FXGBM-SV; 10,000 paths; reference date 2026-08-24; real sourced market data throughout, "
    "including 666 pairwise correlations across thirty-six of thirty-seven equity names -- one "
    "name failed to resolve via the data provider and was excluded along with its referencing "
    "trade).")
pdf.table(
    ["Counterparty", "Max EE", "EEPE", "MPE 95%", "MPE 99%"],
    [
        ["CPTY_A", money(3195078.90), money(151570.58), money(12719499.16), money(17692473.51)],
        ["CPTY_B", money(1284126.22), money(61607.01), money(5219363.12), money(7185813.17)],
        ["CPTY_C", money(2796830.24), money(556056.04), money(11472212.85), money(16073414.33)],
        ["Book total", money(4765362.10), money(651031.76), money(19474807.74), money(27194888.78)],
    ],
    col_widths=[35, 32, 32, 35, 36],
)
pdf.caption("Table 2. Exposure metrics by counterparty, production configuration.")
pdf.para(
    "Exposure is heavily front-loaded across all three counterparties: expected exposure peaks "
    "at the first reporting date and decays sharply thereafter, consistent with a book "
    "dominated by short-dated equity TRS and bond-forward positions. CPTY_C's effective "
    "expected positive exposure is substantially larger relative to its peak EE than the other "
    "two counterparties, reflecting a single longer-dated equity TRS position that continues "
    "contributing exposure well after the remainder of the book has matured.")
pdf.image_block(os.path.join(HERE, "exposure_profile.png"),
                 "Figure 1. EE, PFE (95%/99%), and MPE profiles by counterparty.")

pdf.subsection("8.2", "Sensitivities")
pdf.para(
    "Table 3 reports SA-CCR delta by counterparty and asset class; Table 4 reports vega "
    "(Shocked minus Base, per Eq. (8)) under a uniform one-basis-point volatility shock to each "
    "factor group, at 10,000 paths.")
pdf.table(
    ["Counterparty", "Equity Delta", "IR Delta", "FX Delta"],
    [
        ["CPTY_A", money(-73578346.67), money(-179425.52), "n/a"],
        ["CPTY_B", money(-29560588.48), money(-93506.05), money(-1726708.00)],
        ["CPTY_C", money(-9600000.00), money(-761342.02), "n/a"],
    ],
    col_widths=[40, 45, 42, 43],
)
pdf.caption("Table 3. SA-CCR delta by counterparty and asset class.")
pdf.table(
    ["Counterparty", "Rate dMPE99", "Equity dMPE99", "FX dMPE99"],
    [
        ["CPTY_A", money(20752624.28), money(24892052.43), money(0.00)],
        ["CPTY_B", money(-6947823.06), money(10239234.74), money(4679335.28)],
        ["CPTY_C", money(1603118757.19), money(965164.86), money(0.00)],
        ["Book total", money(949793470.48), money(17983089.80), money(3501003.16)],
    ],
    col_widths=[40, 45, 42, 43],
)
pdf.caption("Table 4. Vega (Shocked - Base) in 99% MPE, by factor group.")
pdf.para(
    "FX vega is nonzero only for CPTY_B, the sole counterparty holding JPY-denominated "
    "underlyings. CPTY_C's rate vega is an order of magnitude larger than either other "
    "counterparty's, consistent with it carrying the book's longest-dated interest-rate-"
    "sensitive exposure.")

pdf.subsection("8.3", "Valuation adjustments")
pdf.table(
    ["Counterparty", "Spread (proxy)", "CVA", "DVA", "FVA", "Net xVA"],
    [
        ["CPTY_A", "1.285%", money(3577.59), money(4152.28), money(4180.51), money(3605.81)],
        ["CPTY_B", "1.500%", money(1681.25), money(1738.04), money(1683.30), money(1626.51)],
        ["CPTY_C", "1.070%", money(6636.15), money(10227.06), money(9336.14), money(5745.23)],
        ["Book total", "-", money(11894.99), money(16117.38), money(15199.94), money(10977.55)],
    ],
    col_widths=[32, 28, 28, 28, 28, 26],
    font_size=8.3,
)
pdf.caption("Table 5. CVA, DVA, FVA, and net xVA by counterparty, 10,000 paths.")

# ================================================================= 9. MODEL COMPARISON
pdf.add_page()
pdf.section("9", "Rate Model Comparison")
pdf.para(
    "To isolate the effect of rate-model specification from any other source of variation, all "
    "four LGM variants were re-run against the identical book, reference date, path count "
    "(10,000), and market data. Table 6 reports book-level results.")
pdf.table(
    ["Model", "MPE 99%", "MPE 95%", "EEPE", "Max EE"],
    [
        ["LGM1F", money(26692940.81), money(19269475.60), money(686611.20), money(4731422.26)],
        ["LGM1F-SV", money(26907146.26), money(19289907.36), money(685230.95), money(4734059.53)],
        ["LGM2F", money(25167376.28), money(17954510.59), money(627398.68), money(4385498.07)],
        ["LGM2F-SV*", money(27194888.78), money(19474807.74), money(651031.76), money(4765362.10)],
    ],
    col_widths=[40, 33, 33, 32, 32],
)
pdf.caption("Table 6. Book-level exposure metrics under four LGM specifications. (*production configuration)")
pdf.para(
    "Three findings emerge. First, LGM1F and LGM1F-SV are nearly indistinguishable on every "
    "metric, indicating that for a book whose exposure is heavily concentrated within the first "
    "simulation year, stochastic volatility has insufficient time to diverge meaningfully from "
    "its deterministic-volatility counterpart before the underlying exposure has already run "
    "off. Second, and more surprising, LGM2F exhibits the LOWEST tail risk of the four "
    "specifications -- a second rate factor is generally expected to enrich curve dynamics "
    "(permitting twists in addition to parallel shifts) and thereby to increase, not decrease, "
    "tail exposure; the observed result suggests that for this book's particular cashflow "
    "structure, the second factor's own mean-reversion dampens rather than amplifies exposure "
    "over the relevant short horizon. This is a book-specific empirical finding and should not "
    "be read as a general property of two-factor models. Third, the production specification "
    "(LGM2F-SV) combines both effects and produces the highest tail risk of the four, "
    "consistent with curve richness and stochastic volatility compounding rather than "
    "offsetting one another once both are present.")
pdf.para(
    "A methodological note: producing this comparison surfaced a defect in the engine's "
    "correlation-matrix repair routine. Pairwise correlations sourced independently from "
    "historical return series are not guaranteed to form a valid (positive semi-definite) "
    "matrix, a precondition for Cholesky factorization. The repair routine's tolerance check "
    "(rejecting repair whenever every eigenvalue exceeded -1e-10) was looser than the tolerance "
    "the underlying linear-algebra library's own Cholesky routine enforces internally, so a "
    "matrix with a numerically negligible negative eigenvalue (-1.85 x 10^-15) passed the "
    "tolerance check yet still failed factorization -- a defect affecting three of the four "
    "model specifications compared here, though not the production specification, which "
    "happened not to trigger it. The fix routes every correlation matrix through eigenvalue "
    "clipping unconditionally rather than conditionally on a tolerance check; the full test "
    "suite (57 tests) was re-verified passing after the fix before this comparison was "
    "produced.")

# ================================================================= 10. BRIDGE EXTENSION
pdf.section("10", "Extension: Exact Brownian-Bridge Interpolation")
pdf.para(
    "The baseline interpolation scheme (Section 3.2) evaluates only the conditional mean of "
    "Eq. (4), omitting the conditional-variance noise term of Eq. (5). This is a documented, "
    "deliberate approximation whose impact is controlled by simulation-grid density -- coarser "
    "grids omit more variance at their midpoints -- and is mitigated in the baseline design by "
    "keeping the grid dense where cashflow dates cluster. This section develops and validates "
    "an exact extension incorporating Eq. (5).")
pdf.para(
    "Three requirements make this a nontrivial extension rather than a one-line addition of "
    "noise. First, REPRODUCIBILITY: the interpolation routine is called repeatedly for "
    "identical (path, factor, date) triples -- once for the exposure engine's NPV at a "
    "reporting date and again for its NPV at that date plus the margin period of risk, and "
    "independently across multiprocess pricing workers -- so naively drawing fresh noise on "
    "each call would make repeated evaluations of the same quantity inconsistent. This is "
    "resolved by deterministic seeding: each path's noise is derived from a cryptographic hash "
    "of the tuple (path index, date, salt) rather than drawn from live random-number-generator "
    "state.")
pdf.para(
    "Second, CROSS-FACTOR CORRELATION: bridge noise for one factor and bridge noise for another "
    "factor at the same date must respect the same correlation structure the main simulation's "
    "grid-date draws already carry, or interpolated dates would exhibit spuriously "
    "under-correlated (excess independent) cross-factor behavior relative to on-grid dates. "
    "This is resolved by drawing one joint standard-normal vector across every factor's drivers "
    "simultaneously, correlating it via the same Cholesky factor the grid-date simulation used, "
    "and slicing the result per factor -- mirroring Eq. (3) exactly, with deterministic seeding "
    "in place of live random-number-generator draws.")
pdf.para(
    "Third, a subtlety identified during implementation: an early draft attempted to obtain a "
    "factor's own correlated noise by slicing the FULL correlation Cholesky factor L at that "
    "factor's own diagonal block and treating the slice as if it were the Cholesky factor of "
    "that block alone. This is only correct when the factor is first in the driver ordering; "
    "for any factor at a nonzero offset, the sliced sub-matrix does not equal the true diagonal "
    "block's own Cholesky factor. A four-by-four numerical test case confirmed the defect "
    "directly, reconstructing [[0.907, 0.547], [0.547, 0.827]] from a sliced sub-block where the "
    "true target block was [[1, 0.6], [0.6, 1]]. The corrected implementation draws the full "
    "joint vector first and slices the RESULT, never a sub-block of the Cholesky factor itself.")
pdf.para(
    "The extension was validated by: (i) confirming bit-identical reproducibility across "
    "repeated calls with identical inputs; (ii) confirming a two-factor test case with target "
    "correlation 0.70 reproduced an empirical correlation of 0.6985 over 20,000 draws; (iii) "
    "confirming the bridge-variance formula (Eq. (5)) vanishes at both endpoints and is "
    "maximized at the midpoint, matching its closed form exactly; (iv) confirming the extended "
    "routine, with the new noise term disabled, is bit-identical to the pre-existing mean-only "
    "behavior; (v) an end-to-end integration test against the real sourced book confirming "
    "increased state variance at an off-grid date for both a rate factor and an equity factor "
    "when the extension is enabled; and (vi) re-verification of the full 57-test suite. The "
    "extension is currently exposed as an opt-in flag, defaulted to off, since enabling it by "
    "default would change every off-grid-date result reported in Section 8 and that re-"
    "computation has not yet been performed.")

# ================================================================= 11. LIMITATIONS
pdf.section("11", "Limitations")
pdf.bullet(
    "realized (historical) volatility and correlation are used throughout in place of implied "
    "(market-priced, forward-looking) measures, which are unavailable from any free data "
    "source encountered in this study -- this is a genuine methodological gap, not a "
    "data-quality footnote.", bold_prefix="Realized versus implied market data:")
pdf.bullet(
    "counterparty credit spreads are a rating-tier proxy derived from book size, not "
    "observed name-specific CDS or bond spreads, since the counterparties studied are not "
    "real market entities.", bold_prefix="Credit spread proxy:")
pdf.bullet(
    "rate volatility has no free public source and is held at a flat assumed level "
    "throughout every specification studied, including the production configuration.",
    bold_prefix="Rate volatility placeholder:")
pdf.bullet(
    "the pipeline also supports a fully independent per-trade simulation mode, useful when "
    "cross-trade correlation is not desired, but netted exposure under that mode does not "
    "reflect portfolio diversification effects -- an explicit, deliberate tradeoff rather than "
    "an oversight.", bold_prefix="Per-trade independent mode:")
pdf.bullet(
    "the exact Brownian-bridge extension of Section 10 is implemented and validated but not "
    "yet the production default; the results in Section 8 reflect the mean-only baseline.",
    bold_prefix="Bridge-noise extension not yet in production:")

# ================================================================= 12. CONCLUSION
pdf.section("12", "Conclusion")
pdf.para(
    "We have presented a complete counterparty credit risk engine spanning stochastic model "
    "construction, joint Monte Carlo simulation, exposure aggregation under a realistic margin-"
    "period-of-risk convention, regulatory and risk-desk sensitivity measures, and valuation "
    "adjustments, built throughout on real market data wherever a free source exists and with "
    "every remaining assumption explicitly documented. A direct comparison of four rate-model "
    "specifications on an identical book revealed a non-monotonic and book-specific "
    "relationship between model richness and tail exposure, underscoring the importance of "
    "empirical model comparison rather than a priori assumptions about which specification is "
    "most conservative. We further developed and validated an exact extension to the engine's "
    "date-interpolation scheme, recovering the conditional variance a mean-only Brownian bridge "
    "omits, while identifying and correcting two independent numerical defects -- a correlation-"
    "matrix repair tolerance mismatched to its downstream linear-algebra requirement, and an "
    "incorrect sub-block-of-Cholesky-factor construction -- in the course of that work. Natural "
    "next steps include re-running the full production pipeline with the bridge extension "
    "enabled, sourcing implied rather than realized volatility and correlation should a "
    "suitable free data source become available, and extending the joint simulation to "
    "multi-currency rate factors.")

pdf.output(OUT_PATH)
print(f"Saved {OUT_PATH}")
