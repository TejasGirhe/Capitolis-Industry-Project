"""
Quick standalone PDF: how the correlation matrix driving the joint Monte
Carlo simulation is actually computed. See market_data/vol_corr.py and
simulation/joint.py for the source this is transcribed from.
"""
import os
from fpdf import FPDF

HERE = os.path.dirname(__file__)
OUT_PATH = os.path.join(HERE, "Correlation_Methodology_Note.pdf")

BLUE = (30, 60, 110)
GREY = (90, 90, 90)


class Note(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, "Capitolis Risk Engine -- Correlation Methodology", align="L")
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, 16, 200, 16)
        self.ln(4)

    def h1(self, text):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*BLUE)
        self.ln(2)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def h2(self, text):
        self.set_font("Helvetica", "B", 11.5)
        self.set_text_color(*BLUE)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def formula(self, text):
        self.set_font("Courier", "", 10.5)
        self.set_fill_color(244, 246, 250)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 6.5, text, fill=True)
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def note(self, text):
        self.set_font("Helvetica", "I", 8.5)
        self.set_text_color(*GREY)
        self.multi_cell(0, 4.5, text)
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def bullet(self, text, bold_prefix=None):
        self.set_font("Helvetica", "", 9.5)
        x0 = self.get_x()
        self.set_x(x0 + 4)
        self.cell(3, 5, chr(149))
        if bold_prefix:
            self.set_font("Helvetica", "B", 9.5)
            w = self.get_string_width(bold_prefix + "  ")
            self.cell(w, 5, bold_prefix)
            self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.set_x(x0)


pdf = Note()
pdf.set_auto_page_break(auto=True, margin=18)
pdf.add_page()

pdf.set_font("Helvetica", "B", 19)
pdf.set_text_color(*BLUE)
pdf.cell(0, 12, "Correlation Matrix Methodology", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 7, "How the joint Monte Carlo simulator's cross-factor correlation is built", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)
pdf.ln(4)

pdf.body(
    "The engine never builds a full covariance matrix directly. It works entirely in "
    "correlation space (unit-variance driving Brownian increments); each model's own "
    "calibrated volatility term structure sigma_i(t) is applied separately, inside that "
    "model's own simulate_paths, to scale the correlated draws into that factor's actual "
    "diffusion. Conceptually Cov_ij(t) = rho_ij * sigma_i(t) * sigma_j(t), but that product "
    "is never materialized as a matrix -- rho (Cholesky-factored) and each sigma_i(t) are "
    "kept and applied separately.")

pdf.h1("1. Realized volatility (per factor, marginal)")
pdf.body("Log returns from daily closes:")
pdf.formula("r_i = ln( P_i / P_(i-1) )")
pdf.body("Annualized realized volatility (sample stdev of log returns, Bessel-corrected):")
pdf.formula("sigma_hat = sqrt( (1/(n-1)) * sum_i (r_i - r_bar)^2 ) * sqrt(252)")
pdf.note("252 = trading days/year. This feeds a flat/ATM vol surface entry point per factor "
         "-- realized vol from one price history has no natural tenor/strike axis.")

pdf.h1("2. Pairwise correlation (Pearson, on log returns)")
pdf.body(
    "For each factor pair, histories are truncated to the shortest overlapping length and "
    "aligned from the end (most recent N observations), then:")
pdf.formula(
    "rho_ab = [ sum_i (x_i - x_bar)(y_i - y_bar) ]\n"
    "         -----------------------------------------\n"
    "         sqrt( sum_i (x_i-x_bar)^2 ) * sqrt( sum_i (y_i-y_bar)^2 )")
pdf.note(
    "x, y are the two factors' log-return series over the same aligned window. This is the "
    "standard Pearson correlation coefficient -- the explicit 1/n factors in the textbook "
    "covariance/variance formulas cancel between numerator and denominator, so it is computed "
    "directly from sums of cross/self products.")

pdf.h1("3. Assembling the full correlation matrix")
pdf.body(
    "Start from the identity (rho_ii = 1 on the diagonal). For every pair of simulated "
    "factors, fill in the sourced pairwise rho_ij:")
pdf.formula("corr = I(n_drivers)\nfor each ordered pair (i, j), i != j:\n    corr[i, j] = corr[j, i] = rho_ij")
pdf.bullet(
    "a 2-factor rate model's two internal drivers both receive the SAME rho_ij to any other "
    "factor -- no free data source can distinguish factor-1-vs-equity from factor-2-vs-equity "
    "correlation for a 2F rate model, so this uniform broadcast is a documented, deliberate "
    "simplification, not an oversight.", bold_prefix="Multi-driver models:")

pdf.h1("4. Positive-semidefinite repair")
pdf.body(
    "Pairwise correlations are estimated independently, pair by pair, from possibly "
    "different overlapping windows -- the resulting matrix is not guaranteed positive "
    "semi-definite (a mathematical requirement to represent any real joint distribution), "
    "which Cholesky factorization requires. Fix: eigendecompose, clip any negative "
    "eigenvalues to a small positive floor, reconstruct, and rescale back to a unit diagonal:")
pdf.formula(
    "Sigma = V * Lambda * V^T                  (eigendecomposition)\n"
    "Lambda' = max(Lambda, epsilon)            (clip negative eigenvalues)\n"
    "Sigma' = V * Lambda' * V^T                (reconstruct)\n"
    "rho'_ij = Sigma'_ij / sqrt(Sigma'_ii * Sigma'_jj)   (rescale to unit diagonal)")
pdf.note("Applied only when needed -- the matrix is used as-is if all eigenvalues are already >= ~0.")

pdf.h1("5. Cholesky factorization -- correlated draws")
pdf.body("Factor the (now valid) correlation matrix:")
pdf.formula("rho = L * L^T                      (numpy.linalg.cholesky)")
pdf.body("Draw independent standard normals per path / time step / driver, then correlate them:")
pdf.formula("Z_indep ~ N(0, I)      shape (n_paths, n_steps, n_drivers)\nZ_joint = Z_indep @ L^T")
pdf.body(
    "Z_joint now carries the target cross-factor correlation structure. It is sliced per "
    "factor (rate drivers, each spot factor's own slot) and passed into that model's own "
    "simulate_paths(), where the model's calibrated sigma(t) scales these correlated shocks "
    "into its actual diffusion.")

pdf.h1("6. Data source & a documented limitation")
pdf.bullet(
    "computed from real historical daily closes (Yahoo Finance for equities/FX), not a paid "
    "implied-correlation product.", bold_prefix="Realized, not implied:")
pdf.bullet(
    "realized (backward-looking) correlation reflects what already happened; implied "
    "(forward-looking, market-priced) correlation would price in the market's current "
    "expectation of co-movement, including event risk -- a genuine methodological gap, not "
    "just a data-quality footnote.", bold_prefix="Limitation:")
pdf.bullet(
    "the rate factor's OWN volatility has no free source and stays a flat placeholder "
    "(Sec. 8 of the full session report); correlation and equity/FX vol are real, sourced "
    "data.", bold_prefix="Rate vol still a placeholder:")

pdf.output(OUT_PATH)
print(f"Saved {OUT_PATH}")
