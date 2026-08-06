"""
SellmeierFormula
================
Parses a Sellmeier expression provided as a string, substitutes
numerical coefficient values, and computes symbolic derivatives
(dn/dλ, d²n/dλ², d³n/dλ³) via SymPy.  All dispersion parameters
(β₁, β₂, β₃, GVD) are derived analytically from those derivatives.

Variables in formula strings
-----------------------------
    L  : wavelength  [μm]
    T  : temperature [°C]
    Any other name is treated as a coefficient (given in `coeffs`).

Example
-------
    formula = "sqrt((A1 + B1*(T-24.5)*(T+570.82)) + "
              "(A2 + B2*(T-24.5)*(T+570.82)) / "
              "(L**2 - (A3 + B3*(T-24.5)*(T+570.82))**2) + "
              "(A4 + B4*(T-24.5)*(T+570.82)) / (L**2 - A5**2) - A6*L**2)"
    coeffs  = {"A1": 5.756, "A2": 0.0983, ..., "B4": 1.516e-4}
    sf = SellmeierFormula(formula, coeffs, label="MgO:PPLN e-axis")
    print(sf.n(1.064, 27.0))    # → 2.145...
    print(sf.beta2(1.064, 27.0) * 1e6)  # GVD in fs²/mm
"""

import numpy as np
import sympy as sp

# Speed of light in μm/ps  (c = 2.998e8 m/s = 299.792458 μm/ps)
_C = 299.792_458


class SellmeierFormula:
    """
    Refractive index and dispersion from a user-supplied Sellmeier formula.

    Parameters
    ----------
    formula_str : str
        SymPy-parseable expression for n(L, T).
    coeffs : dict
        Mapping coefficient-name → float value.
    label : str, optional
        Human-readable name (for display / error messages).
    """

    def __init__(self, formula_str: str, coeffs: dict, label: str = ""):
        self.formula_str = formula_str
        self.coeffs      = dict(coeffs)
        self.label       = label
        self._ready      = False
        self._error      = None
        self._T_in_expr  = False
        self._parse()

    # ------------------------------------------------------------------
    def _parse(self):
        try:
            L = sp.Symbol("L", positive=True)
            T = sp.Symbol("T")

            # Build local dict: coefficient names → SymPy symbols
            local = {"L": L, "T": T, "sqrt": sp.sqrt,
                     "exp": sp.exp, "log": sp.log,
                     "sin": sp.sin, "cos": sp.cos}
            for k in self.coeffs:
                local[k] = sp.Symbol(k)

            expr = sp.sympify(self.formula_str, locals=local)

            # Substitute numerical coefficient values
            subs = [(sp.Symbol(k), float(v)) for k, v in self.coeffs.items()]
            expr = expr.subs(subs)

            # Detect whether T appears (temperature-dependent formula)
            self._T_in_expr = T in expr.free_symbols

            # Any symbol left besides L, T means a coefficient wasn't
            # given a value — fail now with a clear message instead of a
            # cryptic numpy ufunc error at evaluation time.
            leftover = expr.free_symbols - {L, T}
            if leftover:
                names = ", ".join(sorted(s.name for s in leftover))
                raise ValueError(f"Undefined coefficients: {names}")

            # ── Symbolic derivatives w.r.t. L ──────────────────────────
            dn  = sp.diff(expr, L)
            d2n = sp.diff(dn,   L)
            d3n = sp.diff(d2n,  L)

            # ── Lambdify (numpy backend for vectorised evaluation) ──────
            args = (L, T)
            self._fn   = sp.lambdify(args, expr,  modules="numpy")
            self._fdn  = sp.lambdify(args, dn,    modules="numpy")
            self._fd2n = sp.lambdify(args, d2n,   modules="numpy")
            self._fd3n = sp.lambdify(args, d3n,   modules="numpy")

            self._expr_sympy = expr
            self._ready = True

        except Exception as exc:
            self._error = str(exc)
            self._ready = False

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------
    # Core evaluators — accept scalars or numpy arrays
    # ------------------------------------------------------------------
    def _L(self, L):  return np.asarray(L, dtype=float)
    def _T(self, T):  return np.asarray(T, dtype=float)

    def n(self, L, T=25.0) -> np.ndarray:
        """Refractive index n(λ [μm], T [°C])."""
        self._check()
        val = self._fn(self._L(L), self._T(T))
        return np.where(np.isreal(val), np.real(val).astype(float), np.nan)

    def dn_dL(self, L, T=25.0) -> np.ndarray:
        """First derivative dn/dλ  [μm⁻¹]."""
        self._check()
        return np.real(self._fdn(self._L(L), self._T(T))).astype(float)

    def d2n_dL2(self, L, T=25.0) -> np.ndarray:
        """Second derivative d²n/dλ²  [μm⁻²]."""
        self._check()
        return np.real(self._fd2n(self._L(L), self._T(T))).astype(float)

    def d3n_dL3(self, L, T=25.0) -> np.ndarray:
        """Third derivative d³n/dλ³  [μm⁻³]."""
        self._check()
        return np.real(self._fd3n(self._L(L), self._T(T))).astype(float)

    # ------------------------------------------------------------------
    # Dispersion parameters
    # ------------------------------------------------------------------
    def k(self, L, T=25.0) -> np.ndarray:
        """Wavenumber k = 2π n / λ  [μm⁻¹]."""
        return 2 * np.pi * self.n(L, T) / self._L(L)

    def GV(self, L, T=25.0) -> np.ndarray:
        """Group velocity v_g = c / (n − λ dn/dλ)  [μm/ps]."""
        L_ = self._L(L)
        return _C / (self.n(L_, T) - L_ * self.dn_dL(L_, T))

    def beta1(self, L, T=25.0) -> np.ndarray:
        """β₁ = 1/v_g = (n − λ dn/dλ) / c  [ps/μm]."""
        L_ = self._L(L)
        return (self.n(L_, T) - L_ * self.dn_dL(L_, T)) / _C

    def beta2(self, L, T=25.0) -> np.ndarray:
        """GVD  β₂ = λ³ d²n/dλ² / (2π c²)  [ps²/μm]."""
        L_ = self._L(L)
        return L_**3 * self.d2n_dL2(L_, T) / (2 * np.pi * _C**2)

    def beta3(self, L, T=25.0) -> np.ndarray:
        """TOD  β₃ = −λ⁴ (3 d²n/dλ² + λ d³n/dλ³) / (4π² c³)  [ps³/μm]."""
        L_ = self._L(L)
        d2 = self.d2n_dL2(L_, T)
        d3 = self.d3n_dL3(L_, T)
        return -L_**4 / (4 * np.pi**2 * _C**3) * (3*d2 + L_*d3)

    def GVD_fs2_mm(self, L, T=25.0) -> np.ndarray:
        """GVD in fs²/mm  (β₂ [ps²/μm] × 10⁹ = fs²/mm)."""
        return self.beta2(L, T) * 1e9

    def formula_latex(self) -> str:
        """LaTeX string of the symbolic expression (after coeff substitution)."""
        if not self._ready:
            return r"\text{Error}"
        return sp.latex(self._expr_sympy)

    # ------------------------------------------------------------------
    def _check(self):
        if not self._ready:
            raise RuntimeError(
                f"SellmeierFormula '{self.label}' not ready: {self._error}")


class MixedExtraordinaryIndex:
    """
    Angle-dependent extraordinary index for a uniaxial crystal:

        1/n^e(theta)^2 = cos^2(theta)/n_o^2 + sin^2(theta)/n_e^2

    Implements the same n/dn_dL/.../k/GV/beta1/beta2/beta3 interface as
    SellmeierFormula so it can be used as a drop-in replacement per field
    in config_builder.py — but since the mix of two independent Sellmeier
    formulas isn't itself a single symbolic expression, derivatives here
    are obtained by central finite differences instead of SymPy.
    """

    def __init__(self, sf_o: SellmeierFormula, sf_e: SellmeierFormula,
                 theta_deg: float, dlam: float = 1e-4):
        self.sf_o = sf_o
        self.sf_e = sf_e
        self.theta_deg = theta_deg
        self._theta = np.deg2rad(theta_deg)
        self._dlam = dlam
        self._ready = sf_o.is_ready and sf_e.is_ready
        self._error = None if self._ready else "underlying o/e formula not ready"

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> str | None:
        return self._error

    def n(self, L, T=25.0) -> np.ndarray:
        no = self.sf_o.n(L, T)
        ne = self.sf_e.n(L, T)
        return 1.0 / np.sqrt((np.cos(self._theta) / no) ** 2 +
                             (np.sin(self._theta) / ne) ** 2)

    def dn_dL(self, L, T=25.0) -> np.ndarray:
        d = self._dlam
        return (self.n(L + d, T) - self.n(L - d, T)) / (2 * d)

    def d2n_dL2(self, L, T=25.0) -> np.ndarray:
        d = self._dlam
        return (self.n(L + d, T) - 2 * self.n(L, T) + self.n(L - d, T)) / d**2

    def d3n_dL3(self, L, T=25.0) -> np.ndarray:
        d = self._dlam
        return (self.n(L + 2*d, T) - 2*self.n(L + d, T)
                + 2*self.n(L - d, T) - self.n(L - 2*d, T)) / (2 * d**3)

    def k(self, L, T=25.0) -> np.ndarray:
        return 2 * np.pi * self.n(L, T) / np.asarray(L, dtype=float)

    def GV(self, L, T=25.0) -> np.ndarray:
        L_ = np.asarray(L, dtype=float)
        return _C / (self.n(L_, T) - L_ * self.dn_dL(L_, T))

    def beta1(self, L, T=25.0) -> np.ndarray:
        L_ = np.asarray(L, dtype=float)
        return (self.n(L_, T) - L_ * self.dn_dL(L_, T)) / _C

    def beta2(self, L, T=25.0) -> np.ndarray:
        L_ = np.asarray(L, dtype=float)
        return L_**3 * self.d2n_dL2(L_, T) / (2 * np.pi * _C**2)

    def beta3(self, L, T=25.0) -> np.ndarray:
        L_ = np.asarray(L, dtype=float)
        d2 = self.d2n_dL2(L_, T)
        d3 = self.d3n_dL3(L_, T)
        return -L_**4 / (4 * np.pi**2 * _C**3) * (3*d2 + L_*d3)

    def GVD_fs2_mm(self, L, T=25.0) -> np.ndarray:
        return self.beta2(L, T) * 1e9
