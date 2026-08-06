"""
CrystalDB
=========
SQLite-backed database of nonlinear crystals.
Provides CRUD operations and returns SellmeierFormula objects.

Usage
-----
    from gui.core_py.crystal_db import get_db
    db = get_db()                      # singleton
    cr = db.get("MgO:PPLN")
    sf = db.sellmeier(cr, axis="e")    # SellmeierFormula instance
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional

from .sellmeier import SellmeierFormula

# ── Default database path ──────────────────────────────────────────────
_DEFAULT_DB = Path(__file__).parent.parent.parent / "crystals" / "crystals.db"

# ── Preloaded crystal data ─────────────────────────────────────────────
# MgO:PPLN: Gayer et al., Appl. Phys. B 91, 343-348 (2008) — LiNbO3 (niobate).
# MgO:sPPLT: Bruner et al., Opt. Lett. 28, 194-196 (2003) — LiTaO3 (tantalate);
#   NOT the same paper/formula as MgO:PPLN (different host crystal entirely).
# ZGP: Zelmon et al., JOSAB 18, 1332 (2001).
# Temperature factor for the LiNbO3/Gayer family: f = (T-24.5)*(T+570.82), T in °C.

_LN_FORMULA = (
    "sqrt("
    "(A1 + B1*(T - 24.5)*(T + 570.82)) + "
    "(A2 + B2*(T - 24.5)*(T + 570.82)) / "
    "(L**2 - (A3 + B3*(T - 24.5)*(T + 570.82))**2) + "
    "(A4 + B4*(T - 24.5)*(T + 570.82)) / (L**2 - A5**2) - "
    "A6*L**2)"
)

# Bruner et al. (2003), Eq. (5): n_e^2(L,T) = A + (B+b(T))/(L^2-(C+c(T))^2)
#   + E/(L^2-F^2) + G/(L^2-H^2) + D*L^2, with b(T)=BT*(T+273.15)^2 and
#   c(T)=CT*(T+273.15)^2 (T in °C; note the extra pole at H, absent from
#   the Gayer/LiNbO3 formula above — a different functional form, not just
#   different coefficients).
_SLT_FORMULA = (
    "sqrt(A + (B + BT*(T + 273.15)**2) / "
    "(L**2 - (C + CT*(T + 273.15)**2)**2) + "
    "E/(L**2 - F**2) + G/(L**2 - H**2) + D*L**2)"
)

_ZGP_FORMULA_E = "sqrt(A + B*L**2/(L**2 - C) + D*L**2/(L**2 - E))"
_ZGP_FORMULA_O = "sqrt(A + B*L**2/(L**2 - C) + D*L**2/(L**2 - E))"

# BBO (beta-BaB2O4): Kato, IEEE J. Quantum Electron. 22, 1013 (1986) —
# Sellmeier fit at T=20 C (no T term in the sqrt; no T-dependence is
# reported for the dispersion fit itself). A linear thermo-optic
# correction dn/dT (also Kato 1986, valid 0.4-1.0 um only) is added
# outside the sqrt so the formula stays usable at T != 20 C.
_BBO_FORMULA_O = "sqrt(A + B/(L**2 - C) - D*L**2) + dndT*(T - 20)"
_BBO_FORMULA_E = "sqrt(A + B/(L**2 - C) - D*L**2) + dndT*(T - 20)"

PRELOADED: list[dict] = [
    # ── MgO:PPLN ──────────────────────────────────────────────────────
    {
        "name":       "MgO:PPLN",
        "type":       "QPM-PPLN",
        "lambda_min": 0.4,
        "lambda_max": 5.0,
        "deff":       25.0,        # pm/V  (dQ = 2×deff/π is applied for QPM in code)
        "alpha_p":    0.002,       # cm⁻¹  at 1064 nm
        "alpha_s":    0.025,       # cm⁻¹  at 532 nm
        "alpha_i":    0.025,
        "formula_e":  _LN_FORMULA,
        "formula_o":  "",
        "coeffs_e":   json.dumps({
            "A1": 5.756,     "A2": 0.0983,    "A3": 0.2020,
            "A4": 189.32,    "A5": 12.52,     "A6": 0.0132,
            "B1": 2.860e-6,  "B2": 4.700e-8,
            "B3": 6.113e-8,  "B4": 1.516e-4,
        }),
        "coeffs_o":   "{}",
        "kappa":      4.6,         # W/(m·K)
        "alpha_th":   14.8e-6,     # K⁻¹  (a-axis)
        "cp":         628.0,       # J/(kg·K)
        "rho":        4640.0,      # kg/m³
        "lambda0":    6.99,        # μm  grating period at T0
        "T0":         27.0,        # °C  reference temperature
        "reference":  "Gayer et al., Appl. Phys. B 91, 343 (2008)",
        "reference_bibtex": (
            "@article{Gayer2008, author={Gayer, O. and Sacks, Z. and "
            "Galun, E. and Arie, A.}, title={Temperature and wavelength "
            "dependent refractive index equations for MgO-doped congruent "
            "and stoichiometric LiNbO3}, journal={Applied Physics B}, "
            "volume={91}, pages={343--348}, year={2008}}"
        ),
        "preloaded":  1,
    },
    # ── MgO:sPPLT (stoichiometric, MgO-doped periodically-poled LiTaO3) ─
    # Formula/coefficients from Bruner et al. (2003) — a different host
    # crystal (lithium TANTALATE) from MgO:PPLN (lithium NIOBATE) above,
    # with its own Sellmeier form (_SLT_FORMULA, 3 poles vs. 2).
    {
        "name":       "MgO:sPPLT",
        "type":       "QPM-PPLN",
        "lambda_min": 0.39,
        "lambda_max": 4.1,
        "deff":       13.7,        # pm/V  (dQ = 2×deff/π is applied for QPM in code)
        "alpha_p":    0.021,
        "alpha_s":    0.002,
        "alpha_i":    0.002,
        "formula_e":  _SLT_FORMULA,
        "formula_o":  "",
        "coeffs_e":   json.dumps({
            "A": 4.502483, "B": 0.007294, "C": 0.185087, "D": -0.02357,
            "E": 0.073423, "F": 0.199595, "G": 0.001,     "H": 7.99724,
            "BT": 3.483933e-8, "CT": 1.607839e-8,
        }),
        "coeffs_o":   "{}",
        "kappa":      4.6,
        "alpha_th":   16.0e-6,     # Bruner et al. explicitly borrow the CLT value (no SLT data available)
        "cp":         628.0,
        "rho":        7450.0,
        "lambda0":    7.57,
        "T0":         27.0,
        "reference":  "Bruner et al., Opt. Lett. 28, 194 (2003)",
        "reference_bibtex": (
            "@article{Bruner2003, author={Bruner, Ariel and Eger, David "
            "and Oron, Moshe B. and Blau, Pinhas and Katz, Moti and "
            "Ruschin, Shlomo}, title={Temperature-dependent Sellmeier "
            "equation for the refractive index of stoichiometric lithium "
            "tantalate}, journal={Optics Letters}, volume={28}, "
            "pages={194--196}, year={2003}}"
        ),
        "preloaded":  1,
    },
    # ── ZGP (ZnGeP2)  birefringent ────────────────────────────────────
    {
        "name":       "ZGP",
        "type":       "Birefringent-uniaxial",
        "lambda_min": 2.0,
        "lambda_max": 8.0,
        "deff":       47.8,        # pm/V  (effective for type-I phase matching)
        "alpha_p":    1.57e-4,
        "alpha_s":    0.17e-6,
        "alpha_i":    0.17e-6,
        "formula_e":  _ZGP_FORMULA_E,
        "formula_o":  _ZGP_FORMULA_O,
        "coeffs_e":   json.dumps({
            "A": 8.0929, "B": 1.8649,
            "C": 0.41468, "D": 0.84052, "E": 452.05,
        }),
        "coeffs_o":   json.dumps({
            "A": 8.0409, "B": 1.68625,
            "C": 0.40824, "D": 1.2880, "E": 611.05,
        }),
        "kappa":      35.0,        # W/(m·K)  — high for ZGP
        "alpha_th":   12.0e-6,
        "cp":         430.0,
        "rho":        4120.0,
        "lambda0":    0.0,         # no QPM
        "T0":         25.0,
        "reference":  "Zelmon et al., JOSAB 18, 1332 (2001)",
        "preloaded":  1,
    },
    # ── BBO (beta-BaB2O4)  negative uniaxial birefringent ─────────────
    # Sellmeier + dn/dT: Kato, IEEE J. Quantum Electron. 22, 1013 (1986).
    # deff left at 0.0 on purpose: d_ooe/d_eoe for BBO depend on the cut
    # angle (theta, phi) via d_ooe = d31*sin(theta) - d22*sin(3*phi)*cos(theta),
    # d_eoe = d22*cos(theta)^2*cos(3*phi), with d22 = +-2.3 pm/V and
    # d31 = -+0.16 pm/V at 1.0642 um (Kato 1986). The user must set the
    # deff value appropriate to their phase-matching angle/configuration
    # via "Set changes" in the Properties tab.
    # Other fields not covered by Kato 1986 (cp, alpha_th, alpha at 1064 nm)
    # use typical literature values, NOT verified against this reference.
    {
        "name":       "BBO",
        "type":       "Birefringent-uniaxial",
        "lambda_min": 0.189,
        "lambda_max": 3.5,
        "deff":       0.0,         # pm/V — user must set (see note above)
        "alpha_p":    1.0e-4,      # cm⁻¹  estimated (not in Kato 1986)
        "alpha_s":    0.01,        # cm⁻¹  at 0.532 um (Kato 1986)
        "alpha_i":    0.01,
        "formula_e":  _BBO_FORMULA_E,
        "formula_o":  _BBO_FORMULA_O,
        "coeffs_e":   json.dumps({
            "A": 2.3753, "B": 0.01224, "C": 0.01667, "D": 0.01516,
            "dndT": -9.3e-6,
        }),
        "coeffs_o":   json.dumps({
            "A": 2.7359, "B": 0.01878, "C": 0.01822, "D": 0.01354,
            "dndT": -16.6e-6,
        }),
        "kappa":      1.2,         # W/(m·K)  literature typical, not from Kato 1986
        "alpha_th":   4.0e-6,      # K⁻¹  literature typical; inert for birefringent PM (Lambda=0)
        "cp":         490.0,       # J/(kg·K)  literature typical, not from Kato 1986
        "rho":        3850.0,      # kg/m³  (Kato 1986)
        "lambda0":    0.0,         # no QPM
        "T0":         20.0,        # °C  — reference T of the Sellmeier fit
        "reference":  "Kato, IEEE J. Quantum Electron. 22, 1013 (1986)",
        "reference_bibtex": (
            "@article{Kato1986, author={Kato, Kiyoshi}, "
            "title={Second-harmonic generation to 2048 {\\AA} in "
            "$\\beta$-{BaB}$_2$O$_4$}, "
            "journal={IEEE Journal of Quantum Electronics}, "
            "volume={22}, pages={1013--1014}, year={1986}}"
        ),
        "preloaded":  1,
    },
]

# ── Schema ─────────────────────────────────────────────────────────────
_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS crystals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    type        TEXT    NOT NULL DEFAULT 'QPM-PPLN',
    lambda_min  REAL    DEFAULT 0.4,
    lambda_max  REAL    DEFAULT 5.0,
    deff        REAL    DEFAULT 0.0,
    alpha_p     REAL    DEFAULT 0.0,
    alpha_s     REAL    DEFAULT 0.0,
    alpha_i     REAL    DEFAULT 0.0,
    beta_p      REAL    DEFAULT 0.0,
    beta_s      REAL    DEFAULT 0.0,
    beta_i      REAL    DEFAULT 0.0,
    rho_p       REAL    DEFAULT 0.0,
    rho_s       REAL    DEFAULT 0.0,
    rho_i       REAL    DEFAULT 0.0,
    formula_e   TEXT    DEFAULT '',
    formula_o   TEXT    DEFAULT '',
    coeffs_e    TEXT    DEFAULT '{}',
    coeffs_o    TEXT    DEFAULT '{}',
    kappa       REAL    DEFAULT 8.0,
    alpha_th    REAL    DEFAULT 14.8e-6,
    cp          REAL    DEFAULT 628.0,
    rho         REAL    DEFAULT 4640.0,
    lambda0     REAL    DEFAULT 0.0,
    T0          REAL    DEFAULT 27.0,
    reference   TEXT    DEFAULT '',
    reference_bibtex TEXT DEFAULT '',
    preloaded   INTEGER DEFAULT 0
)
"""


class CrystalDB:
    """SQLite-backed crystal database with Sellmeier formula support."""

    def __init__(self, db_path: Path = _DEFAULT_DB):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()
        self._migrate()
        self._seed_preloaded()
        self._repair_preloaded()

    # ------------------------------------------------------------------
    # Schema migration (add columns introduced in later versions)
    # ------------------------------------------------------------------
    def _migrate(self):
        existing = {row[1] for row in
                    self._conn.execute("PRAGMA table_info(crystals)").fetchall()}
        for col, defn in [
            ("beta_p", "REAL DEFAULT 0.0"),
            ("beta_s", "REAL DEFAULT 0.0"),
            ("beta_i", "REAL DEFAULT 0.0"),
            ("rho_p", "REAL DEFAULT 0.0"),
            ("rho_s", "REAL DEFAULT 0.0"),
            ("rho_i", "REAL DEFAULT 0.0"),
            ("reference_bibtex", "TEXT DEFAULT ''"),
        ]:
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE crystals ADD COLUMN {col} {defn}")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    def _seed_preloaded(self):
        """Insert preloaded crystals if they don't already exist."""
        for cr in PRELOADED:
            exists = self._conn.execute(
                "SELECT 1 FROM crystals WHERE name=?", (cr["name"],)
            ).fetchone()
            if not exists:
                self._insert(cr)

    def _repair_preloaded(self):
        """Keep already-seeded preloaded rows in sync with PRELOADED.

        _seed_preloaded() only inserts crystals that don't exist yet, so a
        row seeded by an older version of this file (e.g. before a wrong
        Sellmeier formula/citation was fixed here) would otherwise never
        pick up the correction. Safe to overwrite unconditionally: the GUI
        refuses to save user edits to preloaded=1 rows (they're read-only,
        see CrystalsTab._on_save), so there is no user data to lose here.
        Also drops any preloaded row no longer present in PRELOADED
        (e.g. plain "PPLN", removed in favor of MgO:PPLN).
        """
        current_names = {cr["name"] for cr in PRELOADED}
        stale = self._conn.execute(
            "SELECT name FROM crystals WHERE preloaded=1"
        ).fetchall()
        for row in stale:
            if row["name"] not in current_names:
                self._conn.execute(
                    "DELETE FROM crystals WHERE name=? AND preloaded=1",
                    (row["name"],))
        for cr in PRELOADED:
            d = dict(cr)
            name = d.pop("name")
            d["coeffs_e"] = json.dumps(d.get("coeffs_e", {})) \
                if isinstance(d.get("coeffs_e"), dict) else d.get("coeffs_e", "{}")
            d["coeffs_o"] = json.dumps(d.get("coeffs_o", {})) \
                if isinstance(d.get("coeffs_o"), dict) else d.get("coeffs_o", "{}")
            sets = ", ".join(f"{k}=?" for k in d)
            self._conn.execute(
                f"UPDATE crystals SET {sets} WHERE name=? AND preloaded=1",
                list(d.values()) + [name])
        self._conn.commit()

    def _insert(self, d: dict):
        cols = ", ".join(d.keys())
        vals = ", ".join("?" * len(d))
        self._conn.execute(
            f"INSERT OR IGNORE INTO crystals ({cols}) VALUES ({vals})",
            list(d.values())
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def all_names(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM crystals ORDER BY preloaded DESC, name").fetchall()
        return [r["name"] for r in rows]

    def get(self, name: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM crystals WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["coeffs_e"] = json.loads(d["coeffs_e"] or "{}")
        d["coeffs_o"] = json.loads(d["coeffs_o"] or "{}")
        return d

    def add(self, crystal: dict) -> bool:
        """Add a new user-defined crystal.  Returns True on success."""
        try:
            c = dict(crystal)
            if "coeffs_e" in c and isinstance(c["coeffs_e"], dict):
                c["coeffs_e"] = json.dumps(c["coeffs_e"])
            if "coeffs_o" in c and isinstance(c["coeffs_o"], dict):
                c["coeffs_o"] = json.dumps(c["coeffs_o"])
            c.setdefault("preloaded", 0)
            self._insert(c)
            return True
        except Exception:
            return False

    def update(self, name: str, fields: dict) -> bool:
        """Update fields of an existing crystal."""
        f = dict(fields)
        if "coeffs_e" in f and isinstance(f["coeffs_e"], dict):
            f["coeffs_e"] = json.dumps(f["coeffs_e"])
        if "coeffs_o" in f and isinstance(f["coeffs_o"], dict):
            f["coeffs_o"] = json.dumps(f["coeffs_o"])
        sets = ", ".join(f"{k}=?" for k in f)
        try:
            self._conn.execute(
                f"UPDATE crystals SET {sets} WHERE name=?",
                list(f.values()) + [name])
            self._conn.commit()
            return True
        except Exception:
            return False

    def delete(self, name: str) -> bool:
        """Delete a non-preloaded crystal."""
        row = self._conn.execute(
            "SELECT preloaded FROM crystals WHERE name=?", (name,)).fetchone()
        if row is None or row["preloaded"]:
            return False
        self._conn.execute("DELETE FROM crystals WHERE name=?", (name,))
        self._conn.commit()
        return True

    # ------------------------------------------------------------------
    # Sellmeier helpers
    # ------------------------------------------------------------------
    def sellmeier(self, name_or_dict, axis: str = "e") -> SellmeierFormula:
        """
        Return a SellmeierFormula for the given crystal and axis ('e' or 'o').

        Parameters
        ----------
        name_or_dict : str or dict
            Crystal name or dict returned by get().
        axis : 'e' | 'o'
        """
        cr = self.get(name_or_dict) if isinstance(name_or_dict, str) \
             else name_or_dict

        if axis == "o":
            formula = cr.get("formula_o", "")
            coeffs  = cr.get("coeffs_o", {})
        else:
            formula = cr.get("formula_e", "")
            coeffs  = cr.get("coeffs_e", {})

        if isinstance(coeffs, str):
            coeffs = json.loads(coeffs)

        return SellmeierFormula(
            formula, coeffs, label=f"{cr['name']} ({axis}-axis)")


# ── Singleton accessor ─────────────────────────────────────────────────
_instance: Optional[CrystalDB] = None


def get_db(db_path: Path = _DEFAULT_DB) -> CrystalDB:
    """Return the application-wide CrystalDB singleton."""
    global _instance
    if _instance is None:
        _instance = CrystalDB(db_path)
    return _instance
