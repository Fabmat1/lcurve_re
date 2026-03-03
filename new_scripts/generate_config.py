#!/usr/bin/env python3
"""
LCURVE MCMC Configuration Generator v5

Interactively collects binary-star observables, finds maximum
joint-probability starting parameters via weighted least-squares
optimisation, fetches Claret limb-darkening AND gravity-darkening
coefficients from VizieR, and writes a complete JSON config for
the MCMC solver.

New in v5
─────────
• Extended priors: K2, M2, R2, q, M_total, logg1, logg2, T1, T2
  in addition to K1, M1, M2_min, R1
• All priors are optional — any combination the user provides is
  written into the config and used by the MCMC solver
• The optimiser now also uses M2, K2, q, M_total when available

    pip install scipy astroquery astropy requests
"""

import json
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import curses
import textwrap
import pickle
import atexit
from enum import Enum, auto
import argparse
import io
import contextlib

AUTOSAVE_PATH = Path(".lcurve_config_session.pkl")

# ══════════════════════════ Constants ═════════════════════════

DEG2RAD = math.pi / 180.0
DAY2SEC = 86400.0
RSUN_KM = 695700.0
G_MSUN = 1.3271244e11  # km³ s⁻² (heliocentric gravitational constant)
LOGG_SUN = 4.4380  # log10( G·M_sun / R_sun² )  [cgs]

BAND_WAVELENGTH = {
    "TESS": 786.5,
    "Kepler": 640.0,
    "SDSS-u": 355.1,
    "SDSS-g": 468.6,
    "SDSS-r": 616.5,
    "SDSS-i": 748.1,
    "SDSS-z": 893.1,
    "Johnson-U": 365.0,
    "Johnson-B": 440.0,
    "Johnson-V": 547.7,
    "Johnson-R": 640.0,
    "Johnson-I": 790.0,
}

# VizieR filter-column suffixes for multi-filter Claret tables
FILTER_SUFFIX = {
    "TESS": "Te",
    "Kepler": "Ke",
    "SDSS-u": "u'",
    "SDSS-g": "g'",
    "SDSS-r": "r'",
    "SDSS-i": "i'",
    "SDSS-z": "z'",
    "Johnson-U": "U",
    "Johnson-B": "B",
    "Johnson-V": "V",
    "Johnson-R": "R",
    "Johnson-I": "I",
}

TEFF_ALIASES = ("Teff", "teff", "Te", "TEFF")
LOGG_ALIASES = ("logg", "Logg", "log(g)", "LOGG", "log_g")

# ANSI colour codes
W = "\033[97m"
C = "\033[96m"
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
D = "\033[2m"
Z = "\033[0m"
BOLD = "\033[1m"


# ══════════════════════════ Data classes ══════════════════════


@dataclass
class Measurement:
    """A measured quantity with asymmetric uncertainties."""

    value: float
    err_lo: float  # positive number: minus direction
    err_hi: float  # positive number: plus direction

    def __post_init__(self):
        self.err_lo = abs(self.err_lo)
        self.err_hi = abs(self.err_hi)

    def sigma_for(self, model_val: float) -> float:
        """Return the relevant sigma given the sign of (model - observed)."""
        return self.err_lo if model_val < self.value else self.err_hi

    def pull(self, model_val: float) -> float:
        sig = self.sigma_for(model_val)
        return (model_val - self.value) / sig if sig > 0 else 0.0

    def as_tuple(self):
        return (self.value, self.err_lo, self.err_hi)

    def as_prior_str(self):
        return f"{self.value} {self.err_lo} {self.err_hi}"


@dataclass
class StarInfo:
    """All user-supplied information about one stellar component."""

    star_type: str = "ms"  # 'sd', 'wd', 'ms'
    teff: float = 10000.0
    teff_err: Optional[Measurement] = None
    logg: Optional[float] = None
    logg_err: Optional[Measurement] = None
    mass: Optional[Measurement] = None
    radius: Optional[Measurement] = None
    ldc: list = field(default_factory=lambda: [0.4, 0.15, -0.05, 0.02])
    gdc: float = 0.15


@dataclass
class ModelParam:
    """One LCURVE model parameter with its MCMC metadata."""

    value: float
    range: float
    step: float
    vary: bool = False
    defined: bool = True

    def to_str(self) -> str:
        return (
            f"{self.value} {self.range} {self.step} "
            f"{int(self.vary)} {int(self.defined)}"
        )


# ══════════════════════════ Physics ══════════════════════════


def implied_from_params(i_deg, q, vs, r1, P_days, r2=None):
    """Derive physical quantities from (i, q, velocity_scale, r1, P)."""
    si = math.sin(i_deg * DEG2RAD)
    P_sec = P_days * DAY2SEC
    a_km = vs * P_sec / (2 * math.pi)
    M_total = 4 * math.pi**2 * a_km**3 / (G_MSUN * P_sec**2)
    M1 = M_total / (1 + q)
    M2 = q * M_total / (1 + q)
    R1_Rsun = r1 * a_km / RSUN_KM
    result = {
        "K1": vs * si * q / (1 + q),
        "K2": vs * si / (1 + q),
        "R1": R1_Rsun,
        "M1": M1,
        "M2": M2,
        "Mt": M_total,
        "a_Rs": a_km / RSUN_KM,
        "q": q,
    }
    # logg1
    if R1_Rsun > 0 and M1 > 0:
        result["logg1"] = LOGG_SUN + math.log10(M1) - 2 * math.log10(R1_Rsun)
    # R2 and logg2
    if r2 is not None and 0 < r2 < 1:
        R2_Rsun = r2 * a_km / RSUN_KM
        result["R2"] = R2_Rsun
        if R2_Rsun > 0 and M2 > 0:
            result["logg2"] = (
                LOGG_SUN + math.log10(M2) - 2 * math.log10(R2_Rsun)
            )
    return result


def solve_exact(i_deg, K1, M1, R1, P_days):
    """
    Solve for (q, vs, r1) given (i, K1, M1, R1, P) by root-finding
    on the mass function.  Returns (q, vs, r1) or None.
    """
    si = math.sin(i_deg * DEG2RAD)
    if si < 0.01:
        return None

    P_sec = P_days * DAY2SEC
    rhs = 2 * math.pi * G_MSUN * M1 * si**3 / (K1**3 * P_sec)

    # Bisect on log(q) to solve (1+q)²/q³ = rhs
    lo, hi = math.log(1e-4), math.log(1e4)
    for _ in range(200):
        m = 0.5 * (lo + hi)
        q = math.exp(m)
        if (1 + q) ** 2 / q**3 > rhs:
            lo = m
        else:
            hi = m
        if hi - lo < 1e-13:
            break

    q = math.exp(0.5 * (lo + hi))
    vs = K1 * (1 + q) / (q * si)
    a_km = vs * P_sec / (2 * math.pi)
    r1 = R1 * RSUN_KM / a_km

    return (q, vs, r1) if 0 < r1 < 1 else None


def optimise_start(i_deg, P_days, observables, i_free=True):
    """
    Find (i, q, vs, r1) that maximise joint probability given
    observables.  Uses scipy if available, otherwise grid search.

    observables: dict mapping 'K1','K2','M1','M2','R1','R2',
                 'Mt','q_obs','logg1','logg2' -> Measurement
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print(f"  {Y}scipy not available — falling back to grid search{Z}")
        return _fallback_solve(i_deg, P_days, observables)

    def chi2(x):
        i_d = x[0] if i_free else i_deg
        q = math.exp(x[1])
        vs = math.exp(x[2])
        r1 = 1.0 / (1.0 + math.exp(-x[3]))  # sigmoid -> (0,1)
        si = math.sin(i_d * DEG2RAD)
        if si < 1e-6:
            return 1e20

        imp = implied_from_params(i_d, q, vs, r1, P_days)
        c2 = 0.0
        for key in ("K1", "K2", "M1", "M2", "R1"):
            if key in observables:
                c2 += observables[key].pull(imp[key]) ** 2

        # Total mass
        if "Mt" in observables:
            c2 += observables["Mt"].pull(imp["Mt"]) ** 2

        # Mass ratio
        if "q_obs" in observables:
            c2 += observables["q_obs"].pull(q) ** 2

        # logg1 (implied from M1, R1)
        if "logg1" in observables and "logg1" in imp:
            c2 += observables["logg1"].pull(imp["logg1"]) ** 2

        # Mild regularisation: prefer edge-on, penalise low sin(i)
        c2 += 0.01 * ((i_d - 80.0) / 20.0) ** 2
        if si > 0:
            c2 -= 0.5 * math.log(si)
        return c2

    # Initial guess
    q0, vs0, r10 = 1.0, 200.0, 0.2
    if all(k in observables for k in ("K1", "M1", "R1")):
        exact = solve_exact(
            i_deg,
            observables["K1"].value,
            observables["M1"].value,
            observables["R1"].value,
            P_days,
        )
        if exact:
            q0, vs0, r10 = exact

    x0 = [
        i_deg,
        math.log(max(q0, 1e-4)),
        math.log(max(vs0, 1.0)),
        math.log(r10 / max(1e-6, 1.0 - r10)),
    ]
    bounds = [
        (5.0, 89.99) if i_free else (i_deg - 0.001, i_deg + 0.001),
        (math.log(0.01), math.log(100.0)),
        (math.log(10.0), math.log(3000.0)),
        (-6.0, 6.0),
    ]

    best = None
    trial_inclinations = [i_deg] if not i_free else [i_deg, 80, 60, 45, 70]
    for i_try in trial_inclinations:
        xt = list(x0)
        xt[0] = i_try
        try:
            res = minimize(
                chi2,
                xt,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 2000, "ftol": 1e-12},
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    if best is None:
        return i_deg, q0, vs0, r10

    x = best.x
    return (
        x[0] if i_free else i_deg,
        math.exp(x[1]),
        math.exp(x[2]),
        1.0 / (1.0 + math.exp(-x[3])),
    )


def _fallback_solve(i_deg, P_days, obs):
    """Grid search fallback when scipy is unavailable."""
    K1 = obs.get("K1")
    M1 = obs.get("M1")
    R1 = obs.get("R1")
    M2 = obs.get("M2")

    if K1 and M1 and R1:
        best_c2, best_sol = 1e30, None
        n_grid = 15
        K_lo = K1.value - 2 * K1.err_lo
        K_hi = K1.value + 2 * K1.err_hi
        for j in range(n_grid):
            Kt = K_lo + (K_hi - K_lo) * j / (n_grid - 1)
            if Kt <= 0:
                continue
            exact = solve_exact(i_deg, Kt, M1.value, R1.value, P_days)
            if not exact:
                continue
            imp = implied_from_params(i_deg, *exact, P_days)
            c2 = K1.pull(Kt) ** 2
            for key in ("M1", "M2", "R1"):
                if key in obs:
                    c2 += obs[key].pull(imp[key]) ** 2
            if c2 < best_c2:
                best_c2, best_sol = c2, exact
        if best_sol:
            return (i_deg, *best_sol)

    # Last-resort defaults
    q = M2.value / M1.value if (M2 and M1) else 1.0
    vs, r1 = 200.0, 0.2
    if M1 and M2:
        Mt = M1.value + M2.value
        P_sec = P_days * DAY2SEC
        a_km = (G_MSUN * Mt * P_sec**2 / (4 * math.pi**2)) ** (1 / 3)
        vs = 2 * math.pi * a_km / P_sec
        if R1:
            r1 = R1.value * RSUN_KM / a_km
    return i_deg, q, vs, r1


def estimate_r2(M2_est, P_days, vs):
    """Rough r2 from a mass–radius relation."""
    if M2_est is None or M2_est <= 0:
        return 0.30
    R2_Rsun = M2_est**0.8 if M2_est < 1 else M2_est**0.57
    a_km = vs * P_days * DAY2SEC / (2 * math.pi)
    return max(0.01, min(0.95, R2_Rsun * RSUN_KM / a_km))


def beam_factor(T, wl_nm):
    """Doppler beaming factor B(T, λ)."""
    x = 1.4388e7 / (T * wl_nm)
    if x > 500:
        return 5.0
    if x < 0.01:
        return 3.0
    ex = math.exp(x)
    return max(0.1, 6.0 - x * ex / (ex - 1))


def count_data_points(filepath):
    """
    Count non-comment, non-empty lines in a light-curve file.
    Comment lines start with # or !.
    Returns 0 if file cannot be read.
    """
    try:
        n = 0
        with open(filepath) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#") and not s.startswith("!"):
                    n += 1
        return n
    except (OSError, IOError):
        return 0


# ═══════════════════ Claret / VizieR queries ═════════════════


def _claret_table(star_type, purpose, band):
    """
    Determine the correct VizieR table, column names, and filter string.

    Parameters
    ----------
    star_type : str  ('sd', 'wd', 'ms')
    purpose   : str  ('ldc4' or 'gdc')
    band      : str  photometric band name

    Returns
    -------
    (table_id, column_names, filter_value)
        filter_value is None for single-filter tables,
        or a string to match against the 'Filter' column.
    """
    # Filter strings used in Claret multi-filter tables

    filt = FILTER_SUFFIX.get(band)

    TABLES = {
        ("sd", "ldc4"): (
            "J/A+A/634/A93/tablea4",
            ["a1", "a2", "a3", "a4"],
            filt,
        ),
        ("sd", "gdc"): (
            "J/A+A/634/A93/tabley",
            ["y"],
            filt,
        ),
        ("wd", "ldc4"): (
            "J/A+A/641/A157/tablea4",
            ["a1", "a2", "a3", "a4"],
            filt,
        ),
        ("wd", "gdc"): (None, None, None),  # No WD GDC table
    }

    key = (star_type, purpose)
    if key in TABLES:
        return TABLES[key]

    # Main-sequence
    if star_type == "ms":
        # Space-mission dedicated single-filter LDC tables (no Filter column)
        space_tables = {
            "TESS":   "J/A+A/618/A20/TESSa",
            "KEPLER": "J/A+A/618/A20/KEPLERa",
        }
        band_upper = band.upper()

        if band_upper in space_tables and purpose == "ldc4":
            return space_tables[band_upper], ["a1", "a2", "a3", "a4"], None

        if purpose == "ldc4":
            # General MS LDC table — multi-filter
            return "J/A+A/529/A75/tableeq5", ["a1", "a2", "a3", "a4"], filt
        else:
            # General MS GDC table — multi-filter
            return "J/A+A/529/A75/tabley", ["y"], filt

    return None, None, None


def _find_col(colnames, aliases):
    cn_map = {c.lower().strip(): c for c in colnames}
    for a in aliases:
        if a.lower() in cn_map:
            return cn_map[a.lower()]
    return None


def _nearest_row(table, T, logg, tcol, gcol):
    best_d, best_i = 1e30, 0
    for idx in range(len(table)):
        try:
            t_val = float(table[tcol][idx])
        except (ValueError, TypeError):
            continue
        d = abs(t_val - T) / max(T, 1)
        if gcol and logg is not None:
            try:
                d += abs(float(table[gcol][idx]) - logg) / 5.0
            except (ValueError, TypeError):
                pass
        if d < best_d:
            best_d, best_i = d, idx
    return best_i


def _resolve_columns(available_colnames, target_cols):
    avail = {c.strip().lower(): c.strip() for c in available_colnames}
    resolved = []
    for tc in target_cols:
        if tc.lower() in avail:
            resolved.append(avail[tc.lower()])
        else:
            return None
    return resolved


def _query_vizier(table_id, target_cols, T, logg, filter_value=None):
    """
    Query a VizieR table for rows near (T, logg), optionally
    filtering on a 'Filter' column.
    Returns (values_list, matched_Teff, matched_logg) or None.
    """
    result = _query_vizier_astroquery(table_id, target_cols, T, logg,
                                      filter_value)
    if result is not None:
        return result
    return _query_vizier_http(table_id, target_cols, T, logg, filter_value)


def _query_vizier_astroquery(table_id, target_cols, T, logg,
                              filter_value=None):
    """Try fetching via astroquery."""
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        return None

    try:
        dT = max(500, int(0.05 * T))
        constraints = {"Teff": f"{T - dT}..{T + dT}"}
        if logg is not None:
            constraints["logg"] = (
                f"{max(0, logg - 0.5):.1f}..{min(10, logg + 0.5):.1f}"
            )
        if filter_value is not None:
            constraints["Filter"] = f"=={filter_value}"

        v = Vizier(columns=["**"], row_limit=500)
        result = v.query_constraints(catalog=table_id, **constraints)
        if not result or len(result) == 0:
            return None

        tbl = result[0]
        tcol = _find_col(tbl.colnames, TEFF_ALIASES)
        gcol = _find_col(tbl.colnames, LOGG_ALIASES)
        if tcol is None:
            return None

        resolved = _resolve_columns(tbl.colnames, target_cols)
        if resolved is None:
            return None

        bi = _nearest_row(tbl, T, logg, tcol, gcol)
        vals = []
        for c in resolved:
            try:
                vals.append(float(tbl[c][bi]))
            except (ValueError, TypeError):
                return None

        t_match = float(tbl[tcol][bi])
        g_match = float(tbl[gcol][bi]) if gcol else None
        return vals, t_match, g_match
    except Exception as e:
        print(f"    {D}astroquery: {e}{Z}")
        return None


def _query_vizier_http(table_id, target_cols, T, logg, filter_value=None):
    """HTTP/TSV fallback for VizieR queries."""
    try:
        import requests
    except ImportError:
        return None

    try:
        dT = max(500, int(0.05 * T))
        out_cols = ["Teff", "logg"] + target_cols
        if filter_value is not None:
            out_cols.append("Filter")

        params = {
            "-source": table_id,
            "-out": " ".join(out_cols),
            "-out.max": "500",
            "Teff": f"{T - dT}..{T + dT}",
        }
        if logg is not None:
            params["logg"] = (
                f"{max(0, logg - 0.5):.1f}..{min(10, logg + 0.5):.1f}"
            )
        if filter_value is not None:
            params["Filter"] = f"=={filter_value}"

        rsp = requests.get(
            "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
            params=params,
            timeout=20,
        )
        if not rsp.ok:
            return None

        lines = [
            l
            for l in rsp.text.split("\n")
            if l.strip() and not l.startswith("#") and not l.startswith("-")
        ]
        if len(lines) < 2:
            return None

        hdr = {h.strip().lower(): i for i, h in enumerate(lines[0].split("\t"))}

        ti = next(
            (hdr[a.lower()] for a in TEFF_ALIASES if a.lower() in hdr), None
        )
        gi = next(
            (hdr[a.lower()] for a in LOGG_ALIASES if a.lower() in hdr), None
        )
        if ti is None:
            return None

        col_indices = []
        for tc in target_cols:
            if tc.lower() in hdr:
                col_indices.append(hdr[tc.lower()])
            else:
                return None

        # Filter column index (for row-level filtering if server
        # didn't honour the constraint exactly)
        fi = hdr.get("filter")

        best_d, best_vals, best_t, best_g = 1e30, None, None, None
        for ln in lines[1:]:
            fields = ln.split("\t")
            try:
                # Skip rows that don't match the filter
                if filter_value is not None and fi is not None:
                    row_filter = fields[fi].strip()
                    if row_filter != filter_value:
                        continue

                tv = float(fields[ti])
                gv = float(fields[gi]) if gi is not None else 4.5
                d = abs(tv - T) / max(T, 1) + abs(gv - (logg or 4.5)) / 5
                if d < best_d:
                    best_d = d
                    best_vals = [float(fields[ci]) for ci in col_indices]
                    best_t, best_g = tv, gv
            except (ValueError, IndexError):
                continue

        if best_vals is not None:
            return best_vals, best_t, best_g
    except Exception as e:
        print(f"    {D}HTTP: {e}{Z}")

    return None


def _default_ldc(T):
    if T > 20000:
        return [0.26, 0.12, -0.10, 0.03]
    if T > 10000:
        return [0.38, 0.10, -0.05, 0.01]
    if T > 6000:
        return [0.45, 0.15, -0.08, 0.02]
    return [0.55, 0.20, -0.10, 0.03]


def query_ldc(T, logg, star_type, band):
    """Fetch 4-coefficient limb-darkening from Claret. Returns [a1..a4]."""
    table_id, cols, filter_value = _claret_table(star_type, "ldc4", band)
    if table_id is None:
        print(f"    {Y}No LDC table for type={star_type}, band={band}{Z}")
        return _default_ldc(T)

    finfo = f"  Filter={filter_value}" if filter_value else ""
    print(f"    Querying {table_id}  cols={cols}{finfo}")
    result = _query_vizier(table_id, cols, T, logg, filter_value)
    if result:
        vals, t_m, g_m = result
        logg_str = f", logg={g_m:.1f}" if g_m else ""
        print(
            f"    {G}LDC: [{', '.join(f'{v:.4f}' for v in vals)}]"
            f"  (grid Teff={t_m:.0f}{logg_str}){Z}"
        )
        return vals

    print(f"    {Y}Query returned nothing — using defaults{Z}")
    return _default_ldc(T)


def query_gdc(T, logg, star_type, band):
    """Fetch gravity-darkening coefficient y. Returns float."""
    table_id, cols, filter_value = _claret_table(star_type, "gdc", band)
    if table_id is None:
        gd = 0.25 if T > 7500 else 0.08
        print(
            f"    {D}No GDC table for type={star_type} — "
            f"using theoretical β={gd:.2f}{Z}"
        )
        return gd

    finfo = f"  Filter={filter_value}" if filter_value else ""
    print(f"    Querying {table_id}  col={cols[0]}{finfo}")
    result = _query_vizier(table_id, cols, T, logg, filter_value)
    if result:
        vals, t_m, g_m = result
        y = vals[0]
        logg_str = f", logg={g_m:.1f}" if g_m else ""
        print(f"    {G}GDC: y = {y:.4f}  (grid Teff={t_m:.0f}{logg_str}){Z}")
        return y

    gd = 0.25 if T > 7500 else 0.08
    print(f"    {Y}Query returned nothing — using theoretical β={gd:.2f}{Z}")
    return gd


# ═══════════════════ Parameter builder ═══════════════════════


def _make_param(val, rng, step, vary=False, defined=True):
    return ModelParam(val, rng, step, vary, defined).to_str()


def build_model_params(
    q, i, r1, r2, vs, t1, t2, ldc1, ldc2, gd1, gd2, bf1, bf2, t0, varied
):
    is_varied = lambda name: name in varied  # noqa: E731

    p = {}

    # ── Physical parameters ──
    p["q"] = _make_param(
        round(q, 6), max(q, 5), max(0.01, 0.02 * q), is_varied("q")
    )
    p["iangle"] = _make_param(round(i, 4), 45, 1.0, is_varied("iangle"))
    p["r1"] = _make_param(
        round(r1, 6), 0.5, max(0.001, 0.01 * r1), is_varied("r1")
    )
    p["r2"] = _make_param(
        round(r2, 6), 0.5, max(0.002, 0.015 * r2), is_varied("r2")
    )
    p["velocity_scale"] = _make_param(
        round(vs, 4),
        max(vs, 500),
        max(1.0, 0.02 * vs),
        is_varied("velocity_scale"),
    )

    # ── Temperatures ──
    p["t1"] = _make_param(
        round(t1, 3), 15000, max(10, 0.005 * t1), is_varied("t1")
    )
    p["t2"] = _make_param(
        round(t2, 3), 10000, max(10, 0.02 * t2), is_varied("t2")
    )

    # ── Limb darkening ──
    for star_idx, ldc in enumerate([ldc1, ldc2], start=1):
        for j, coeff in enumerate(ldc, start=1):
            name = f"ldc{star_idx}_{j}"
            p[name] = _make_param(round(coeff, 4), 0.5, 0.001, is_varied(name))

    # ── Beaming ──
    p["beam_factor1"] = _make_param(
        round(bf1, 4), 1, 0.01, is_varied("beam_factor1")
    )
    p["beam_factor2"] = _make_param(
        round(bf2, 4), 1, 0.01, is_varied("beam_factor2")
    )

    # ── Ephemeris ──
    p["t0"] = _make_param(round(t0, 8), 0.1, 1e-05, is_varied("t0"))
    p["period"] = _make_param(1, 0.001, 1e-08, False)
    p["pdot"] = _make_param(0, 0.01, 1e-05, False)
    p["deltat"] = _make_param(0, 0.001, 0.0001, False)

    # ── Gravity darkening ──
    p["gravity_dark1"] = _make_param(round(gd1, 4), 0.1, 1e-06, False)
    p["gravity_dark2"] = _make_param(round(gd2, 4), 0.1, 1e-06, False)

    # ── Misc fixed parameters ──
    p["absorb"] = _make_param(1.0, 0.5, 0.01, False)
    p["cphi3"] = _make_param(0.01, 0.05, 0.01, False)
    p["cphi4"] = _make_param(0.055, 0.05, 0.01, False)
    p["spin1"] = _make_param(1, 0.1, 0.01, False)
    p["spin2"] = _make_param(1, 0.1, 0.01, False)

    for name in ("slope", "quad", "cube", "third"):
        p[name] = _make_param(0, 0.01, 1e-05, False)

    # ── Disc parameters ──
    disc_params = [
        ("rdisc1", 0, 0.01, 0.001),
        ("rdisc2", 0, 0.01, 0.02),
        ("height_disc", 0, 0.01, 1e-05),
        ("beta_disc", 0, 0.01, 1e-05),
        ("temp_disc", 0, 50, 40),
        ("texp_disc", 0, 0.2, 0.001),
        ("lin_limb_disc", 0, 0.02, 0.0001),
        ("quad_limb_disc", 0, 0.02, 0.0001),
    ]
    for name, val, rng, step in disc_params:
        p[name] = _make_param(val, rng, step, False)

    # ── Spot parameters ──
    spot_params = [
        ("radius_spot", 0, 0.01, 0.01),
        ("length_spot", 0, 0.01, 0.005),
        ("height_spot", 0, 0.01, 1e-05),
        ("expon_spot", 0, 0.2, 0.1),
        ("epow_spot", 0, 0.01, 0.01),
        ("angle_spot", 0, 5, 2),
        ("yaw_spot", 0, 5, 2),
        ("temp_spot", 0, 500, 200),
        ("tilt_spot", 0, 5, 2),
        ("cfrac_spot", 0, 0.05, 0.008),
    ]
    for name, val, rng, step in spot_params:
        p[name] = _make_param(val, rng, step, False)

    # ── Starspot grid parameters ──
    for star in ("1", "2"):
        for attr in ("long", "lat", "fwhm", "tcen"):
            p[f"stsp{star}1_{attr}"] = _make_param(0, 0, 0, False, False)

    # ── Grid/control scalars ──
    p.update(
        {
            "delta_phase": "1e-07",
            "nlat1f": "50",
            "nlat2f": "150",
            "nlat1c": "50",
            "nlat2c": "150",
            "npole": "1",
            "nlatfill": "2",
            "nlngfill": "2",
            "lfudge": "0",
            "llo": "90",
            "lhi": "-90",
            "phase1": "0.1",
            "phase2": "0.4",
            "roche1": "1",
            "roche2": "1",
            "eclipse1": "1",
            "eclipse2": "1",
            "glens1": "0",
            "use_radii": "1",
            "gdark_bolom1": "1",
            "gdark_bolom2": "1",
            "mucrit1": "0",
            "mucrit2": "0",
            "limb1": "Poly",
            "limb2": "Poly",
            "mirror": "0",
            "add_disc": "0",
            "nrad": "40",
            "opaque": "0",
            "add_spot": "0",
            "nspot": "0",
            "iscale": "0",
        }
    )

    return p


# ═══════════════════ Interactive TUI ═════════════════════════

# ══════════════════════════ Field types ═══════════════════════


class FieldType(Enum):
    TEXT = auto()
    FLOAT = auto()
    INT = auto()
    CHOICE = auto()
    BOOL = auto()
    MEASUREMENT = auto()  # value ± err_lo / + err_hi
    READONLY = auto()
    SEPARATOR = auto()


class FormField:
    """One editable field in a form page."""

    def __init__(
        self,
        key: str,
        label: str,
        ftype: FieldType = FieldType.TEXT,
        default=None,
        choices=None,
        required=False,
        help_text="",
        group=None,
        visible=True,
    ):
        self.key = key
        self.label = label
        self.ftype = ftype
        self.default = default
        self.choices = choices or []
        self.required = required
        self.help_text = help_text
        self.group = group
        self.visible = visible
        self.buf = str(default) if default is not None else ""
        self.choice_idx = 0
        self.error = ""

        if ftype == FieldType.CHOICE and default and default in self.choices:
            self.choice_idx = self.choices.index(default)

        if ftype == FieldType.BOOL:
            self.buf = "Yes" if default else "No"

        if ftype == FieldType.MEASUREMENT and isinstance(default, Measurement):
            self.buf = f"{default.value} {default.err_lo} {default.err_hi}"

    @property
    def editable(self):
        return self.ftype not in (FieldType.READONLY, FieldType.SEPARATOR)

    def get_display(self):
        if self.ftype == FieldType.CHOICE:
            return self.choices[self.choice_idx] if self.choices else ""
        if self.ftype == FieldType.SEPARATOR:
            return ""
        return self.buf

    def get_value(self):
        if self.ftype == FieldType.SEPARATOR:
            return None
        if self.ftype == FieldType.BOOL:
            return self.buf.lower().startswith("y")
        if self.ftype == FieldType.CHOICE:
            return self.choices[self.choice_idx] if self.choices else None
        if self.ftype == FieldType.FLOAT:
            try:
                return float(self.buf) if self.buf.strip() else None
            except ValueError:
                return None
        if self.ftype == FieldType.INT:
            try:
                return int(self.buf) if self.buf.strip() else None
            except ValueError:
                return None
        if self.ftype == FieldType.MEASUREMENT:
            return self._parse_measurement()
        return self.buf if self.buf.strip() else None

    def _parse_measurement(self):
        s = self.buf.strip()
        if not s:
            return None
        parts = s.split()
        try:
            if len(parts) == 1:
                return Measurement(float(parts[0]), 0, 0)
            elif len(parts) == 2:
                v, e = float(parts[0]), abs(float(parts[1]))
                return Measurement(v, e, e)
            elif len(parts) >= 3:
                v = float(parts[0])
                e_lo = abs(float(parts[1]))
                e_hi = abs(float(parts[2]))
                return Measurement(v, e_lo, e_hi)
        except ValueError:
            return None
        return None

    def set_from_value(self, val):
        if val is None:
            self.buf = ""
            return
        if self.ftype == FieldType.BOOL:
            self.buf = "Yes" if val else "No"
        elif self.ftype == FieldType.CHOICE:
            if val in self.choices:
                self.choice_idx = self.choices.index(val)
        elif self.ftype == FieldType.MEASUREMENT:
            if isinstance(val, Measurement):
                self.buf = f"{val.value} {val.err_lo} {val.err_hi}"
            else:
                self.buf = str(val)
        else:
            self.buf = str(val)

    def validate(self):
        if not self.editable:
            return True
        if self.required and not self.buf.strip():
            self.error = "Required"
            return False
        if self.ftype == FieldType.FLOAT and self.buf.strip():
            try:
                float(self.buf)
            except ValueError:
                self.error = "Must be a number"
                return False
        if self.ftype == FieldType.INT and self.buf.strip():
            try:
                int(self.buf)
            except ValueError:
                self.error = "Must be an integer"
                return False
        if self.ftype == FieldType.MEASUREMENT and self.buf.strip():
            if self._parse_measurement() is None:
                self.error = "Format: value [err_lo [err_hi]]"
                return False
        self.error = ""
        return True


class FormPage:
    """One page/screen of the form wizard."""

    def __init__(self, page_id: str, title: str, fields: list,
                 on_leave=None, on_enter=None, description=""):
        self.page_id = page_id
        self.title = title
        self.fields = fields
        self.on_leave = on_leave    # callback(state) -> state, when leaving forward
        self.on_enter = on_enter    # callback(state) -> state, when arriving at page
        self.description = description
        self.status_lines = []
        self.cursor = 0
        self._init_cursor()

    def _init_cursor(self):
        for i, f in enumerate(self.fields):
            if f.editable and f.visible:
                self.cursor = i
                return

    def visible_fields(self):
        return [(i, f) for i, f in enumerate(self.fields) if f.visible]

    def move_cursor(self, direction):
        vf = self.visible_fields()
        editable_indices = [i for i, f in vf if f.editable]
        if not editable_indices:
            return
        try:
            pos = editable_indices.index(self.cursor)
        except ValueError:
            pos = 0
        pos = (pos + direction) % len(editable_indices)
        self.cursor = editable_indices[pos]

    def current_field(self):
        if 0 <= self.cursor < len(self.fields):
            return self.fields[self.cursor]
        return None

    def to_state(self):
        d = {}
        for f in self.fields:
            if f.ftype != FieldType.SEPARATOR:
                d[f.key] = f.get_value()
        return d

    def from_state(self, state):
        for f in self.fields:
            if f.key in state and state[f.key] is not None:
                f.set_from_value(state[f.key])

    def validate_all(self):
        ok = True
        for f in self.fields:
            if f.visible and not f.validate():
                ok = False
        return ok

# ══════════════════════════ TUI Engine ═══════════════════════


class FormApp:
    """Curses-based multi-page form application."""

    def __init__(self, pages: list, state=None):
        self.pages = pages
        self.page_idx = 0
        self.state = state or {}
        self.running = True
        self.message = ""
        self.message_color = 0

    @property
    def current_page(self):
        return self.pages[self.page_idx]

    def save_session(self):
        """Persist state to disk for resume."""
        self._collect_state()
        try:
            with open(AUTOSAVE_PATH, "wb") as f:
                pickle.dump(
                    {"state": self.state, "page_idx": self.page_idx}, f
                )
        except Exception:
            pass

    def _collect_state(self):
        """Pull values from all pages into self.state."""
        for page in self.pages:
            self.state.update(page.to_state())

    def _distribute_state(self):
        """Push state values into all page fields."""
        for page in self.pages:
            page.from_state(self.state)

    def run(self, stdscr):
        curses.curs_set(1)
        curses.use_default_colors()
        self._init_colors()
        stdscr.timeout(-1)
        stdscr.keypad(True)

        self._distribute_state()

        # Register auto-save on exit
        atexit.register(self.save_session)

        while self.running:
            self._draw(stdscr)
            self._handle_input(stdscr)

        atexit.unregister(self.save_session)
        # Clean up autosave on successful completion
        if AUTOSAVE_PATH.exists():
            AUTOSAVE_PATH.unlink()

    def _init_colors(self):
        try:
            curses.init_pair(1, curses.COLOR_CYAN, -1)
            curses.init_pair(2, curses.COLOR_GREEN, -1)
            curses.init_pair(3, curses.COLOR_YELLOW, -1)
            curses.init_pair(4, curses.COLOR_RED, -1)
            curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)
            curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(7, curses.COLOR_WHITE, -1)
            curses.init_pair(8, curses.COLOR_MAGENTA, -1)
        except Exception:
            pass

    def _draw(self, stdscr):
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        page = self.current_page

        # ── Title bar ──
        title_str = (
            f" [{self.page_idx + 1}/{len(self.pages)}] "
            f"{page.title} "
        ).ljust(w)
        try:
            stdscr.addnstr(0, 0, title_str, w, curses.color_pair(6) | curses.A_BOLD)
        except curses.error:
            pass

        # ── Description ──
        row = 2
        if page.description:
            for line in page.description.split("\n"):
                if row < h - 4:
                    try:
                        stdscr.addnstr(row, 2, line[:w - 4], w - 4,
                                       curses.color_pair(1))
                    except curses.error:
                        pass
                    row += 1
            row += 1

        # ── Fields ──
        visible = page.visible_fields()
        label_w = max((len(f.label) for _, f in visible), default=12) + 2
        field_w = max(20, w - label_w - 8)

        cursor_screen_row = row  # track for cursor placement

        for real_idx, fld in visible:
            if row >= h - 5:
                break

            is_active = real_idx == page.cursor
            prefix = " ▸ " if is_active else "   "

            if fld.ftype == FieldType.SEPARATOR:
                try:
                    sep_text = f"{'─' * 3} {fld.label} {'─' * (w - len(fld.label) - 10)}"
                    stdscr.addnstr(row, 1, sep_text[:w - 2], w - 2,
                                   curses.color_pair(1) | curses.A_DIM)
                except curses.error:
                    pass
                row += 1
                continue

            # Label
            label_attr = curses.A_BOLD if is_active else curses.A_NORMAL
            try:
                stdscr.addnstr(row, 1, prefix, 3,
                               curses.color_pair(2) if is_active else curses.A_DIM)
                stdscr.addnstr(row, 4, fld.label.ljust(label_w)[:label_w],
                               label_w, curses.color_pair(7) | label_attr)
            except curses.error:
                pass

            # Value
            val_col = 4 + label_w + 1
            display = fld.get_display()

            if fld.ftype == FieldType.CHOICE:
                choice_str = f"◂ {display} ▸"
                attr = curses.color_pair(2) | curses.A_BOLD if is_active else curses.A_NORMAL
                try:
                    stdscr.addnstr(row, val_col, choice_str[:field_w],
                                   field_w, attr)
                except curses.error:
                    pass
            elif fld.ftype == FieldType.BOOL:
                bool_str = f"[{'✓' if fld.buf.startswith('Y') else ' '}] {fld.buf}"
                attr = curses.color_pair(2) if fld.buf.startswith("Y") else curses.A_DIM
                if is_active:
                    attr |= curses.A_BOLD
                try:
                    stdscr.addnstr(row, val_col, bool_str[:field_w],
                                   field_w, attr)
                except curses.error:
                    pass
            elif fld.ftype == FieldType.READONLY:
                try:
                    stdscr.addnstr(row, val_col, display[:field_w],
                                   field_w, curses.A_DIM)
                except curses.error:
                    pass
            else:
                # Editable text/number/measurement
                if is_active:
                    # Draw input box
                    box = fld.buf.ljust(field_w)[:field_w]
                    try:
                        stdscr.addnstr(row, val_col, box, field_w,
                                       curses.color_pair(5))
                    except curses.error:
                        pass
                    cursor_screen_row = row
                else:
                    disp = display if display else (
                        f"({fld.default})" if fld.default is not None else ""
                    )
                    attr = curses.A_NORMAL if display else curses.A_DIM
                    try:
                        stdscr.addnstr(row, val_col, disp[:field_w],
                                       field_w, attr)
                    except curses.error:
                        pass

            # Help text / error
            if is_active and (fld.help_text or fld.error):
                row += 1
                if row < h - 5:
                    msg = fld.error if fld.error else fld.help_text
                    attr = curses.color_pair(4) if fld.error else curses.A_DIM
                    try:
                        stdscr.addnstr(row, val_col, msg[:field_w],
                                       field_w, attr)
                    except curses.error:
                        pass

            # Required marker
            if fld.required:
                try:
                    stdscr.addnstr(row, val_col + field_w + 1, "*", 1,
                                   curses.color_pair(4))
                except curses.error:
                    pass

            row += 1

        # ── Status lines (computed output) ──
        if page.status_lines:
            row = max(row + 1, h - 4 - len(page.status_lines))
            for sl in page.status_lines:
                if row < h - 3:
                    try:
                        stdscr.addnstr(row, 2, sl[:w - 4], w - 4,
                                       curses.color_pair(8))
                    except curses.error:
                        pass
                    row += 1

        # ── Message bar ──
        if self.message:
            try:
                stdscr.addnstr(h - 3, 2, self.message[:w - 4], w - 4,
                               curses.color_pair(self.message_color))
            except curses.error:
                pass

        # ── Footer ──
        footer = (
            " ↑↓ Navigate  ←→ Prev/Next page  "
            "Enter Confirm  Tab Next field  "
            "Ctrl+S Save  Ctrl+Q Quit "
        )
        try:
            stdscr.addnstr(h - 2, 0, " " * w, w, curses.color_pair(6))
            stdscr.addnstr(h - 2, 0, footer[:w], w, curses.color_pair(6))
        except curses.error:
            pass

        page_ind = " ".join(
            f"[{i + 1}]" if i == self.page_idx else f" {i + 1} "
            for i in range(len(self.pages))
        )
        try:
            stdscr.addnstr(h - 1, 0, " " * w, w, curses.A_DIM)
            stdscr.addnstr(h - 1, (w - len(page_ind)) // 2,
                           page_ind, w, curses.A_DIM)
        except curses.error:
            pass

        # Place cursor in active field
        fld = page.current_field()
        if fld and fld.editable and fld.ftype not in (
            FieldType.CHOICE, FieldType.BOOL
        ):
            cx = 4 + label_w + 1 + len(fld.buf)
            try:
                stdscr.move(cursor_screen_row, min(cx, w - 2))
            except curses.error:
                pass

        stdscr.refresh()

    def _handle_input(self, stdscr):
        page = self.current_page
        fld = page.current_field()

        try:
            ch = stdscr.get_wch()
        except curses.error:
            return

        self.message = ""

        # ── Special keys ──
        if isinstance(ch, int):
            if ch == curses.KEY_UP:
                page.move_cursor(-1)
            elif ch == curses.KEY_DOWN or ch == 9:  # Tab
                page.move_cursor(1)
            elif ch == curses.KEY_LEFT:
                if fld and fld.ftype == FieldType.CHOICE:
                    fld.choice_idx = (fld.choice_idx - 1) % len(fld.choices)
                else:
                    self._go_prev_page()
            elif ch == curses.KEY_RIGHT:
                if fld and fld.ftype == FieldType.CHOICE:
                    fld.choice_idx = (fld.choice_idx + 1) % len(fld.choices)
                else:
                    self._go_next_page()
            elif ch == curses.KEY_BACKSPACE or ch == 127:
                if fld and fld.editable and fld.ftype not in (
                    FieldType.CHOICE, FieldType.BOOL
                ):
                    fld.buf = fld.buf[:-1]
                    fld.error = ""
            elif ch == curses.KEY_DC:  # Delete
                if fld and fld.editable:
                    fld.buf = ""
                    fld.error = ""
            elif ch == curses.KEY_NPAGE:  # Page Down
                self._go_next_page()
            elif ch == curses.KEY_PPAGE:  # Page Up
                self._go_prev_page()
            elif ch == curses.KEY_HOME:
                page._init_cursor()
            elif ch == curses.KEY_RESIZE:
                pass
            return

        # ── Character input ──
        if isinstance(ch, str):
            if ch == "\n" or ch == "\r":
                # Enter: move to next field, or next page if last field
                vf = page.visible_fields()
                editable = [i for i, f in vf if f.editable]
                try:
                    pos = editable.index(page.cursor)
                    if pos < len(editable) - 1:
                        page.move_cursor(1)
                    else:
                        self._go_next_page()
                except ValueError:
                    self._go_next_page()

            elif ch == "\x13":  # Ctrl+S
                self.save_session()
                self.message = "Session saved!"
                self.message_color = 2

            elif ch == "\x11" or ch == "\x03":  # Ctrl+Q or Ctrl+C
                self.save_session()
                self.message = "Saved. Exiting."
                self.running = False

            elif ch == "\x1b":  # Escape
                self._go_prev_page()

            elif ch == "\t":  # Tab
                page.move_cursor(1)

            elif ch == "\x15":  # Ctrl+U: clear field
                if fld and fld.editable:
                    fld.buf = ""

            else:
                if fld and fld.editable:
                    if fld.ftype == FieldType.BOOL:
                        if ch.lower() in ("y", "n"):
                            fld.buf = "Yes" if ch.lower() == "y" else "No"
                        elif ch == " ":
                            fld.buf = "No" if fld.buf.startswith("Y") else "Yes"
                    elif fld.ftype == FieldType.CHOICE:
                        # Type first letter to jump
                        for ci, cv in enumerate(fld.choices):
                            if cv.lower().startswith(ch.lower()):
                                fld.choice_idx = ci
                                break
                    else:
                        fld.buf += ch
                        fld.error = ""

    def _go_next_page(self):
        page = self.current_page
        if not page.validate_all():
            self.message = "Please fix errors before continuing"
            self.message_color = 4
            return

        self._collect_state()
        # Extract T/logg central values from measurement fields
        _extract_teff_logg(self.state)

        # Run on_leave callback
        if page.on_leave:
            try:
                self.state = page.on_leave(self.state)
                self._distribute_state()
                page.status_lines = self.state.pop("__status__", [])
            except Exception as e:
                self.message = f"Error: {e}"
                self.message_color = 4
                return

        self.save_session()

        if self.page_idx < len(self.pages) - 1:
            self.page_idx += 1
            self.pages[self.page_idx].from_state(self.state)
            # Run on_enter callback for the new page
            new_page = self.pages[self.page_idx]
            if new_page.on_enter:
                try:
                    self.state = new_page.on_enter(self.state)
                    self._distribute_state()
                    new_page.from_state(self.state)
                    new_page.status_lines = self.state.pop("__status__", [])
                except Exception as e:
                    self.message = f"Error: {e}"
                    self.message_color = 4
        else:
            self.running = False

    def _go_prev_page(self):
        if self.page_idx > 0:
            self._collect_state()
            _extract_teff_logg(self.state)
            self.page_idx -= 1
            self.pages[self.page_idx].from_state(self.state)
            # Run on_enter for the page we're going back to
            page = self.pages[self.page_idx]
            if page.on_enter:
                try:
                    self.state = page.on_enter(self.state)
                    self._distribute_state()
                    page.from_state(self.state)
                    page.status_lines = self.state.pop("__status__", [])
                except Exception:
                    pass

def _run_solve(state):
    """
    Core solver logic, used by both on_enter and on_leave.
    Returns updated state with results and status lines.
    """
    P = state.get("period")
    if P is None:
        state["__status__"] = ["Period not set — go back to page 1"]
        return state

    obs = _collect_observables_from_state(state)

    i_override = state.get("i_override")
    i_guess = i_override if i_override else 80.0

    status = []

    if len(obs) >= 2:
        i_free = i_override is None
        i_opt, q_opt, vs_opt, r1_opt = optimise_start(
            i_guess, P, obs, i_free=i_free
        )
        status.append(f"Optimised from {len(obs)} constraints")
    elif len(obs) == 1:
        status.append("Only 1 constraint — using defaults + constraint")
        i_opt, q_opt, vs_opt, r1_opt = i_guess, 1.0, 200.0, 0.2
    else:
        status.append("No constraints — using defaults (no priors)")
        i_opt, q_opt, vs_opt, r1_opt = i_guess, 1.0, 200.0, 0.2

    # Estimate r2
    imp = implied_from_params(i_opt, q_opt, vs_opt, r1_opt, P)
    M2_est = None
    m2_obs = state.get("obs_M2")
    if isinstance(m2_obs, Measurement):
        M2_est = m2_obs.value
    elif imp.get("M2"):
        M2_est = imp["M2"]
    r2_opt = estimate_r2(M2_est, P, vs_opt)

    r2_obs = state.get("obs_R2")
    if isinstance(r2_obs, Measurement) and r2_obs.value > 0:
        a_km = vs_opt * P * DAY2SEC / (2 * math.pi)
        r2_opt = r2_obs.value * RSUN_KM / a_km

    state["i"] = round(i_opt, 2)
    state["q"] = q_opt
    state["vs"] = vs_opt
    state["r1"] = r1_opt
    state["r2"] = r2_opt

    state["i_result"] = f"{i_opt:.2f}"
    state["q_result"] = f"{q_opt:.6f}"
    state["vs_result"] = f"{vs_opt:.4f}"
    state["r1_result"] = f"{r1_opt:.6f}"
    state["r2_result"] = f"{r2_opt:.6f}"

    # Implied values
    imp = implied_from_params(i_opt, q_opt, vs_opt, r1_opt, P, r2=r2_opt)
    parts = []
    parts.append(f"K1={imp['K1']:.1f}  K2={imp['K2']:.1f} km/s  "
                 f"M1={imp['M1']:.3f}  M2={imp['M2']:.3f} M☉")
    if "R2" in imp:
        parts.append(f"R1={imp['R1']:.3f}  R2={imp['R2']:.3f} R☉  "
                     f"a={imp['a_Rs']:.2f} R☉")

    for key in ("K1", "K2", "M1", "M2", "R1"):
        if key in obs and key in imp:
            pull = obs[key].pull(imp[key])
            parts.append(f"  {key}: {imp[key]:.4f} [{pull:+.2f}σ]")

    if "q_obs" in obs:
        pull = obs["q_obs"].pull(q_opt)
        parts.append(f"  q: {q_opt:.4f} [{pull:+.2f}σ]")

    status.extend(parts)
    state["__status__"] = status
    return state


def _on_enter_solve(state):
    """Compute results when arriving at the solve page."""
    return _run_solve(state)


def _on_leave_solve(state):
    """Re-compute when leaving (in case user changed i_override)."""
    return _run_solve(state)

# ═══════════════════ Page definitions ════════════════════════

def make_pages():
    """Build all form pages."""
    pages = []

    # ── Page 1: Data & Period ──
    pages.append(FormPage(
        "data", "Data & Period",
        description="Specify the light-curve data file and orbital period.",
        fields=[
            FormField("data_path", "Data file", FieldType.TEXT,
                      required=True,
                      help_text="Path to the light-curve data file"),
            FormField("period", "Period [days]", FieldType.FLOAT,
                      required=True,
                      help_text="Orbital period in days"),
            FormField("_sep1", "Output", FieldType.SEPARATOR),
            FormField("chain_path", "Chain output", FieldType.TEXT,
                      default="chain_out.txt"),
            FormField("output_path", "Model output", FieldType.TEXT,
                      default="output.txt"),
        ],
    ))

    # ── Page 2: Band ──
    band_list = list(BAND_WAVELENGTH.keys())
    pages.append(FormPage(
        "band", "Observation Band",
        description="Select the photometric band of the observations.",
        fields=[
            FormField("band", "Band", FieldType.CHOICE,
                      default="TESS", choices=band_list,
                      help_text="Use ◂ ▸ arrows or type first letter"),
            FormField("wavelength", "λ_eff [nm]", FieldType.READONLY,
                      default="786.5",
                      help_text="Effective wavelength (auto-filled)"),
        ],
        on_leave=_on_leave_band,
    ))

    # ── Page 3: Star 1 ──
    pages.append(FormPage(
        "star1", "Star 1 (Hotter Component)",
        description="Physical properties of the primary star.\n"
                    "All fields optional. Measurement format: value [err_lo [err_hi]]",
        fields=[
            FormField("type1", "Type", FieldType.CHOICE,
                      default="ms", choices=["ms", "sd", "wd"]),
            FormField("T1_meas", "T_eff [K]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]  e.g. 28100 500"),
            FormField("logg1_meas", "log g [cgs]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]  e.g. 5.4 0.2"),
            FormField("_sep_m1", "Mass & Radius", FieldType.SEPARATOR),
            FormField("obs_M1", "M₁ [M☉]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]"),
            FormField("obs_R1", "R₁ [R☉]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]"),
        ],
    ))

    # ── Page 4: Star 2 ──
    pages.append(FormPage(
        "star2", "Star 2 (Cooler Component)",
        description="Physical properties of the secondary star.\n"
                    "All fields optional. Measurement format: value [err_lo [err_hi]]",
        fields=[
            FormField("type2", "Type", FieldType.CHOICE,
                      default="ms", choices=["ms", "sd", "wd"]),
            FormField("T2_meas", "T_eff [K]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]  e.g. 3700 200"),
            FormField("logg2_meas", "log g [cgs]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]  e.g. 4.5 0.3"),
            FormField("_sep_m2", "Mass & Radius", FieldType.SEPARATOR),
            FormField("obs_M2", "M₂ [M☉]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]"),
            FormField("obs_R2", "R₂ [R☉]", FieldType.MEASUREMENT,
                      help_text="value [err_lo [err_hi]]"),
        ],
    ))

    # ── Page 5: Radial Velocities & Constraints ──
    pages.append(FormPage(
        "rv", "Radial Velocities & Constraints",
        description="Observational constraints (all optional).\n"
                    "Format for measurements: value [err_lo [err_hi]]",
        fields=[
            FormField("_sep_rv", "Radial Velocities", FieldType.SEPARATOR),
            FormField("obs_K1", "K₁ [km/s]", FieldType.MEASUREMENT,
                      help_text="Primary RV semi-amplitude"),
            FormField("obs_K2", "K₂ [km/s]", FieldType.MEASUREMENT,
                      help_text="Secondary RV semi-amplitude"),
            FormField("_sep_mass", "Mass Constraints", FieldType.SEPARATOR),
            FormField("obs_M2mn", "M₂_min [M☉]", FieldType.MEASUREMENT,
                      help_text="Minimum companion mass (one-sided prior)"),
            FormField("obs_q", "q = M₂/M₁", FieldType.MEASUREMENT,
                      help_text="Mass ratio with uncertainties"),
            FormField("obs_Mtotal", "M_total [M☉]", FieldType.MEASUREMENT,
                      help_text="Total system mass"),
        ],
    ))

    # ── Page 6: Solve ──
    pages.append(FormPage(
        "solve", "Compute Starting Parameters",
        description="Optimal (i, q, vs, r1) from your constraints.\n"
                    "Results update live as you fill fields.\n"
                    "If no constraints given, defaults are used.",
        fields=[
            FormField("i_override", "Inclination [°]", FieldType.FLOAT,
                      help_text="Leave blank for automatic optimisation"),
            FormField("_sep_res", "Results", FieldType.SEPARATOR),
            FormField("i_result", "i [°]", FieldType.READONLY),
            FormField("q_result", "q", FieldType.READONLY),
            FormField("vs_result", "v_scale [km/s]", FieldType.READONLY),
            FormField("r1_result", "r₁", FieldType.READONLY),
            FormField("r2_result", "r₂", FieldType.READONLY),
        ],
        on_enter=_on_enter_solve,
        on_leave=_on_leave_solve,
    ))

    # ── Page 7: LDC / GDC ──
    pages.append(FormPage(
        "ldc", "Limb & Gravity Darkening",
        description="Coefficients fetched from Claret tables on entry.\n"
                    "Edit any field to override. Query runs only for\n"
                    "empty fields when entering this page.",
        fields=[
            FormField("_sep_s1", "Star 1", FieldType.SEPARATOR),
            FormField("ldc1_1", "LDC1 a₁", FieldType.FLOAT),
            FormField("ldc1_2", "LDC1 a₂", FieldType.FLOAT),
            FormField("ldc1_3", "LDC1 a₃", FieldType.FLOAT),
            FormField("ldc1_4", "LDC1 a₄", FieldType.FLOAT),
            FormField("gd1", "GDC₁ (y)", FieldType.FLOAT),
            FormField("_sep_s2", "Star 2", FieldType.SEPARATOR),
            FormField("ldc2_1", "LDC2 a₁", FieldType.FLOAT),
            FormField("ldc2_2", "LDC2 a₂", FieldType.FLOAT),
            FormField("ldc2_3", "LDC2 a₃", FieldType.FLOAT),
            FormField("ldc2_4", "LDC2 a₄", FieldType.FLOAT),
            FormField("gd2", "GDC₂ (y)", FieldType.FLOAT),
        ],
        on_enter=_on_enter_ldc,
    ))

    # ── Page 8: Beaming & Ephemeris ──
    pages.append(FormPage(
        "beam", "Beaming & Ephemeris",
        description="Beaming factors: enter values manually, or leave\n"
                    "blank and set 'Auto-compute' to Yes to calculate\n"
                    "from T_eff and λ.",
        fields=[
            FormField("_sep_beam", "Beaming Factors", FieldType.SEPARATOR),
            FormField("auto_beaming", "Auto-compute from T_eff?",
                      FieldType.BOOL, default=False,
                      help_text="Yes = compute B(T,λ); No = use values below"),
            FormField("bf1", "B₁", FieldType.FLOAT, default=1.0),
            FormField("bf2", "B₂", FieldType.FLOAT, default=1.0),
            FormField("_sep_eph", "Ephemeris", FieldType.SEPARATOR),
            FormField("t0", "t₀ (eclipse phase)", FieldType.FLOAT,
                      default=0.0),
        ],
        on_leave=_on_leave_beaming,
    ))

    # ── Page 9: MCMC Settings ──
    pages.append(FormPage(
        "mcmc", "MCMC Settings",
        fields=[
            FormField("mcmc_steps", "Total steps", FieldType.INT,
                      default=100000),
            FormField("mcmc_burn", "Burn-in", FieldType.INT,
                      default=25000),
            FormField("mcmc_thin", "Thinning", FieldType.INT,
                      default=1),
            FormField("_sep_adapt", "Adaptation", FieldType.SEPARATOR),
            FormField("adapt_enabled", "Robbins-Monro", FieldType.BOOL,
                      default=True),
            FormField("adapt_covariance", "Covariance adapt", FieldType.BOOL,
                      default=True),
            FormField("target_acc", "Target accept %", FieldType.FLOAT,
                      default=23.4),
            FormField("_sep_anneal", "Annealing", FieldType.SEPARATOR),
            FormField("anneal_enabled", "Anneal burn-in?", FieldType.BOOL,
                      default=True,
                      help_text="Temper χ² during early burn-in for exploration"),
            FormField("anneal_T0", "Initial temperature", FieldType.FLOAT,
                      default=10.0,
                      help_text="Higher = more exploration (1=off)"),
            FormField("_sep_prior", "Prior settings", FieldType.SEPARATOR),
            FormField("use_sin_i", "sin(i) prior", FieldType.BOOL,
                      default=True),
            FormField("prior_weight", "Prior weight", FieldType.FLOAT,
                      help_text="Auto = N_data/N_priors. Set manually to override."),
        ],
    ))

    # ── Page 10: Varied Parameters ──
    pages.append(FormPage(
        "varied", "Parameters to Vary",
        description="Toggle parameters to vary in the MCMC.\n"
                    "Press Space or Y/N to toggle.",
        fields=[
            FormField("vary_q", "q", FieldType.BOOL, default=True),
            FormField("vary_iangle", "iangle", FieldType.BOOL, default=True),
            FormField("vary_r1", "r1", FieldType.BOOL, default=True),
            FormField("vary_r2", "r2", FieldType.BOOL, default=True),
            FormField("vary_vs", "velocity_scale", FieldType.BOOL,
                      default=True),
            FormField("vary_t0", "t0", FieldType.BOOL, default=True),
            FormField("vary_t1", "t1", FieldType.BOOL, default=False,
                      help_text="⚠ LC alone constrains T poorly"),
            FormField("vary_t2", "t2", FieldType.BOOL, default=False),
            FormField("_sep_extra", "Additional", FieldType.SEPARATOR),
            FormField("extra_varied", "Other (comma-sep)", FieldType.TEXT,
                      help_text="e.g. ldc1_1,beam_factor1"),
        ],
    ))

    # ── Page 11: Write Config ──
    pages.append(FormPage(
        "write", "Write Configuration",
        description="Review and write the JSON configuration file.\n"
                    "Press → or Enter on the last field to write.",
        fields=[
            FormField("json_path", "Output JSON path", FieldType.TEXT,
                      default="config.json", required=True),
            FormField("_sep_summary", "Summary (updated after write)",
                      FieldType.SEPARATOR),
            FormField("_summary", "Status", FieldType.READONLY,
                      default="Press Enter/→ to write config"),
        ],
        on_leave=_on_leave_write,
    ))

    return pages


# ═══════════════════ Page callbacks ══════════════════════════

def _on_leave_band(state):
    """Update wavelength when band changes."""
    band = state.get("band", "TESS")
    state["wavelength"] = str(BAND_WAVELENGTH.get(band, 786.5))
    return state


def _on_leave_solve(state):
    """Run the optimiser and fill in results."""
    P = state.get("period")
    if P is None:
        raise ValueError("Period not set — go back to page 1")

    obs = _collect_observables_from_state(state)

    i_override = state.get("i_override")
    i_guess = i_override if i_override else 80.0

    status = []

    if len(obs) >= 2:
        i_free = i_override is None
        i_opt, q_opt, vs_opt, r1_opt = optimise_start(
            i_guess, P, obs, i_free=i_free
        )
    elif all(k in obs for k in ("K1", "M1", "R1")):
        exact = solve_exact(
            i_guess, obs["K1"].value, obs["M1"].value,
            obs["R1"].value, P
        )
        if exact:
            q_opt, vs_opt, r1_opt = exact
            i_opt = i_guess
        else:
            i_opt, q_opt, vs_opt, r1_opt = i_guess, 1.0, 200.0, 0.2
    else:
        status.append("Not enough constraints — using defaults")
        i_opt, q_opt, vs_opt, r1_opt = i_guess, 1.0, 200.0, 0.2

    # Estimate r2
    imp = implied_from_params(i_opt, q_opt, vs_opt, r1_opt, P)
    M2_est = None
    m2_obs = state.get("obs_M2")
    if isinstance(m2_obs, Measurement):
        M2_est = m2_obs.value
    elif imp.get("M2"):
        M2_est = imp["M2"]
    r2_opt = estimate_r2(M2_est, P, vs_opt)

    # Override r2 from obs
    r2_obs = state.get("obs_R2")
    if isinstance(r2_obs, Measurement) and r2_obs.value > 0:
        a_km = vs_opt * P * DAY2SEC / (2 * math.pi)
        r2_opt = r2_obs.value * RSUN_KM / a_km

    state["i"] = round(i_opt, 2)
    state["q"] = q_opt
    state["vs"] = vs_opt
    state["r1"] = r1_opt
    state["r2"] = r2_opt

    state["i_result"] = f"{i_opt:.2f}"
    state["q_result"] = f"{q_opt:.6f}"
    state["vs_result"] = f"{vs_opt:.4f}"
    state["r1_result"] = f"{r1_opt:.6f}"
    state["r2_result"] = f"{r2_opt:.6f}"

    # Compute implied values for status
    imp = implied_from_params(i_opt, q_opt, vs_opt, r1_opt, P, r2=r2_opt)
    status.append(f"K1={imp['K1']:.1f}  K2={imp['K2']:.1f} km/s  "
                  f"M1={imp['M1']:.3f}  M2={imp['M2']:.3f} M☉  "
                  f"R1={imp['R1']:.3f} R☉")
    if "R2" in imp:
        status.append(f"R2={imp['R2']:.3f} R☉  "
                      f"a={imp['a_Rs']:.2f} R☉  Mt={imp['Mt']:.3f} M☉")

    # Show pulls
    for key in ("K1", "K2", "M1", "M2", "R1"):
        if key in obs and key in imp:
            pull = obs[key].pull(imp[key])
            status.append(f"  {key}: {imp[key]:.4f} (obs {obs[key].value}"
                          f" ±{obs[key].err_lo}/{obs[key].err_hi})"
                          f" [{pull:+.2f}σ]")

    state["__status__"] = status
    return state


def _suppress_stdout():
    """Context manager that suppresses all stdout/stderr (for curses safety)."""
    return contextlib.redirect_stdout(io.StringIO()), \
           contextlib.redirect_stderr(io.StringIO())


def _query_ldc_quiet(T, logg, stype, band):
    """Query LDC with all output suppressed."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        try:
            return query_ldc(T, logg, stype, band)
        except Exception:
            return _default_ldc(T)


def _query_gdc_quiet(T, logg, stype, band):
    """Query GDC with all output suppressed."""
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        try:
            return query_gdc(T, logg, stype, band)
        except Exception:
            return 0.25 if T > 7500 else 0.08


def _on_enter_ldc(state):
    """
    Query Claret on page entry for any empty LDC/GDC fields.
    User-filled values are preserved.
    """
    band = state.get("band", "TESS")
    status = []

    for n in ("1", "2"):
        T = state.get(f"T{n}")
        logg = state.get(f"logg{n}")
        stype = state.get(f"type{n}", "ms")

        if T is None:
            # No temperature known — use rough defaults
            ldc_keys = [f"ldc{n}_{j}" for j in range(1, 5)]
            has_ldc = all(state.get(k) is not None for k in ldc_keys)
            if not has_ldc:
                ldc = [0.4, 0.15, -0.05, 0.02]
                for j, v in enumerate(ldc, start=1):
                    if state.get(f"ldc{n}_{j}") is None:
                        state[f"ldc{n}_{j}"] = v
                status.append(f"Star {n}: no T_eff — using generic LDC defaults")
            if state.get(f"gd{n}") is None:
                state[f"gd{n}"] = 0.15
                status.append(f"Star {n}: no T_eff — using GDC=0.15")
            continue

        # LDC: query only if empty
        ldc_keys = [f"ldc{n}_{j}" for j in range(1, 5)]
        has_ldc = all(state.get(k) is not None for k in ldc_keys)
        if not has_ldc:
            status.append(f"Star {n}: querying Claret LDC "
                          f"(T={T:.0f}, {stype}, {band})...")
            ldc = _query_ldc_quiet(T, logg, stype, band)
            for j, v in enumerate(ldc, start=1):
                if state.get(f"ldc{n}_{j}") is None:
                    state[f"ldc{n}_{j}"] = round(v, 4)
            status.append(f"  → [{', '.join(f'{v:.4f}' for v in ldc)}]")
        else:
            status.append(f"Star {n}: LDC already set (keeping)")

        # GDC: query only if empty
        if state.get(f"gd{n}") is None:
            status.append(f"Star {n}: querying Claret GDC...")
            gd = _query_gdc_quiet(T, logg, stype, band)
            state[f"gd{n}"] = round(gd, 4)
            status.append(f"  → y = {gd:.4f}")
        else:
            status.append(f"Star {n}: GDC already set (keeping)")

    state["__status__"] = status
    return state


def _on_leave_beaming(state):
    """Compute beaming factors if auto-compute is set and fields are default/empty."""
    if state.get("auto_beaming"):
        T1 = state.get("T1")
        T2 = state.get("T2")
        wl = float(state.get("wavelength", 786.5))
        if T1:
            state["bf1"] = round(beam_factor(T1, wl), 4)
        if T2:
            state["bf2"] = round(beam_factor(T2, wl), 4)
        state["__status__"] = [
            f"Computed: B1={state.get('bf1', 1.0):.4f}  "
            f"B2={state.get('bf2', 1.0):.4f}"
        ]
    return state


def _on_leave_write(state):
    """Build and write the config JSON."""
    json_path = state.get("json_path", "config.json")

    # Collect varied parameters
    varied = set()
    vary_map = {
        "vary_q": "q",
        "vary_iangle": "iangle",
        "vary_r1": "r1",
        "vary_r2": "r2",
        "vary_vs": "velocity_scale",
        "vary_t0": "t0",
        "vary_t1": "t1",
        "vary_t2": "t2",
    }
    for sk, pname in vary_map.items():
        if state.get(sk):
            varied.add(pname)

    extra = state.get("extra_varied", "")
    if extra:
        for name in extra.split(","):
            name = name.strip()
            if name:
                varied.add(name)

    state["varied"] = varied

    ldc1 = [state.get(f"ldc1_{j}", 0.0) or 0.0 for j in range(1, 5)]
    ldc2 = [state.get(f"ldc2_{j}", 0.0) or 0.0 for j in range(1, 5)]

    t1 = state.get("T1", 10000.0) or 10000.0
    t2 = state.get("T2", 5000.0) or 5000.0

    mp = build_model_params(
        q=state.get("q", 1.0) or 1.0,
        i=state.get("i", 80.0) or 80.0,
        r1=state.get("r1", 0.2) or 0.2,
        r2=state.get("r2", 0.3) or 0.3,
        vs=state.get("vs", 200.0) or 200.0,
        t1=t1,
        t2=t2,
        ldc1=ldc1,
        ldc2=ldc2,
        gd1=state.get("gd1", 0.15) or 0.15,
        gd2=state.get("gd2", 0.08) or 0.08,
        bf1=state.get("bf1", 1.0) or 1.0,
        bf2=state.get("bf2", 1.0) or 1.0,
        t0=state.get("t0", 0.0) or 0.0,
        varied=varied,
    )
    mp["wavelength"] = str(state.get("wavelength", 786.5))
    mp["tperiod"] = str(state.get("period", 1.0))

    priors = _build_priors_from_state(state)
    P = state.get("period", 1.0)

    # ── Compute prior weight from data size ──
    n_priors = len(priors)
    data_path = state.get("data_path", "")
    n_data = count_data_points(data_path) if data_path else 0

    user_pw = state.get("prior_weight")
    # "auto" or None means compute automatically
    if user_pw is not None and user_pw > 0 and user_pw != 1.0:
        # User explicitly set a value — respect it
        prior_weight = user_pw
        pw_source = "user"
    elif n_priors > 0 and n_data > 0:
        # Auto-compute: each prior σ-violation should cost as much
        # as a typical per-point χ² contribution.
        #
        #   Good fit: χ² ≈ N_data, so -0.5·χ² ≈ -N_data/2
        #   One prior at 1σ contributes -0.5 (unweighted)
        #   N_prior priors at 1σ contribute -N_prior/2
        #
        #   We want: weight * N_prior/2 ≈ N_data/2
        #   => weight = N_data / N_prior
        #
        # Clamp to a sensible range to avoid pathological cases.
        prior_weight = max(1.0, min(n_data / max(n_priors, 1), 500.0))
        pw_source = "auto"
    else:
        prior_weight = 1.0
        pw_source = "default"

    cfg = {
        "data_file_path": data_path,
        "time1": 0,
        "time2": 1,
        "ntime": 1000000,
        "expose": 0,
        "ndivide": 1,
        "noise": 0,
        "seed": 42,
        "nfile": 1,
        "output_file_path": state.get("output_path", "output.txt"),
        "plot_device": "qt",
        "residual_offset": 0.0,
        "autoscale": True,
        "sstar1": 1,
        "sstar2": 1,
        "sdisc": 1,
        "sspot": 1,
        "ssfac": 1,
        "mcmc_steps": state.get("mcmc_steps", 100000) or 100000,
        "mcmc_burn_in": state.get("mcmc_burn", 25000) or 25000,
        "mcmc_thin": state.get("mcmc_thin", 1) or 1,
        "chain_out_path": state.get("chain_path", "chain_out.txt"),
        "use_priors": n_priors > 0,
        "true_period": P,
        "use_sin_i_prior": state.get("use_sin_i", True),
        "auto_consistent_init": True,
        "adapt_enabled": state.get("adapt_enabled", True),
        "target_acceptance_rate": (state.get("target_acc", 23.4) or 23.4) / 100.0,
        "adapt_interval": 100,
        "adapt_rate": 1.0,
        "adapt_decay": 0.6,
        "adapt_min_stepscale": 1e-4,
        "adapt_max_stepscale": 1e4,
        "adapt_covariance": state.get("adapt_covariance", True),
        "cov_warmup": max(500, 20 * len(varied)),
        "cov_epsilon": 1e-6,
        "anneal_enabled": state.get("anneal_enabled", True),
        "anneal_T0": state.get("anneal_T0", 10.0) or 10.0,
        "anneal_steps": (state.get("mcmc_burn", 25000) or 25000) // 2,
        "prior_weight": prior_weight,
        "priors": priors,
        "model_parameters": mp,
    }

    with open(json_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Status lines ──
    status = [f"Config written: {json_path}"]
    status.append(f"Varied: {', '.join(sorted(varied))}")

    if n_data > 0:
        status.append(f"Data points: {n_data}")
    else:
        status.append("⚠ Could not read data file — check path")

    if n_priors > 0:
        status.append(
            f"Priors ({n_priors}): {', '.join(sorted(priors.keys()))}"
        )
        status.append(
            f"Prior weight: {prior_weight:.1f} ({pw_source}"
            + (f" — N_data/N_prior = {n_data}/{n_priors}"
               if pw_source == "auto" else "")
            + ")"
        )
        # Explain what this means
        status.append(
            f"  → 1σ prior violation costs "
            f"~{0.5 * prior_weight:.0f} in log-posterior "
            f"(vs ~{0.5 * n_data:.0f} for full χ²)"
        )
    else:
        status.append("Priors: none (pure light-curve fit)")

    state["_summary"] = f"Written to {json_path}"
    state["__status__"] = status
    return state

# ═══════════════════ State helper functions ══════════════════


def _collect_observables_from_state(state):
    """
    Gather Measurement objects from state into a dict for the optimiser.
    Handles both old-style state keys and new measurement-style keys.
    """
    obs = {}
    direct_map = [
        ("K1", "obs_K1"),
        ("K2", "obs_K2"),
        ("M1", "obs_M1"),
        ("M2", "obs_M2"),
        ("R1", "obs_R1"),
        ("R2", "obs_R2"),
        ("Mt", "obs_Mtotal"),
        ("q_obs", "obs_q"),
    ]
    for key, state_key in direct_map:
        val = state.get(state_key)
        if val is not None and isinstance(val, Measurement):
            if val.err_lo > 0 or val.err_hi > 0:
                obs[key] = val

    # logg from measurement fields
    for n, okey in [("1", "logg1"), ("2", "logg2")]:
        val = state.get(f"logg{n}_err")
        if val is not None and isinstance(val, Measurement):
            obs[okey] = val

    return obs

def _extract_teff_logg(state):
    """
    Extract central T_eff and logg values from MEASUREMENT fields
    so they're available as plain floats for LDC queries etc.
    Also populate the error Measurement objects for priors.
    """
    for n in ("1", "2"):
        meas = state.get(f"T{n}_meas")
        if isinstance(meas, Measurement):
            state[f"T{n}"] = meas.value
            if meas.err_lo > 0 or meas.err_hi > 0:
                state[f"T{n}_err"] = meas
            else:
                state[f"T{n}_err"] = None
        elif state.get(f"T{n}") is None:
            state[f"T{n}"] = None
            state[f"T{n}_err"] = None

        meas = state.get(f"logg{n}_meas")
        if isinstance(meas, Measurement):
            state[f"logg{n}"] = meas.value
            if meas.err_lo > 0 or meas.err_hi > 0:
                state[f"logg{n}_err"] = meas
            else:
                state[f"logg{n}_err"] = None
        elif state.get(f"logg{n}") is None:
            state[f"logg{n}"] = None
            state[f"logg{n}_err"] = None


def _build_priors_from_state(state):
    """Collect prior strings from the state."""
    priors = {}
    prior_map = [
        ("obs_K1", "K1"),
        ("obs_K2", "K2"),
        ("obs_M1", "M1"),
        ("obs_M2", "M2"),
        ("obs_M2mn", "M2_min"),
        ("obs_Mtotal", "M_total"),
        ("obs_q", "q"),
        ("obs_R1", "R1"),
        ("obs_R2", "R2"),
        ("logg1_err", "logg1"),
        ("logg2_err", "logg2"),
        ("T1_err", "T1"),
        ("T2_err", "T2"),
    ]
    for state_key, prior_name in prior_map:
        m = state.get(state_key)
        if m is not None and isinstance(m, Measurement):
            if m.err_lo > 0 or m.err_hi > 0:
                priors[prior_name] = m.as_prior_str()
    return priors


def _state_from_existing_config(cfg):
    """Convert an existing JSON config dict back into form state."""
    state = {}

    state["data_path"] = cfg.get("data_file_path")
    state["period"] = cfg.get("true_period")
    state["chain_path"] = cfg.get("chain_out_path", "chain_out.txt")
    state["output_path"] = cfg.get("output_file_path", "output.txt")
    state["mcmc_steps"] = cfg.get("mcmc_steps", 100000)
    state["mcmc_burn"] = cfg.get("mcmc_burn_in", 25000)
    state["mcmc_thin"] = cfg.get("mcmc_thin", 1)
    state["adapt_enabled"] = cfg.get("adapt_enabled", True)
    state["adapt_covariance"] = cfg.get("adapt_covariance", True)
    state["target_acc"] = cfg.get("target_acceptance_rate", 0.234) * 100
    state["use_sin_i"] = cfg.get("use_sin_i_prior", True)

    mp = cfg.get("model_parameters", {})

    def _first_float(key):
        v = mp.get(key, "")
        if isinstance(v, str):
            try:
                return float(v.split()[0])
            except (ValueError, IndexError):
                return None
        return None

    # Core model parameters
    state["q"] = _first_float("q")
    state["i"] = _first_float("iangle")
    state["r1"] = _first_float("r1")
    state["r2"] = _first_float("r2")
    state["vs"] = _first_float("velocity_scale")
    state["t0"] = _first_float("t0")
    state["bf1"] = _first_float("beam_factor1")
    state["bf2"] = _first_float("beam_factor2")
    state["gd1"] = _first_float("gravity_dark1")
    state["gd2"] = _first_float("gravity_dark2")

    for n in ("1", "2"):
        for j in range(1, 5):
            state[f"ldc{n}_{j}"] = _first_float(f"ldc{n}_{j}")

    # T_eff and logg: reconstruct as Measurement from model + priors
    priors = cfg.get("priors", {})

    for n in ("1", "2"):
        t_val = _first_float(f"t{n}")
        state[f"T{n}"] = t_val
        t_prior_key = f"T{n}"
        if t_prior_key in priors:
            v, elo, ehi = Helpers_parseThreeDoubles(priors[t_prior_key])
            state[f"T{n}_meas"] = Measurement(v, elo, ehi)
            state[f"T{n}_err"] = Measurement(v, elo, ehi)
        elif t_val is not None:
            state[f"T{n}_meas"] = Measurement(t_val, 0, 0)

        logg_prior_key = f"logg{n}"
        if logg_prior_key in priors:
            v, elo, ehi = Helpers_parseThreeDoubles(priors[logg_prior_key])
            state[f"logg{n}_meas"] = Measurement(v, elo, ehi)
            state[f"logg{n}"] = v
            state[f"logg{n}_err"] = Measurement(v, elo, ehi)

    # Detect band from wavelength
    wl = _first_float("wavelength")
    if wl:
        state["wavelength"] = str(wl)
        for bname, bwl in BAND_WAVELENGTH.items():
            if abs(bwl - wl) < 1:
                state["band"] = bname
                break

    # Detect varied parameters
    def _is_varied(key):
        v = mp.get(key, "")
        if isinstance(v, str):
            parts = v.split()
            return len(parts) >= 4 and parts[3] == "1"
        return False

    vary_map = {
        "q": "vary_q",
        "iangle": "vary_iangle",
        "r1": "vary_r1",
        "r2": "vary_r2",
        "velocity_scale": "vary_vs",
        "t0": "vary_t0",
        "t1": "vary_t1",
        "t2": "vary_t2",
    }
    extra_varied = []
    for pname, skey in vary_map.items():
        state[skey] = _is_varied(pname)

    known_vary = set(vary_map.keys())
    for pname in mp:
        if _is_varied(pname) and pname not in known_vary:
            extra_varied.append(pname)
    if extra_varied:
        state["extra_varied"] = ",".join(extra_varied)

    # Parse priors -> measurement fields
    prior_to_state = {
        "K1": "obs_K1",
        "K2": "obs_K2",
        "M1": "obs_M1",
        "M2": "obs_M2",
        "M2_min": "obs_M2mn",
        "M_total": "obs_Mtotal",
        "q": "obs_q",
        "R1": "obs_R1",
        "R2": "obs_R2",
    }
    for pname, skey in prior_to_state.items():
        if pname in priors:
            val, elo, ehi = Helpers_parseThreeDoubles(priors[pname])
            state[skey] = Measurement(val, elo, ehi)

    # Fill readonly result fields
    if state.get("i") is not None:
        state["i_result"] = f"{state['i']:.2f}"
    if state.get("q") is not None:
        state["q_result"] = f"{state['q']:.6f}"
    if state.get("vs") is not None:
        state["vs_result"] = f"{state['vs']:.4f}"
    if state.get("r1") is not None:
        state["r1_result"] = f"{state['r1']:.6f}"
    if state.get("r2") is not None:
        state["r2_result"] = f"{state['r2']:.6f}"

    state["anneal_enabled"] = cfg.get("anneal_enabled", True)
    state["anneal_T0"] = cfg.get("anneal_T0", 10.0)
    pw = cfg.get("prior_weight", 1.0)
    # Only set if it was explicitly non-default, so auto-compute
    # can re-run on next write if the user clears it
    if pw != 1.0:
        state["prior_weight"] = pw

    return state

def Helpers_parseThreeDoubles(s):
    """Parse 'val err_lo err_hi' string."""
    parts = s.strip().split()
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]), float(parts[1]), float(parts[1])
    elif len(parts) == 1:
        return float(parts[0]), 0.0, 0.0
    return 0.0, 0.0, 0.0


# ══════════════════════════ Main ═════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="LCURVE MCMC Configuration Generator v6"
    )
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Existing JSON config to load and modify"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any auto-saved session"
    )
    args = parser.parse_args()

    state = {}
    resume_page = 0

    # ── Load existing config from disk ──
    if args.config:
        try:
            with open(args.config) as f:
                cfg = json.load(f)
            state = _state_from_existing_config(cfg)
            state["json_path"] = args.config
            print(f"Loaded existing config: {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    # ── Resume from autosave ──
    elif not args.no_resume and AUTOSAVE_PATH.exists():
        try:
            with open(AUTOSAVE_PATH, "rb") as f:
                saved = pickle.load(f)
            state = saved.get("state", {})
            resume_page = saved.get("page_idx", 0)

            # Ask user (outside curses)
            print(f"\n{C}Found saved session (page {resume_page + 1}).{Z}")
            r = input(f"  Resume? [Y/n]: ").strip().lower()
            if r and not r.startswith("y"):
                state = {}
                resume_page = 0
                AUTOSAVE_PATH.unlink(missing_ok=True)
                print("  Starting fresh.")
            else:
                print(f"  Resuming at page {resume_page + 1}.")
        except Exception:
            state = {}
            resume_page = 0

    pages = make_pages()

    app = FormApp(pages, state)
    app.page_idx = min(resume_page, len(pages) - 1)

    # ── Run the TUI ──
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        app.save_session()
        print(f"\n{Y}Session saved to {AUTOSAVE_PATH}{Z}")
        print(f"Run again to resume, or --no-resume to start fresh.")
        sys.exit(0)

    # ── Post-TUI summary ──
    if app.state.get("_summary"):
        print(f"\n{G}✓ {app.state['_summary']}{Z}")

    json_path = app.state.get("json_path", "config.json")
    if Path(json_path).exists():
        print(f"\n{C}Configuration written to: {json_path}{Z}")
        print(f"Run the MCMC solver with:")
        print(f"  ./mcmc_solver {json_path}\n")


if __name__ == "__main__":
    main()