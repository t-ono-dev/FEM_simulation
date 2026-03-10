# -*- coding: utf-8 -*-
"""
core_fem.py
Common FEM + post-processing core shared by:
  - run_Rs_sweep.py
  - run_RVtip_sweep.py
  - fit_tip_radius.py

Design rule (important):
  * All three scripts must go through the EXACT SAME numerical path.
  * Therefore, the following items live ONLY here:
      - PhysicalParameters / GeometricParameters
      - NumPy F_{1/2} approximations + find_fermi_level()
      - create_mesh(), weak form, warm start, homotopy+Newton
      - save/load, sample_profile_on_z(), ring_radius_from_profile()
      - score utilities for fitting
      - output utilities (timestamped run dir, json sanitize)
      - optional stdout tee + verbose printing gate

This file is derived from your "Fittingコード.txt" as the reference implementation.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import ngsolve as ng
from ngsolve.solvers import Newton
from netgen.geom2d import SplineGeometry
from scipy.optimize import brentq

# ---------------------- Logging utility (ON/OFF) ----------------------
class LogConfig:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)

LOG = LogConfig(enabled=True)

def log_print(*args, **kwargs):
    """Print only when LOG.enabled is True."""
    if LOG.enabled:
        print(*args, **kwargs)

class NullContext:
    """A do-nothing context manager."""
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

class TeeStdout:
    """Duplicate stdout to both console and a log file (captures NGSolve warnings)."""
    def __init__(self, logfile_path: Path):
        self.logfile_path = Path(logfile_path)
        self._f = None
        self._old = None

    def __enter__(self):
        self._old = sys.stdout
        self._f = self.logfile_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._old
        if self._f:
            self._f.close()

    def write(self, s):
        self._old.write(s)
        self._f.write(s)

    def flush(self):
        self._old.flush()
        self._f.flush()

def make_stdout_tee(run_dir: Path):
    """Return TeeStdout(run_dir/stdout.log) if LOG.enabled else NullContext()."""
    if not LOG.enabled:
        return NullContext()
    return TeeStdout(Path(run_dir) / "stdout.log")

# ---------------------- Output helpers ----------------------
def make_run_outdir(out_root: str) -> Path:
    """
    Create timestamped run directory under out_root.
    Keeps compatibility with legacy name "fit_out_ringA_Vtip+" (auto-migrate to "fit_out").
    """
    root = Path(out_root)

    # Legacy cleanup/migration
    if root.name == "fit_out_ringA_Vtip+":
        if root.exists() and root.is_dir():
            shutil.rmtree(root)
        root = Path("fit_out")

    root.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def _json_sanitize(obj):
    """Convert non-JSON-serializable objects (numpy scalars/arrays, Path, etc.) to serializable types."""
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    return obj

def dump_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_sanitize)

# ---------------------- Levels utility (list + range) ----------------------
def build_levels(levels_list: Optional[Sequence[float]],
                 levels_range: Optional[Sequence[float]],
                 default_levels: Sequence[float]) -> np.ndarray:
    """
    Combine:
      - explicit list: levels_list
      - range: levels_range = (start, stop, step)
    If both None => default_levels.
    Returns sorted unique float array.
    """
    vals: List[float] = []

    if (levels_list is None) and (levels_range is None):
        vals.extend([float(x) for x in default_levels])

    if levels_list is not None and len(levels_list) > 0:
        vals.extend([float(x) for x in levels_list])

    if levels_range is not None:
        if len(levels_range) != 3:
            raise ValueError("--levels_range expects 3 numbers: START STOP STEP")
        s0, s1, ds = map(float, levels_range)
        if ds <= 0:
            raise ValueError("--levels_range STEP must be > 0")
        if s1 < s0:
            s0, s1 = s1, s0
        n = int(np.floor((s1 - s0) / ds + 1e-12)) + 1
        grid = s0 + ds * np.arange(n, dtype=float)
        grid = grid[grid <= s1 + 1e-12]
        vals.extend(grid.tolist())

    if len(vals) == 0:
        raise ValueError("No levels specified. Use --levels_list and/or --levels_range.")

    lev = np.array(vals, dtype=float)
    lev = lev[np.isfinite(lev)]
    lev = np.unique(np.round(lev, 12))
    lev.sort()
    return lev

# ---------------------- Fermi-Dirac integral F_{1/2} (NumPy only) ----------------------
AH_A = 9.6
AH_B = 2.13
AH_C = 2.4
INV_GAMMA_3_2 = 2.0 / np.sqrt(np.pi)  # 1/Gamma(3/2)

def F_half_aymerich_humet_np(eta: np.ndarray | float) -> np.ndarray:
    """Aymerich–Humet approximation for F_{1/2}(eta) (NumPy)."""
    eta = np.asarray(eta, dtype=float)
    denom_core = (AH_B + eta + (np.abs(eta - AH_B) ** AH_C + AH_A) ** (1.0 / AH_C))
    term1 = (3.0 * np.sqrt(2.0)) / (denom_core ** 1.5)
    term2 = INV_GAMMA_3_2 * np.exp(-eta)
    return 1.0 / (term1 + term2)

def F_half_legacy_np(eta: np.ndarray | float) -> np.ndarray:
    """Legacy piecewise approximation kept from original code."""
    a1 = 6.316
    a2 = 12.92
    C_deg = 0.75224956896
    eta = np.asarray(eta, dtype=float)
    return np.piecewise(
        eta,
        [eta < -10.0],
        [
            lambda x: np.exp(x),
            lambda x: 1.0 / (np.exp(-x) + (C_deg * (x**2 + a1 * x + a2)**0.75)**(-1.0))
        ]
    )

def F_half_np(eta: np.ndarray | float, method: str = "aymerich_humet") -> np.ndarray:
    method = str(method).lower()
    if method in ("aymerich_humet", "ah", "aymerich"):
        return F_half_aymerich_humet_np(eta)
    if method in ("legacy", "old"):
        return F_half_legacy_np(eta)
    raise ValueError(f"Unknown F_half method: {method}")

# ---------------------- Core FEM dataclasses ----------------------
@dataclass
class PhysicalParameters:
    T: float = 300.0
    Nd: float = 1e22
    Na: float = 0.0
    sigma_s: float = 1e16

    # updated effective masses
    m_de: float = 0.40 * const.m_e
    m_dh: float = 2.6  * const.m_e

    # valley degeneracy factors
    g_c: float = 3.0
    g_v: float = 1.0

    Eg: float = 3.26
    Ed_offset_hex: float = 0.124
    Ed_offset_cub: float = 0.066
    Ed_ratio_c_to_h: float = 1.88
    Ea_offset: float = 0.2
    ni: float = 8.2e15

    eps_sic: float = 9.7
    eps_sio2: float = 3.9
    eps_vac: float = 1.0

    n0: float = 0.0
    p0: float = 0.0
    Nc: float = 0.0
    Nv: float = 0.0
    Ec: float = 0.0
    Ev: float = 0.0
    Edh: float = 0.0
    Edc: float = 0.0
    Nd_h: float = 0.0
    Nd_c: float = 0.0
    kTeV: float = 0.0
    Ef: float = 0.0

    def __post_init__(self):
        self.kTeV = const.k * self.T / const.e
        total_ratio = 1.0 + self.Ed_ratio_c_to_h
        self.Nd_h = self.Nd * 1.0 / total_ratio
        self.Nd_c = self.Nd * self.Ed_ratio_c_to_h / total_ratio

        self.n0 = (self.Nd + np.sqrt(self.Nd**2 + 4 * self.ni**2)) / 2
        self.p0 = self.ni**2 / self.n0

        # effective density of states with valley degeneracy
        self.Nc = 2 * self.g_c * (2 * np.pi * self.m_de * const.k * self.T / (const.h**2)) ** 1.5
        self.Nv = 2 * self.g_v * (2 * np.pi * self.m_dh * const.k * self.T / (const.h**2)) ** 1.5

        self.Ev = 0.0
        self.Ec = self.Ev + self.Eg
        self.Edh = self.Ec - self.Ed_offset_hex
        self.Edc = self.Ec - self.Ed_offset_cub

@dataclass
class GeometricParameters:
    L_c: float = 1e-9
    l_sio2: float = 5.0
    tip_radius: float = 45.0
    tip_height: float = 8.0  # tip–sample distance s [nm]
    l_vac: float = 200.0
    region_radius: float = 500.0
    n_tip_arc_points: int = 7
    def __post_init__(self):
        assert self.n_tip_arc_points % 2 == 1, "n_tip_arc_points should be odd"

# ---------------------- Fermi level determination (NumPy F_half) ----------------------
def find_fermi_level(params: PhysicalParameters, out_dir: str,
                     plot: bool=False, fermi_np: str="aymerich_humet") -> float:
    def charge_neutrality_eq(Ef: float) -> float:
        p = params.Nv * F_half_np((params.Ev - Ef) / params.kTeV, method=fermi_np)
        n = params.Nc * F_half_np((Ef - params.Ec) / params.kTeV, method=fermi_np)
        Ndp_h = params.Nd_h / (1 + 2 * np.exp((Ef - params.Edh) / params.kTeV))
        Ndp_c = params.Nd_c / (1 + 2 * np.exp((Ef - params.Edc) / params.kTeV))
        return np.log(p + Ndp_h + Ndp_c) - np.log(n)

    Ef = brentq(charge_neutrality_eq, params.Ev + params.kTeV, params.Ec - params.kTeV)
    if plot:
        _plot_fermi_level_determination(params, Ef, out_dir, fermi_np=fermi_np)
    return Ef

def _plot_fermi_level_determination(params: PhysicalParameters, Ef: float, out_dir: str, fermi_np: str="aymerich_humet"):
    ee = np.linspace(params.Ev - 0.1, params.Ec + 0.1, 500)
    p = params.Nv * F_half_np((params.Ev - ee) / params.kTeV, method=fermi_np)
    n = params.Nc * F_half_np((ee - params.Ec) / params.kTeV, method=fermi_np)
    Ndp_h = params.Nd_h / (1 + 2 * np.exp((ee - params.Edh) / params.kTeV))
    Ndp_c = params.Nd_c / (1 + 2 * np.exp((ee - params.Edc) / params.kTeV))
    plt.figure(figsize=(8,6))
    plt.plot(ee, p + Ndp_h + Ndp_c, label="p + N_D^+")
    plt.plot(ee, n, label="n")
    plt.yscale("log"); plt.axvline(Ef, color="r", ls="-.")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "fermi_level_determination.png"), dpi=150)
    plt.close()

# ---------------------- Mesh / FEM solver (REFERENCE: fitting code) ----------------------
def create_mesh(geom: GeometricParameters):
    L_c = geom.L_c
    R_dimless = geom.region_radius * 1e-9 / L_c
    sio2_depth_dimless = geom.l_sio2 * 1e-9 / L_c
    vac_depth_dimless  = geom.l_vac * 1e-9 / L_c
    tip_z_dimless = geom.tip_height * 1e-9 / L_c
    tip_radius_dimless = geom.tip_radius * 1e-9 / L_c
    tip_arc_angle = 75 * np.pi / 180
    n_middle_points = geom.n_tip_arc_points

    geo = SplineGeometry()
    p1 = geo.AppendPoint(0, -vac_depth_dimless - sio2_depth_dimless)
    p2 = geo.AppendPoint(0, -sio2_depth_dimless)
    origin = geo.AppendPoint(0, 0)

    tip1 = geo.AppendPoint(0, tip_z_dimless)
    tip2 = geo.AppendPoint(tip_radius_dimless*np.sin(tip_arc_angle),
                           tip_z_dimless + tip_radius_dimless*(1-np.cos(tip_arc_angle)))
    tipMlst = [geo.AppendPoint(tip_radius_dimless*np.sin(a),
                               tip_z_dimless + tip_radius_dimless*(1-np.cos(a)))
               for a in np.linspace(0, tip_arc_angle, n_middle_points+2)[1:-1]]

    tip3 = geo.AppendPoint(
        tip_radius_dimless*np.sin(tip_arc_angle) +
        (vac_depth_dimless - tip_z_dimless - tip_radius_dimless*(1-np.cos(tip_arc_angle)))/np.tan(tip_arc_angle),
        vac_depth_dimless
    )

    q1 = geo.AppendPoint(R_dimless, -vac_depth_dimless - sio2_depth_dimless)
    q2 = geo.AppendPoint(R_dimless, -sio2_depth_dimless)
    q3 = geo.AppendPoint(R_dimless, 0)
    q4 = geo.AppendPoint(R_dimless, vac_depth_dimless)

    geo.Append(["line", p1, p2], bc="axis", leftdomain=0, rightdomain=1, maxh=5)
    geo.Append(["line", p2, q2], bc="sic/sio2", leftdomain=2, rightdomain=1, maxh=1)
    geo.Append(["line", q2, q1], bc="far-field", leftdomain=0, rightdomain=1)
    geo.Append(["line", q1, p1], bc="ground", leftdomain=0, rightdomain=1)

    geo.Append(["line", origin, p2], bc="axis", leftdomain=2, rightdomain=0, maxh=0.5)
    geo.Append(["line", q2, q3], bc="far-field", leftdomain=2, rightdomain=0)
    geo.Append(["line", q3, origin], bc="sio2/vacuum", leftdomain=2, rightdomain=3)

    geo.Append(["line", origin, tip1], bc="axis", leftdomain=0, rightdomain=3, maxh=0.5)
    for i in range(0, len(tipMlst), 2):
        pts = [tip1 if i==0 else tipMlst[i-1],
               tipMlst[i],
               tip2 if i==len(tipMlst)-1 else tipMlst[i+1]]
        geo.Append(["spline3", *pts], bc="tip", leftdomain=0, rightdomain=3, maxh=0.5)
    geo.Append(["line", tip2, tip3], bc="tip", leftdomain=0, rightdomain=3)
    geo.Append(["line", tip3, q4], bc="top", leftdomain=0, rightdomain=3)
    geo.Append(["line", q4, q3], bc="far-field", leftdomain=0, rightdomain=3)

    geo.SetMaterial(1, "sic"); geo.SetMaterial(2, "sio2"); geo.SetMaterial(3, "vac")
    ngmesh = geo.GenerateMesh(maxh=10, grading=0.2)
    return ng.Mesh(ngmesh)

def _setup_weak_form(fes, epsilon_r, phys, V_c, L_c, homotopy_charge, homotopy_sigma,
                     geom, msh, Feenstra, assume_full_ionization):
    uh, vh = fes.TnT(); r = ng.x
    C0 = (const.e * L_c**2) / (const.epsilon_0 * V_c)
    Ef_dim, Ec_dim, Ev_dim = phys.Ef / V_c, phys.Ec / V_c, phys.Ev / V_c
    Edh_dim, Edc_dim = phys.Edh / V_c, phys.Edc / V_c
    lambda_ff = 1 / (geom.region_radius * 1e-9 / L_c)
    sigma_s_target = (phys.sigma_s * const.e * L_c) / (const.epsilon_0 * V_c)

    clip_potential, clip_exp = 120.0, 40.0
    def clamp(val, b): return ng.IfPos(val-b, b, ng.IfPos(-b-val, -b, val))
    def safe_exp(x): return ng.exp(clamp(x, clip_exp))

    # NOTE: FEM-side approximation kept as in fitting code
    def fermi_half_ng(x):
        x_clip = clamp(x, clip_exp)
        high = (2/np.sqrt(np.pi))*((2/3)*x_clip**1.5 + (np.pi**2/12)*x_clip**(-0.5))
        low = safe_exp(x_clip)/(1 + 0.27*safe_exp(x_clip))
        return ng.IfPos(x_clip-25.0, high, low)

    u_clip = clamp(uh, clip_potential)

    if Feenstra:
        n_term = C0 * phys.Nc * fermi_half_ng((Ef_dim - Ec_dim) + u_clip)
        p_term = C0 * phys.Nv * fermi_half_ng((Ev_dim - Ef_dim) - u_clip)
        if assume_full_ionization:
            Ndp_term = C0 * phys.Nd
        else:
            Ndp_h = C0 * phys.Nd_h / (1 + 2 * safe_exp((Ef_dim - Edh_dim) + u_clip))
            Ndp_c = C0 * phys.Nd_c / (1 + 2 * safe_exp((Ef_dim - Edc_dim) + u_clip))
            Ndp_term = Ndp_h + Ndp_c
    else:
        n_term = C0 * phys.n0 * safe_exp(u_clip)
        p_term = C0 * phys.p0 * safe_exp(-u_clip)
        Ndp_term = C0 * phys.Nd

    rho_dim = homotopy_charge * (p_term + Ndp_term - n_term)
    sigma_s_dim = homotopy_sigma * sigma_s_target

    a = ng.BilinearForm(fes, symmetric=False)
    a += epsilon_r * ng.grad(uh) * ng.grad(vh) * r * ng.dx
    a += epsilon_r * lambda_ff * uh * vh * r * ng.ds("far-field")
    a += -rho_dim * vh * r * ng.dx(definedon=msh.Materials("sic"))
    a += -sigma_s_dim * vh * r * ng.ds("sic/sio2")
    return a

def _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh):
    w = fes.TrialFunction(); v = fes.TestFunction(); r = ng.x
    lambda_ff = 1 / geom.region_radius
    a_lin = ng.BilinearForm(fes); a_lin += epsilon_r * ng.grad(w) * ng.grad(v) * r * ng.dx
    a_lin += epsilon_r * lambda_ff * w * v * r * ng.ds("far-field")
    f_lin = ng.LinearForm(fes)

    u.vec[:] = 0.0
    for v_val in np.linspace(0.0, V_tip, 5)[1:]:
        uh = ng.GridFunction(fes)
        uh.Set(0, definedon=msh.Boundaries("ground"))
        uh.Set(v_val / V_c, definedon=msh.Boundaries("tip"))
        a_lin.Assemble(); f_lin.Assemble()
        rvec = f_lin.vec.CreateVector(); rvec.data = f_lin.vec - a_lin.mat * uh.vec
        freedofs = fes.FreeDofs()
        freedofs &= ~fes.GetDofs(msh.Boundaries("ground"))
        freedofs &= ~fes.GetDofs(msh.Boundaries("tip"))
        uh.vec.data += a_lin.mat.Inverse(freedofs, inverse="sparsecholesky") * rvec
        u.vec.data = uh.vec

def solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma):
    """
    Homotopy continuation with:
      - fast default initial step (0.1)
      - only the first trial uses a fallback schedule if Newton fails,
        then resumes normal stepping.
    """
    def _stage(param, name,
               step_init=0.1,
               min_step=1e-4,
               first_try_factors=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
               step_grow=1.5,
               step_cap=0.5):

        theta, step = 0.0, float(step_init)

        backup = ng.GridFunction(fes)
        backup.vec.data = u.vec

        freedofs = fes.FreeDofs()
        freedofs &= ~fes.GetDofs(msh.Boundaries("ground"))
        freedofs &= ~fes.GetDofs(msh.Boundaries("tip"))

        newton_kwargs = dict(
            freedofs=freedofs, maxit=50, maxerr=1e-11,
            inverse="sparsecholesky", dampfactor=0.7, printing=False
        )

        def _try_newton(trial_theta: float) -> bool:
            param.Set(trial_theta)
            a.Assemble()
            converged, _ = Newton(a, u, **newton_kwargs)
            if converged < 0:
                return False
            backup.vec.data = u.vec
            return True

        while theta < 1.0 - 1e-12:
            trial = min(1.0, theta + step)

            # first trial only
            if theta == 0.0:
                ok = False
                u.vec.data = backup.vec
                for f in first_try_factors:
                    step_try = step * float(f)
                    if step_try < min_step:
                        continue
                    trial_try = min(1.0, theta + step_try)
                    log_print(f"[Newton] stage={name} theta-> {trial_try:.4f} step={step_try:.4g} (first-trial)")
                    try:
                        u.vec.data = backup.vec
                        ok = _try_newton(trial_try)
                    except Exception:
                        ok = False
                    if ok:
                        theta = trial_try
                        step = min(float(step_init), step_cap)
                        break
                if not ok:
                    raise RuntimeError(f"Homotopy stage '{name}' failed at first trial even after fallback schedule.")
                continue

            # normal stepping
            log_print(f"[Newton] stage={name} theta-> {trial:.4f} step={step:.4g}")
            try:
                ok = _try_newton(trial)
                if not ok:
                    raise RuntimeError("Newton failed")
                theta = trial
                if step < step_cap:
                    step *= step_grow
            except Exception:
                log_print(f"[Newton][FAIL] stage={name} theta={theta:.4f} trial={trial:.4f} step-> {step*0.5:.4g}")
                u.vec.data = backup.vec
                step *= 0.5
                if step < min_step:
                    raise RuntimeError(f"Homotopy stage '{name}' failed.")

    homotopy_sigma.Set(0.0)
    _stage(homotopy_charge, "Space Charge", step_init=0.1)
    homotopy_charge.Set(1.0)
    _stage(homotopy_sigma, "Interface Charge", step_init=0.1)

def save_results(msh, u, epsilon_r, V_c, Feenstra: bool, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    msh.ngmesh.Save(os.path.join(out_dir, "mesh.vol"))
    u_np = u.vec.FV().NumPy()
    np.save(os.path.join(out_dir, "u_dimless.npy"), u_np)
    np.save(os.path.join(out_dir, "u_volts.npy"), u_np * V_c)

    mat_names = list(msh.GetMaterials())
    eps_map = ({name: val for name, val in zip(mat_names, [float(v) for v in epsilon_r.components])}
               if hasattr(epsilon_r, "components") else {name: None for name in mat_names})
    with open(os.path.join(out_dir, "epsilon_r.json"), "w", encoding="utf-8") as f:
        json.dump(eps_map, f, indent=2, ensure_ascii=False)

    meta = {
        "V_c": float(V_c),
        "Feenstra": bool(Feenstra),
        "ndof": int(u.space.ndof),
        "fes": "H1",
        "order": int(u.space.globalorder),
        "materials": mat_names,
        "boundaries": list(msh.GetBoundaries()),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    vtk = ng.VTKOutput(ma=msh, coefs=[u], names=["potential_dimless"],
                       filename=os.path.join(out_dir, "solution"), subdivision=0)
    vtk.Do()

def load_results(out_dir: str, geom: GeometricParameters, V_c: float):
    msh = create_mesh(geom)
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")
    with open(os.path.join(out_dir, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if int(meta["ndof"]) != int(fes.ndof):
        raise RuntimeError(f"DOF mismatch: saved={meta['ndof']} current={fes.ndof}")
    u_vec = np.load(os.path.join(out_dir, "u_dimless.npy"))
    if u_vec.size != fes.ndof:
        raise RuntimeError("Vector length mismatch")
    u.vec.FV().NumPy()[:] = u_vec
    return msh, u, u_vec * V_c

def run_fem_simulation(phys: PhysicalParameters, geom: GeometricParameters, V_tip: float,
                       Feenstra: bool, out_dir: str, assume_full_ionization: bool):
    msh = create_mesh(geom)
    V_c = const.k * phys.T / const.e
    fes = ng.H1(msh, order=1)
    u = ng.GridFunction(fes, name="potential_dimless")
    epsilon_r = ng.CoefficientFunction([phys.eps_sic, phys.eps_sio2, phys.eps_vac])
    homotopy_charge = ng.Parameter(0.0)
    homotopy_sigma  = ng.Parameter(0.0)
    a = _setup_weak_form(fes, epsilon_r, phys, V_c, geom.L_c,
                         homotopy_charge, homotopy_sigma, geom, msh, Feenstra, assume_full_ionization)
    u.Set(0, definedon=msh.Boundaries("ground"))
    u.Set(V_tip / V_c, definedon=msh.Boundaries("tip"))
    _warm_start_with_linear_solve(fes, u, epsilon_r, V_tip, V_c, geom, msh)
    solve_with_homotopy(a, u, fes, msh, homotopy_charge, homotopy_sigma)
    save_results(msh, u, epsilon_r, V_c, Feenstra, out_dir)
    return V_c

# ---------------------- Ring extraction / metrics ----------------------
def sample_profile_on_z(msh, u, V_c, z_nm: float, r_max_nm: float, npts: int=1001):
    rs = np.linspace(0.0, r_max_nm, npts)
    prof = np.full_like(rs, np.nan, dtype=float)
    for i, rr in enumerate(rs):
        try:
            prof[i] = float(u(msh(rr, z_nm))) * V_c
        except Exception:
            prof[i] = np.nan
    return rs, prof

def ring_radius_from_profile(rs, prof, level, eps=1e-12):
    rs = np.asarray(rs, float); prof = np.asarray(prof, float); lev = float(level)
    dif = prof - lev
    m_pair = np.isfinite(dif[:-1]) & np.isfinite(dif[1:])
    chg = ((dif[:-1] <= 0) & (dif[1:] >= 0)) | ((dif[:-1] >= 0) & (dif[1:] <= 0))
    idx = np.where(m_pair & chg)[0]
    if idx.size == 0:
        return np.nan
    roots = []
    for i in idx:
        r0, r1 = rs[i], rs[i+1]; y0, y1 = prof[i], prof[i+1]
        if not np.isfinite(y0+y1) or (y1 == y0):
            continue
        r = r0 + (lev-y0) * (r1-r0) / (y1-y0)
        if r >= -eps:
            roots.append(max(0.0, float(r)))
    return float(min(roots)) if roots else np.nan

# ---------------------- Scoring utilities ----------------------
def weighted_rmse(y_true, y_pred, w=None, min_valid=6):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    if w is None:
        w = np.ones_like(y_true, float)
    else:
        w = np.asarray(w, float)
    m = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(w)) & (w > 0)
    if m.sum() < min_valid:
        return np.inf
    d = y_true[m] - y_pred[m]
    return float(np.sqrt(np.sum(w[m]*d*d) / np.sum(w[m])))

def rmse_interp_x_to_y(model_x, model_y, exp_x, exp_y, w=None, min_valid=6):
    model_x = np.asarray(model_x, float); model_y = np.asarray(model_y, float)
    exp_x = np.asarray(exp_x, float); exp_y = np.asarray(exp_y, float)
    m_mod = np.isfinite(model_x) & np.isfinite(model_y)
    if m_mod.sum() < 2:
        return np.inf
    xx, yy = model_x[m_mod], model_y[m_mod]
    o = np.argsort(xx); xx, yy = xx[o], yy[o]
    x_u, idx = np.unique(xx, return_index=True); y_u = yy[idx]
    xmin, xmax = float(x_u.min()), float(x_u.max())
    m_exp = np.isfinite(exp_x) & np.isfinite(exp_y) & (exp_x >= xmin) & (exp_x <= xmax)
    if m_exp.sum() < min_valid:
        return np.inf
    y_on = np.interp(exp_x[m_exp], x_u, y_u)
    mfin = np.isfinite(y_on) & np.isfinite(exp_y[m_exp])
    if mfin.sum() < min_valid:
        return np.inf
    w_use = None if w is None else np.asarray(w, float)[m_exp][mfin]
    return weighted_rmse(exp_y[m_exp][mfin], y_on[mfin], w=w_use, min_valid=min_valid)

def build_symmetric_R_csv(path, x_vals, R_vals, levels):
    records = []
    for xv, rv in zip(x_vals, R_vals):
        if not (np.isfinite(xv) and np.isfinite(rv)):
            continue
        rabs = abs(float(rv))
        records.append((-rabs, xv))
        records.append((+rabs, xv))
    if records:
        arr = np.asarray(records, float)
        arr = arr[np.argsort(arr[:,0])]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"R (nm)_{levels:.3f}", f"x (unit)_{levels:.3f}"])
            w.writerows(arr.tolist())
