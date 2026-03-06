# -*- coding: utf-8 -*-
"""
fit_tip_radius.py
Fit tip radius by comparing simulated ring radii to experiment.

This file is a thin orchestration layer:
  - Uses core_fem.py for ALL numerical work.
  - Ensures identical behavior between:
      * this fitter
      * run_Rs_sweep.py
      * run_RVtip_sweep.py
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core_fem import (
    PhysicalParameters, GeometricParameters,
    find_fermi_level, run_fem_simulation, load_results,
    sample_profile_on_z, ring_radius_from_profile,
    rmse_interp_x_to_y,
    make_run_outdir, make_stdout_tee, dump_json,
    LOG, log_print
)

def main(argv=None):
    ap = argparse.ArgumentParser()

    ap.add_argument("--file_z", type=str, default="", help="CSV for R(s): first col s[nm], second R[nm]")
    ap.add_argument("--file_v", type=str, default="", help="CSV for R(Vtip): first col Vtip[V], second R[nm]")

    ap.add_argument("--tip_radius_list", type=float, nargs="+",
                    default=[*[round(x, 1) for x in np.arange(30, 51, 1)]])

    ap.add_argument("--levels", type=float, nargs=2, default=[-0.2, -1.2])
    ap.add_argument("--n_levels", type=int, default=100)

    ap.add_argument("--sample_z_nm", type=float, default=-5.0)
    ap.add_argument("--Vtip_fixed", type=float, default=-2.0)
    ap.add_argument("--s_grid", type=float, nargs=3, default=[3.0, 14.0, 30], help="start stop num for s sweep")
    ap.add_argument("--v_grid", type=float, nargs=3, default=None, help="start stop num for Vtip sweep")
    ap.add_argument("--use_exp_v_points", action="store_true")
    ap.add_argument("--s_fixed", type=float, default=None, help="fixed s[nm] during Vtip sweep (if None uses s_grid start)")

    ap.add_argument("--Nd_cm3", type=float, default=1e16)
    ap.add_argument("--sigma_s", type=float, default=1e12)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--l_sio2", type=float, default=5.0)
    ap.add_argument("--model", type=str, choices=["Feenstra","Boltzmann","F","B"], default="Feenstra")
    ap.add_argument("--assume_full_ionization", action="store_true")
    ap.add_argument("--CPD", type=float, default=0.6, help="V_app = Vtip - CPD")

    ap.add_argument("--fermi_np", type=str, default="aymerich_humet", choices=["aymerich_humet","legacy"])

    ap.add_argument("--out_root", type=str, default="fit_out", help="root dir; a timestamped run dir will be created under it")
    ap.add_argument("--out_RV", type=str, default="RV_best_R{R}.csv")
    ap.add_argument("--out_Rs", type=str, default="Rs_best_R{R}.csv")
    ap.add_argument("--show_plots", action="store_true")

    ap.add_argument("--enable_log", action="store_true")
    ap.add_argument("--disable_log", action="store_true")

    args, _ = ap.parse_known_args(argv)

    if args.enable_log and args.disable_log:
        raise ValueError("Use only one of --enable_log or --disable_log.")
    if args.enable_log:
        LOG.enabled = True
    elif args.disable_log:
        LOG.enabled = False

    run_dir = make_run_outdir(args.out_root)

    with make_stdout_tee(run_dir):
        print(f"[INFO] run dir: {run_dir}")
        if LOG.enabled:
            print(f"[INFO] stdout tee: {run_dir/'stdout.log'}")
        else:
            print("[INFO] logging is DISABLED")

        dump_json(run_dir/"args.json", vars(args))

        phys = PhysicalParameters(T=args.T, Nd=args.Nd_cm3*1e6, sigma_s=args.sigma_s*1e4)
        phys.Ef = find_fermi_level(phys, str(run_dir), plot=False, fermi_np=args.fermi_np)
        Feenstra = (args.model[0].upper() == "F")

        dump_json(run_dir/"dos_effective_density.json", {
            "タイムスタンプ": run_dir.name,
            "温度_T[K]": float(phys.T),
            "有効状態密度_Nc[m^-3]": float(phys.Nc),
            "有効状態密度_Nv[m^-3]": float(phys.Nv),
            "電子有効質量_m_de/me": 0.4,
            "正孔有効質量_m_dh/me": 2.6,
            "谷縮退度_g_c": float(phys.g_c),
            "谷縮退度_g_v": float(phys.g_v),
            "ドナー濃度_Nd[cm^-3]": float(args.Nd_cm3),
            "界面電荷密度_sigma_s[cm^-2]": float(args.sigma_s),
            "モデル(model)": str(args.model),
            "CPD[V]": float(args.CPD),
            "Fermi積分近似(fermi_np)": str(args.fermi_np),
        })

        have_Z = bool(args.file_z) and os.path.exists(args.file_z)
        have_V = bool(args.file_v) and os.path.exists(args.file_v)
        if not (have_Z or have_V):
            print("No experiment file found. Provide --file_z and/or --file_v.")
            sys.exit(1)

        if have_Z:
            dfZ = pd.read_csv(args.file_z).iloc[:, :2]
            Z_exp  = dfZ.iloc[:,0].to_numpy(float)
            Rz_exp = dfZ.iloc[:,1].to_numpy(float)
            s_arr = np.linspace(args.s_grid[0], args.s_grid[1], int(args.s_grid[2]))
            _validZ = np.isfinite(Z_exp) & np.isfinite(Rz_exp)
            min_valid_Z = max(3, int(0.25 * int(_validZ.sum())))
        else:
            Z_exp = Rz_exp = np.array([])
            s_arr = np.linspace(args.s_grid[0], args.s_grid[1], int(args.s_grid[2]))
            min_valid_Z = 3

        if have_V:
            dfV = pd.read_csv(args.file_v).iloc[:, :2]
            dfV.columns = ["Vtip","R"]
            dfV = dfV.groupby("Vtip", as_index=False).mean().sort_values("Vtip")
            Vtip_exp = dfV["Vtip"].to_numpy(float)
            Rv_exp   = dfV["R"].to_numpy(float)
        else:
            Vtip_exp = np.array([]); Rv_exp = np.array([])

        if have_V:
            if (args.v_grid is not None) and (not args.use_exp_v_points):
                V_grid = np.linspace(args.v_grid[0], args.v_grid[1], int(args.v_grid[2]))
            else:
                V_grid = Vtip_exp.copy()
        else:
            V_grid = np.array([])

        s_fixed_for_V = float(args.s_grid[0]) if (args.s_fixed is None) else float(args.s_fixed)
        LEVEL_GRID = np.linspace(args.levels[0], args.levels[1], args.n_levels)

        all_radius_results = []

        for tip_R in args.tip_radius_list:
            print(f"\n=== Tip radius = {tip_R:.1f} nm ===")
            tipR_tag = f"{tip_R:.1f}".replace(".", "p")

            # s sweep
            R_s_levels = None
            if have_Z:
                R_s_levels = np.full((len(s_arr), len(LEVEL_GRID)), np.nan, dtype=float)
                for i, s in enumerate(s_arr):
                    geom = GeometricParameters(l_sio2=args.l_sio2, tip_radius=tip_R, tip_height=float(s))
                    subdir = str(run_dir / "sweep_s" / f"R{tipR_tag}" / f"s_{s:.3f}")
                    V_app = float(args.Vtip_fixed - args.CPD)
                    log_print(f"[RUN] tipR={tip_R:.1f} nm | mode=s-sweep | s={s:.3f} nm | Vtip_fixed={args.Vtip_fixed:+.3f} V | V_app={V_app:+.3f} V")
                    try:
                        V_c = run_fem_simulation(
                            phys, geom, V_tip=V_app,
                            Feenstra=Feenstra, out_dir=subdir,
                            assume_full_ionization=args.assume_full_ionization
                        )
                        msh, u, _ = load_results(subdir, geom, V_c)
                        rs, prof = sample_profile_on_z(msh, u, V_c, z_nm=args.sample_z_nm,
                                                       r_max_nm=geom.region_radius, npts=1201)
                        for j, lev in enumerate(LEVEL_GRID):
                            R_s_levels[i, j] = ring_radius_from_profile(rs, prof, lev)
                    except Exception as e:
                        print(f"[WARN] s-scan failed (R={tip_R:.1f} nm, s={s:.3f} nm): {e}")

            # V sweep
            R_v_levels = None
            if have_V:
                R_v_levels = np.full((len(V_grid), len(LEVEL_GRID)), np.nan, dtype=float)
                geom = GeometricParameters(l_sio2=args.l_sio2, tip_radius=tip_R, tip_height=s_fixed_for_V)
                for i, Vtip in enumerate(V_grid):
                    subdir = str(run_dir / "sweep_V" / f"R{tipR_tag}" / f"V_{Vtip:+.3f}V")
                    V_app = float(Vtip - args.CPD)
                    log_print(f"[RUN] tipR={tip_R:.1f} nm | mode=V-sweep | s_fixed={s_fixed_for_V:.3f} nm | Vtip={Vtip:+.3f} V | V_app={V_app:+.3f} V")
                    try:
                        V_c = run_fem_simulation(
                            phys, geom, V_tip=V_app,
                            Feenstra=Feenstra, out_dir=subdir,
                            assume_full_ionization=args.assume_full_ionization
                        )
                        msh, u, _ = load_results(subdir, geom, V_c)
                        rs, prof = sample_profile_on_z(msh, u, V_c, z_nm=args.sample_z_nm,
                                                       r_max_nm=geom.region_radius, npts=1201)
                        for j, lev in enumerate(LEVEL_GRID):
                            R_v_levels[i, j] = ring_radius_from_profile(rs, prof, lev)
                    except Exception as e:
                        print(f"[WARN] V-scan failed (R={tip_R:.1f} nm, s_fixed={s_fixed_for_V:.3f} nm, Vtip={Vtip:+.3f} V): {e}")

            # scoring
            best = dict(score=np.inf, level=None, rmse_s=np.inf, rmse_v=np.inf, Rs=None, Rv=None, Vgrid=None)
            trial_rows=[]
            for j, lev in enumerate(LEVEL_GRID):
                rmse_s = 0.0; rmse_v = 0.0
                if have_Z:
                    rmse_s = rmse_interp_x_to_y(s_arr, R_s_levels[:,j], Z_exp, Rz_exp, w=None, min_valid=min_valid_Z)
                if have_V:
                    rmse_v = rmse_interp_x_to_y(V_grid, R_v_levels[:,j], Vtip_exp, Rv_exp, w=None, min_valid=6)
                use_terms = []
                if have_Z: use_terms.append(rmse_s)
                if have_V: use_terms.append(rmse_v)
                score = math.sqrt(np.mean(np.square(use_terms))) if use_terms else np.inf
                trial_rows.append((tip_R, lev, rmse_s, rmse_v, score))
                if np.isfinite(score) and score < best["score"]:
                    best.update(dict(score=float(score), level=float(lev),
                                     rmse_s=float(rmse_s), rmse_v=float(rmse_v),
                                     Rs=None if R_s_levels is None else R_s_levels[:,j].copy(),
                                     Rv=None if R_v_levels is None else R_v_levels[:,j].copy(),
                                     Vgrid=None if R_v_levels is None else V_grid.copy()))

            log_csv = Path(run_dir) / f"{run_dir.name}_fit_log_R{tipR_tag}.csv"
            with open(log_csv, "w", newline="") as f:
                w=csv.writer(f)
                w.writerow(["tip_R[nm]","level[V]","RMSE_s[nm]","RMSE_Vtip[nm]","OBJ[nm]"])
                w.writerows(trial_rows)
            print(f"[INFO] log saved: {log_csv}")

            # guard
            if best["level"] is None or (not np.isfinite(best["score"])):
                print(f"[WARN] No valid best level found for R_tip={tip_R:.1f} nm (all scores invalid).")
                pack = dict(tip_radius=float(tip_R), best_level=None,
                            rmse_s=None if not np.isfinite(best["rmse_s"]) else float(best["rmse_s"]),
                            rmse_v=None if not np.isfinite(best["rmse_v"]) else float(best["rmse_v"]),
                            score=None if not np.isfinite(best["score"]) else float(best["score"]))
                with open(Path(run_dir)/f"{run_dir.name}_best_R{tipR_tag}.json", "w") as f:
                    json.dump(pack, f, indent=2)
                all_radius_results.append(pack)
                continue

            # plot & export
            stamp = run_dir.name
            save_dir = Path(run_dir)

            if have_Z and best["Rs"] is not None:
                fig, ax = plt.subplots(figsize=(6.4,4.3))
                ax.plot(Z_exp, Rz_exp, "ks", label="Experiment R(s)")
                ax.plot(s_arr, best["Rs"], "o-", label=f"Fit (RMSE_s={best['rmse_s']:.3f} nm)")
                ax.set_xlabel("s [nm]"); ax.set_ylabel("R [nm]")
                ax.grid(True); ax.legend()
                out_png = save_dir / f"{stamp}_best_Rs_R{tipR_tag}.png"
                fig.savefig(out_png, dpi=200, bbox_inches="tight")
                if args.show_plots: plt.show()
                plt.close(fig)

                out_rs_csv = save_dir / args.out_Rs.format(R=tipR_tag, stamp=stamp)
                with open(out_rs_csv, "w", newline="") as f:
                    w=csv.writer(f); w.writerow(["s (nm)", "R_model (nm)", "R_exp (nm)"])
                    Rz_interp = np.interp(s_arr, Z_exp, Rz_exp, left=np.nan, right=np.nan)
                    for s, rm, re in zip(s_arr, best["Rs"], Rz_interp):
                        w.writerow([s, rm, re])

            if have_V and (best["Rv"] is not None) and (best["Vgrid"] is not None):
                Vg = best["Vgrid"]
                fig, ax = plt.subplots(figsize=(6.4,4.3))
                ax.plot(Vtip_exp, Rv_exp, "ks", label="Experiment R(Vtip)")
                ax.plot(Vg, best["Rv"], "o-", label=f"Fit (RMSE_V={best['rmse_v']:.3f} nm)")
                ax.set_xlabel("V_tip [V]"); ax.set_ylabel("R [nm]")
                ax.grid(True); ax.legend()
                out_png = save_dir / f"{stamp}_best_RV_R{tipR_tag}.png"
                fig.savefig(out_png, dpi=200, bbox_inches="tight")
                if args.show_plots: plt.show()
                plt.close(fig)

                x_u, idx = np.unique(Vg, return_index=True)
                y_u = best["Rv"][idx]
                R_model_on_exp = np.interp(Vtip_exp, x_u, y_u, left=np.nan, right=np.nan)
                out_rv_csv = save_dir / args.out_RV.format(R=tipR_tag, stamp=stamp)
                with open(out_rv_csv, "w", newline="") as f:
                    w=csv.writer(f); w.writerow(["Vtip (V)", "R_model (nm)", "R_exp (nm)"])
                    for Vt, rm, re in zip(Vtip_exp, R_model_on_exp, Rv_exp):
                        w.writerow([Vt, rm, re])

            pack = dict(tip_radius=float(tip_R), best_level=float(best["level"]),
                        rmse_s=float(best["rmse_s"]), rmse_v=float(best["rmse_v"]),
                        score=float(best["score"]))
            with open(save_dir / f"{stamp}_best_R{tipR_tag}.json", "w") as f:
                json.dump(pack, f, indent=2)
            all_radius_results.append(pack)

        with open(Path(run_dir)/f"{run_dir.name}_summary_over_radii.json","w") as f:
            json.dump(all_radius_results, f, indent=2)
        print("[DONE]")

if __name__ == "__main__":
    main()
