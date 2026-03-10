# -*- coding: utf-8 -*-
"""
run_Rs_sweep.py
R(s) sweep runner that uses core_fem.py (numerically identical to fitting).
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core_fem import (
    PhysicalParameters, GeometricParameters,
    find_fermi_level, run_fem_simulation, load_results,
    sample_profile_on_z, ring_radius_from_profile,
    build_levels, build_symmetric_R_csv,
    make_run_outdir, make_stdout_tee, dump_json, LOG
)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_list", type=float, nargs=3, default=[3.0, 14.0, 30], help="start stop num for s [nm]")
    ap.add_argument("--Vtip_fixed", type=float, default=-2.0, help="fixed Vtip [V] used for s sweep (display value)")
    ap.add_argument("--sample_z_nm", type=float, default=-5.0)
    ap.add_argument("--tip_radius", type=float, default=46.0)

    ap.add_argument("--Nd_cm3", type=float, default=1e16)
    ap.add_argument("--sigma_s", type=float, default=1e12)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--l_sio2", type=float, default=5.0)
    ap.add_argument("--model", type=str, choices=["Feenstra","Boltzmann","F","B"], default="Feenstra")
    ap.add_argument("--assume_full_ionization", action="store_true")
    ap.add_argument("--CPD", type=float, default=0.6, help="Contact Potential Difference [V] used as V_app=Vtip-CPD")

    ap.add_argument("--fermi_np", type=str, default="aymerich_humet", choices=["aymerich_humet","legacy"])
    ap.add_argument("--out_root", type=str, default="Rs_out", help="root dir; a timestamped run dir will be created under it")

    # levels: list + range (combined)
    ap.add_argument("--levels_list", type=float, nargs="*", default=None, help="explicit levels [V]")
    ap.add_argument("--levels_range", type=float, nargs=3, default=None, metavar=("START","STOP","STEP"),
                    help="range levels [V]: START STOP STEP")

    # logging toggle
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

        s_vals = np.linspace(args.s_list[0], args.s_list[1], int(args.s_list[2]))

        DEFAULT_LEVELS = [0.05, 0.075, 0.09, 0.105, 0.125, 0.13, 0.15, 0.175, 0.2]
        LEVELS = build_levels(args.levels_list, args.levels_range, DEFAULT_LEVELS)

        phys = PhysicalParameters(T=args.T, Nd=args.Nd_cm3*1e6, sigma_s=args.sigma_s*1e4)
        phys.Ef = find_fermi_level(phys, str(run_dir), plot=False, fermi_np=args.fermi_np)
        Feenstra = (args.model[0].upper() == "F")
        V_app = float(args.Vtip_fixed - args.CPD)

        dump_json(run_dir/"args.json", vars(args))
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

        R_dict = {float(lev): [] for lev in LEVELS}
        tip_tag = f"R{args.tip_radius:.2f}".replace(".","p")

        for s in s_vals:
            geom = GeometricParameters(l_sio2=args.l_sio2, tip_radius=float(args.tip_radius), tip_height=float(s))
            subdir = str(run_dir / "sweep_s" / tip_tag / f"s_{s:.3f}")
            os.makedirs(subdir, exist_ok=True)

            print(f"[RUN] s={s:.3f} nm | Vtip_fixed={args.Vtip_fixed:+.3f} V | V_app={V_app:+.3f} V | R_tip={args.tip_radius:.2f} nm")
            V_c = run_fem_simulation(phys, geom, V_tip=V_app, Feenstra=Feenstra,
                                     out_dir=subdir, assume_full_ionization=args.assume_full_ionization)
            msh, u, _ = load_results(subdir, geom, V_c)

            rs, prof = sample_profile_on_z(msh, u, V_c, z_nm=args.sample_z_nm,
                                           r_max_nm=geom.region_radius, npts=1201)
            for lev in LEVELS:
                R_dict[float(lev)].append(ring_radius_from_profile(rs, prof, lev))

        stamp = run_dir.name
        out_csv = Path(run_dir)/f"{stamp}_R_vs_s.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["s (nm)"]+[f"R_{lev:.3f}V" for lev in LEVELS])
            for i, s in enumerate(s_vals):
                w.writerow([s] + [R_dict[float(lev)][i] for lev in LEVELS])
        print("[INFO] saved:", out_csv)

        for lev in LEVELS:
            levf = float(lev)
            p = Path(run_dir)/f"{stamp}_R_vs_s_{levf:.3f}V.csv"
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["s (nm)", f"R (nm) @ {levf:.3f} V"])
                for s, R in zip(s_vals, R_dict[levf]):
                    w.writerow([s, R])
            build_symmetric_R_csv(Path(run_dir)/f"{stamp}_R_vs_s_{levf:.3f}V_SYMM.csv", s_vals, R_dict[levf], levf)

        fig, ax = plt.subplots(figsize=(6,4))
        exp_path = Path("exp_data_z_ring3_Vtip-.csv")
        if exp_path.exists():
            exp_df = pd.read_csv(exp_path).iloc[:, :2]
            ax.plot(exp_df.iloc[:,0], exp_df.iloc[:,1], "ks", label="Experiment")

        for lev in LEVELS:
            levf = float(lev)
            ax.plot(s_vals, R_dict[levf], "o-", ms=2, label=f"{levf*1e3:.0f} mV")
        ax.set_xlabel("s [nm]"); ax.set_ylabel("R [nm]")
        ax.set_title(f"R(s) @ Vtip_fixed={args.Vtip_fixed:+.3f} V (V_app={V_app:+.3f} V)")
        ax.grid(True)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.subplots_adjust(right=0.78)
        fig.savefig(Path(run_dir)/f"{stamp}_R_vs_s.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    main()
