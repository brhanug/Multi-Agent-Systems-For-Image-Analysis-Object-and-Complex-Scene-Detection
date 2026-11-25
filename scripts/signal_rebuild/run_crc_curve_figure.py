#!/usr/bin/env python3
"""How the cut-off is chosen: the calibration curve of Section 6.5 on the fixed split, drawn.

An illustration added after the results (it changes no figure): on the fixed split's calibration
pictures it recomputes, for every candidate cut-off lambda on the grid of the main analyses, the share
of calibration pictures that would be approved and wrong, R(lambda), and the share approved. The
conformal cut-off is the largest lambda whose risk stays within the allowed level
(alpha - 1/(n+1)) * (n+1)/n. The script checks that it reproduces the cut-off of worked_example.json
and writes the figure latex_assets/crc_cutoff_curve.png and its numbers to crc_curve_figure.json.

    python run_crc_curve_figure.py [--verify]
"""
import argparse, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_best_config_repeated as rb

OUT = rb.SR / "crc_curve_figure.json"
FIG = rb.SR.parents[1] / "latex_assets" / "crc_cutoff_curve.png"
ALPHA = 0.05


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--slide-out", help="also write a large-type, wide version for slides to this path"); a = ap.parse_args()
    import run_corrected_tier1_end_to_end as e2e
    idx, nat, _o, scene_abs = e2e.native_vectors(); V = {**nat, "scene": scene_abs}
    sp = pd.read_csv(rb.SPLIT); sp = sp[sp.ppn_id.isin(idx)].copy()
    tr, cal = (sp[sp.split == s] for s in ("train", "calibration"))
    rows = cal.ppn_id.map(idx).values
    P = np.column_stack([e2e.to_probability(V[k], tr.ppn_id.map(idx).values, tr.has_scene.values, rows)
                         for k in ("object", "scene", "vlm")]) @ rb.W
    u = 1 - np.abs(P - 0.5); e = ((P >= 0.5).astype(int) != cal.has_scene.values).astype(int)
    n = len(u); grid = rb.GRID
    risk = (e[None, :] * (u[None, :] < grid[:, None])).mean(axis=1)
    approved = (u[None, :] < grid[:, None]).mean(axis=1)
    allowed = (ALPHA - 1 / (n + 1)) * (n + 1) / n
    lam = rb.lam_for(u, e, ALPHA)
    worked = json.loads((rb.SR / "worked_example.json").read_text())["crc"]["cut_off"]
    assert abs(lam - worked) < 1e-9, (lam, worked)
    i = int(np.argmin(np.abs(grid - lam)))
    out = {"extra": "illustration added after the results; reproduces worked_example.json",
           "calibration_pictures": int(n), "errors_in_calibration": int(e.sum()), "alpha": ALPHA,
           "allowed_risk": round(float(allowed), 4), "cut_off": round(float(lam), 3),
           "risk_at_cut_off": round(float(risk[i]), 4), "approved_at_cut_off": round(float(approved[i]), 4),
           "wrong_and_approved_at_cut_off": int(round(risk[i] * n))}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ink, rust, blue, grey = "#1E2733", "#A2461B", "#2D5F8B", "#8A949E"
    x0 = 0.5
    m = grid >= x0
    ax1.step(grid[m], risk[m], where="post", color=blue, lw=2, label="approved and wrong, $\\hat R(\\lambda)$")
    ax1.axhline(allowed, color=rust, lw=1.5, ls="--", label=f"allowed: {allowed:.3f} (at most {int(np.floor(allowed * n + 1e-9))} of {n})")
    ax1.axvline(lam, color=ink, lw=1.2)
    ax1.annotate(f"$\\hat\\lambda$ = {lam:.3f}", xy=(lam, allowed), xytext=(lam - 0.17, allowed + 0.035),
                 arrowprops=dict(arrowstyle="->", color=ink), color=ink, fontsize=10)
    ax1.set_ylabel("share of calibration\npictures", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.set_ylim(0, max(0.16, float(risk[m].max()) * 1.05))
    ax2.step(grid[m], approved[m], where="post", color=grey, lw=2)
    ax2.axvline(lam, color=ink, lw=1.2)
    ax2.annotate(f"{100 * approved[i]:.1f}% approved", xy=(lam, approved[i]), xytext=(lam + 0.03, approved[i] - 0.25),
                 arrowprops=dict(arrowstyle="->", color=ink), color=ink, fontsize=10)
    ax2.set_ylabel("share approved", fontsize=10)
    ax2.set_xlabel("cut-off $\\lambda$ on the uncertainty $u = 1 - |p - 0.5|$ (approve if $u < \\lambda$)", fontsize=10)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E3E0D8", lw=0.8)
        ax.tick_params(labelsize=9)
    ax2.set_xlim(x0, 1.0); ax2.set_ylim(0, 1.02)
    fig.tight_layout()

    if a.verify:
        same = json.loads(OUT.read_text()) == json.loads(json.dumps(out)); print("ALL FIGURES REPRODUCE" if same else "MISMATCH"); return 0 if same else 1
    fig.savefig(FIG, dpi=200); OUT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=1))
    if a.slide_out:
        plt.rcParams.update({"font.size": 20})
        f2, (b1, b2) = plt.subplots(1, 2, figsize=(16, 6.2))
        b1.step(grid[m], risk[m], where="post", color=blue, lw=3.5)
        b1.axhline(allowed, color=rust, lw=3, ls="--"); b1.axvline(lam, color=ink, lw=2.5)
        b1.text(0.51, allowed + 0.006, f"allowed: {int(np.floor(allowed * n + 1e-9))} of {n}", color=rust, fontsize=20)
        b1.text(lam + 0.012, 0.012, f"λ̂ = {lam:.3f}", color=ink, fontsize=22, fontweight="bold")
        b1.set_title("approved AND wrong", fontsize=22, color=ink, loc="left")
        b1.set_ylim(0, max(0.16, float(risk[m].max()) * 1.05)); b1.set_xlim(x0, 1)
        b2.step(grid[m], approved[m], where="post", color=grey, lw=3.5); b2.axvline(lam, color=ink, lw=2.5)
        b2.plot([lam], [approved[i]], "o", color=ink, ms=11)
        b2.text(lam - 0.02, approved[i] + 0.07, f"{100 * approved[i]:.1f}% approved", color=ink, fontsize=21, ha="right", fontweight="bold")
        b2.set_title("approved (right or wrong)", fontsize=22, color=ink, loc="left"); b2.set_xlim(x0, 1); b2.set_ylim(0, 1.02)
        for ax in (b1, b2):
            ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#DDD6C8", lw=1)
            ax.set_xlabel("cut-off λ  (approve if u < λ)", fontsize=20); ax.set_facecolor("#F6F3EC")
        f2.patch.set_facecolor("#F6F3EC"); f2.tight_layout(); f2.savefig(a.slide_out, dpi=120, facecolor="#F6F3EC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
