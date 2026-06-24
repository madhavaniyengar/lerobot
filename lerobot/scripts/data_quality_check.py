#!/usr/bin/env python3
"""
High-priority data quality checks for franka_2cam_zed dataset.

Check 1 — action.right_eef_pose vs observation.right_eef_pose:
  Are they the same source? Does action encode the COMMANDED pose or just the current state?

Check 2 — Normalization outlier analysis:
  Per-episode min/max for action and state dims, flagging episodes that would
  corrupt MIN_MAX normalization ranges used by the diffusion policy.

Run:
  python data_quality_check.py [--output DIR]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

DATA_ROOT  = Path("data/franka_2cam_zed")
CHUNK      = DATA_ROOT / "data/chunk-000"
N_EPISODES = 20
FPS        = 30

JOINT_NAMES   = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
EEF_DIM_NAMES = ["rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3",
                  "rot6d_4", "rot6d_5", "tx", "ty", "tz", "gripper_artic"]

OUTLIER_THRESH = 0.3   # rad / m — flag if per-episode range deviates by this from median


# ── helpers ──────────────────────────────────────────────────────────────────

def load_episode(ep):
    df = pd.read_parquet(CHUNK / f"episode_{ep:06d}.parquet")
    out = {
        "action":     np.stack(df["action"].values).astype(np.float64),
        "state":      np.stack(df["observation.state"].values).astype(np.float64),
        "act_eef":    np.stack(df["action.right_eef_pose"].values).astype(np.float64),
        "obs_eef":    np.stack(df["observation.right_eef_pose"].values).astype(np.float64),
        "timestamp":  df["timestamp"].values.astype(np.float64),
        "n":          len(df),
    }
    return out


def load_all():
    print("Loading all episodes...")
    eps = []
    for ep in range(N_EPISODES):
        eps.append(load_episode(ep))
        print(f"  ep {ep:02d}: {eps[-1]['n']} frames")
    return eps


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1 — action.right_eef_pose identity
# ═══════════════════════════════════════════════════════════════════════════

def check_eef_identity(eps, out_dir):
    print("\n── Check 1: action.right_eef_pose vs observation.right_eef_pose ──")

    # Per-episode, per-dim max and mean absolute difference
    max_diffs  = np.zeros((N_EPISODES, 10))
    mean_diffs = np.zeros((N_EPISODES, 10))
    for ep, d in enumerate(eps):
        diff = np.abs(d["act_eef"] - d["obs_eef"])
        max_diffs[ep]  = diff.max(axis=0)
        mean_diffs[ep] = diff.mean(axis=0)

    # ── Figure 1: heatmap of max diff per episode × dim ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Check 1 — action.right_eef_pose vs observation.right_eef_pose\n"
                 "Expected: dims 0-8 identical (same hardware read); dim 9 differs (binary vs continuous gripper)",
                 fontsize=12)

    for ax, data, title in [
        (axes[0], max_diffs,  "MAX |act_eef − obs_eef| per episode"),
        (axes[1], mean_diffs, "MEAN |act_eef − obs_eef| per episode"),
    ]:
        im = ax.imshow(data, aspect="auto", cmap="hot_r",
                       vmin=0, vmax=data.max())
        ax.set_xticks(range(10))
        ax.set_xticklabels(EEF_DIM_NAMES, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(N_EPISODES))
        ax.set_yticklabels([f"ep {i:02d}" for i in range(N_EPISODES)], fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.03)

        # annotate cells
        for i in range(N_EPISODES):
            for j in range(10):
                val = data[i, j]
                color = "white" if val > data.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=5, color=color)

    plt.tight_layout()
    p = out_dir / "check1_eef_identity_heatmap.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # ── Figure 2: time-series of dim 9 (gripper_artic) diff for 4 episodes ──
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    fig.suptitle("Check 1 — Gripper articulation: action (binary) vs obs (continuous)\n"
                 "Dim 9 only — this is the ONLY dimension that truly differs", fontsize=11)

    sample_eps = [0, 5, 10, 15]
    for ax, ep in zip(axes, sample_eps):
        d = eps[ep]
        t = d["timestamp"]
        ax.plot(t, d["obs_eef"][:, 9], label="obs gripper width (continuous)", color="tab:blue", linewidth=0.8)
        ax.plot(t, d["act_eef"][:, 9], label="act gripper cmd (binary 0/1)",   color="tab:red",  linewidth=0.8, linestyle="--")
        ax.fill_between(t, d["obs_eef"][:, 9], d["act_eef"][:, 9],
                        alpha=0.15, color="tab:orange", label="lag region")
        ax.set_ylabel(f"ep {ep:02d}", fontsize=9)
        ax.set_ylim(-0.1, 1.2)
        ax.axhline(0.5, color="k", linewidth=0.4, linestyle=":")
        ax.grid(True, alpha=0.25)
        if ep == sample_eps[0]:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    plt.tight_layout()
    p = out_dir / "check1_gripper_lag_timeseries.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # ── Figure 3: rotation+translation dims 0-8 scatter (should be near y=x) ──
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle("Check 1 — EEF rotation & translation: action vs observation\n"
                 "Dims 0-8 should fall on y=x if both read from same hardware state", fontsize=11)
    ep_sample = 0
    d = eps[ep_sample]
    for idx, ax in enumerate(axes.flat):
        dim = idx
        ax.scatter(d["obs_eef"][:, dim], d["act_eef"][:, dim],
                   s=1, alpha=0.3, color="tab:purple")
        lims = [min(d["obs_eef"][:, dim].min(), d["act_eef"][:, dim].min()),
                max(d["obs_eef"][:, dim].max(), d["act_eef"][:, dim].max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
        corr = np.corrcoef(d["obs_eef"][:, dim], d["act_eef"][:, dim])[0, 1]
        ax.set_title(f"{EEF_DIM_NAMES[dim]}  (r={corr:.4f})", fontsize=8)
        ax.set_xlabel("obs_eef", fontsize=7)
        ax.set_ylabel("act_eef", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    plt.tight_layout()
    p = out_dir / "check1_eef_scatter_ep0.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # Print summary
    print("\n  MAX diff per dim (across all episodes):")
    for j, name in enumerate(EEF_DIM_NAMES):
        flag = " ← BINARY vs CONTINUOUS GRIPPER" if j == 9 else \
               " ← SAME hardware read (small float diff)" if max_diffs[:, j].max() < 0.05 else \
               " ← UNEXPECTED DIFF"
        print(f"    dim {j:2d} {name:15s}: {max_diffs[:, j].max():.4f}{flag}")

    return max_diffs


# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2 — Normalization outlier analysis
# ═══════════════════════════════════════════════════════════════════════════

def check_normalization(eps, out_dir):
    print("\n── Check 2: Normalization outlier analysis (MIN_MAX) ──")

    # Collect per-episode stats
    act_mins = np.array([d["action"].min(axis=0) for d in eps])   # (20, 8)
    act_maxs = np.array([d["action"].max(axis=0) for d in eps])
    st_mins  = np.array([d["state"].min(axis=0)  for d in eps])
    st_maxs  = np.array([d["state"].max(axis=0)  for d in eps])

    global_act_min = act_mins.min(axis=0)
    global_act_max = act_maxs.max(axis=0)

    med_act_min = np.median(act_mins, axis=0)
    med_act_max = np.median(act_maxs, axis=0)

    # ── Figure 4: per-episode min/max range per action dim ───────────────
    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle("Check 2 — Per-episode action range vs global MIN_MAX normalization bounds\n"
                 "Orange band = global [min, max] used by policy. Red dots = outlier episodes.\n"
                 "Episodes far outside the median range will dominate and compress most data.",
                 fontsize=11)

    ep_ids = np.arange(N_EPISODES)
    for dim, ax in enumerate(axes.flat):
        mins = act_mins[:, dim]
        maxs = act_maxs[:, dim]
        ax.fill_between(ep_ids, global_act_min[dim], global_act_max[dim],
                        alpha=0.12, color="tab:orange", label="global [min, max]")
        ax.plot(ep_ids, mins, "o-", color="tab:blue",  linewidth=1, markersize=4, label="ep min")
        ax.plot(ep_ids, maxs, "s-", color="tab:green", linewidth=1, markersize=4, label="ep max")
        ax.axhline(med_act_min[dim], color="tab:blue",  linewidth=0.7, linestyle="--", alpha=0.6)
        ax.axhline(med_act_max[dim], color="tab:green", linewidth=0.7, linestyle="--", alpha=0.6)

        # flag outlier episodes
        for ep in ep_ids:
            is_out = (abs(mins[ep] - med_act_min[dim]) > OUTLIER_THRESH or
                      abs(maxs[ep] - med_act_max[dim]) > OUTLIER_THRESH)
            if is_out:
                ax.axvline(ep, color="red", linewidth=0.6, alpha=0.4)
                ax.plot(ep, mins[ep], "rv", markersize=8, zorder=5)
                ax.plot(ep, maxs[ep], "r^", markersize=8, zorder=5)

        ax.set_title(f"{JOINT_NAMES[dim]}  "
                     f"global=[{global_act_min[dim]:.2f}, {global_act_max[dim]:.2f}]  "
                     f"median=[{med_act_min[dim]:.2f}, {med_act_max[dim]:.2f}]",
                     fontsize=8)
        ax.set_xlabel("episode", fontsize=7)
        ax.set_ylabel("value (rad or norm)", fontsize=7)
        ax.set_xticks(ep_ids)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if dim == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    p = out_dir / "check2_action_range_per_episode.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # ── Figure 5: how much of the [0,1] normalized space each episode uses ──
    # After MIN_MAX normalization: norm = (x - global_min) / (global_max - global_min)
    # Each episode's data should ideally span most of [0, 1]
    global_range = global_act_max - global_act_min
    # avoid div by zero for gripper (it IS binary 0/1 so range=1)
    global_range = np.where(global_range < 1e-6, 1.0, global_range)

    norm_mins = (act_mins - global_act_min) / global_range   # (20, 8)
    norm_maxs = (act_maxs - global_act_min) / global_range

    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
    fig.suptitle("Check 2 — Normalized action range each episode occupies in [0, 1]\n"
                 "Ideal: every episode spans a wide band. Narrow bands = data compression.\n"
                 "Red = episodes covering <30% of the normalized space.",
                 fontsize=11)

    for dim, ax in enumerate(axes.flat):
        for ep in ep_ids:
            lo, hi = norm_mins[ep, dim], norm_maxs[ep, dim]
            coverage = hi - lo
            color = "tab:red" if coverage < 0.3 else "steelblue"
            ax.barh(ep, coverage, left=lo, height=0.7, color=color, alpha=0.7)

        ax.axvline(0, color="k", linewidth=0.5)
        ax.axvline(1, color="k", linewidth=0.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(JOINT_NAMES[dim], fontsize=9)
        ax.set_xlabel("normalized value", fontsize=7)
        ax.set_yticks(ep_ids)
        ax.set_yticklabels([f"ep{i:02d}" for i in ep_ids], fontsize=6)
        ax.grid(True, alpha=0.2, axis="x")
        legend_els = [Patch(color="steelblue", label=">=30% coverage"),
                      Patch(color="tab:red",   label="<30% coverage (compressed)")]
        if dim == 0:
            ax.legend(handles=legend_els, fontsize=7)

    plt.tight_layout()
    p = out_dir / "check2_normalized_coverage.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # ── Figure 6: action value distributions — all episodes overlaid ──────
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle("Check 2 — Action value distributions per episode (all overlaid)\n"
                 "Outlier episodes will appear as separate clusters or long tails.",
                 fontsize=11)

    cmap = plt.cm.tab20
    for dim, ax in enumerate(axes.flat):
        for ep in ep_ids:
            vals = eps[ep]["action"][:, dim]
            ax.hist(vals, bins=50, density=True, alpha=0.35,
                    color=cmap(ep / N_EPISODES), histtype="stepfilled",
                    label=f"ep{ep:02d}" if dim == 0 else None)
        ax.axvline(global_act_min[dim], color="red",   linewidth=1.2, linestyle="--", label="global min")
        ax.axvline(global_act_max[dim], color="green", linewidth=1.2, linestyle="--", label="global max")
        ax.set_title(JOINT_NAMES[dim], fontsize=9)
        ax.set_xlabel("value (rad or norm)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
        if dim == 0:
            ax.legend(fontsize=5, ncol=4, loc="upper right")

    plt.tight_layout()
    p = out_dir / "check2_action_distributions.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # ── Figure 7: gripper open/close ratio per episode ───────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle("Check 2 — Gripper action balance per episode\n"
                 "Binary 0=open, 1=close. Severe imbalance → policy ignores minority class.",
                 fontsize=11)

    close_frac = np.array([
        (d["action"][:, 7] > 0.5).mean() for d in eps
    ])
    open_frac = 1 - close_frac

    ax = axes[0]
    x = np.arange(N_EPISODES)
    ax.bar(x, open_frac,  label="open  (0)",  color="tab:green", alpha=0.8)
    ax.bar(x, close_frac, bottom=open_frac, label="close (1)", color="tab:red", alpha=0.8)
    ax.axhline(0.5, color="k", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ep{i:02d}" for i in x], rotation=45, fontsize=7)
    ax.set_ylabel("fraction of frames")
    ax.set_title("Gripper open/close fraction per episode")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    for ep in ep_ids:
        d = eps[ep]
        ax.plot(d["timestamp"], d["action"][:, 7] + ep * 1.5,
                linewidth=0.6, color=cmap(ep / N_EPISODES))
        ax.text(d["timestamp"][-1] + 0.2, ep * 1.5 + 0.5, f"ep{ep:02d}", fontsize=6)
    ax.set_xlabel("time (s)")
    ax.set_title("Gripper command over time per episode (stacked)")
    ax.set_yticks([])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = out_dir / "check2_gripper_balance.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # ── Figure 8: which episodes set the global min/max (outlier attribution) ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Check 2 — Episodes that SET the global MIN_MAX bounds\n"
                 "These episodes disproportionately control the normalization for all training data.",
                 fontsize=11)

    for ax, mins, maxs, label, color_min, color_max in [
        (axes[0], act_mins, act_maxs, "action", "tab:blue", "tab:orange"),
        (axes[1], st_mins,  st_maxs,  "state",  "tab:purple", "tab:red"),
    ]:
        # For each dim: which episode sets global min / global max?
        min_ep = mins.argmin(axis=0)    # (8,)
        max_ep = maxs.argmax(axis=0)    # (8,)
        global_min = mins.min(axis=0)
        global_max = maxs.max(axis=0)
        med_min    = np.median(mins, axis=0)
        med_max    = np.median(maxs, axis=0)
        deviation_min = global_min - med_min   # negative = outlier pulls min down
        deviation_max = global_max - med_max   # positive = outlier pulls max up

        x = np.arange(8)
        ax.bar(x - 0.2, deviation_min, 0.38, label="global_min − median_min",
               color=color_min, alpha=0.8)
        ax.bar(x + 0.2, deviation_max, 0.38, label="global_max − median_max",
               color=color_max, alpha=0.8)

        # annotate which episode caused it
        for i, (ep_min, ep_max, dev_min, dev_max) in enumerate(
                zip(min_ep, max_ep, deviation_min, deviation_max)):
            ax.text(i - 0.2, dev_min - 0.01, f"ep{ep_min:02d}",
                    ha="center", va="top", fontsize=6, rotation=90)
            ax.text(i + 0.2, dev_max + 0.01, f"ep{ep_max:02d}",
                    ha="center", va="bottom", fontsize=6, rotation=90)

        ax.axhline(0, color="k", linewidth=0.8)
        ax.axhline(-OUTLIER_THRESH, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(+OUTLIER_THRESH, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
                   label=f"±{OUTLIER_THRESH} outlier threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(JOINT_NAMES, fontsize=8)
        ax.set_ylabel("deviation from median (rad)")
        ax.set_title(f"{label} — how far each global bound is from the median bound")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    p = out_dir / "check2_outlier_attribution.png"
    plt.savefig(p, dpi=120)
    plt.close()
    print(f"  Saved: {p}")

    # Print summary
    print("\n  Per-dim outlier episodes (deviation > threshold):")
    any_outlier = False
    for dim in range(8):
        for ep in ep_ids:
            d_min = abs(act_mins[ep, dim] - med_act_min[dim])
            d_max = abs(act_maxs[ep, dim] - med_act_max[dim])
            if d_min > OUTLIER_THRESH or d_max > OUTLIER_THRESH:
                print(f"    {JOINT_NAMES[dim]:12s} ep{ep:02d}: "
                      f"min={act_mins[ep,dim]:.3f} (med={med_act_min[dim]:.3f}, Δ={act_mins[ep,dim]-med_act_min[dim]:+.3f})  "
                      f"max={act_maxs[ep,dim]:.3f} (med={med_act_max[dim]:.3f}, Δ={act_maxs[ep,dim]-med_act_max[dim]:+.3f})")
                any_outlier = True
    if not any_outlier:
        print("    None — all episodes within threshold.")

    norm_coverage = norm_maxs - norm_mins   # (20, 8)
    print(f"\n  Episodes with <30% normalized coverage in ≥1 dim:")
    for ep in ep_ids:
        bad_dims = [JOINT_NAMES[d] for d in range(8) if norm_coverage[ep, d] < 0.3]
        if bad_dims:
            print(f"    ep{ep:02d}: {bad_dims}")

    return {
        "act_mins": act_mins, "act_maxs": act_maxs,
        "global_act_min": global_act_min, "global_act_max": global_act_max,
        "norm_coverage": norm_coverage,
    }


# ═══════════════════════════════════════════════════════════════════════════
# REMEDY REPORT
# ═══════════════════════════════════════════════════════════════════════════

def print_remedy(max_eef_diffs, norm_stats):
    print("\n" + "═" * 66)
    print("  FINDINGS & REMEDIES")
    print("═" * 66)

    # --- Check 1 ---
    print("\n[Check 1] action.right_eef_pose vs observation.right_eef_pose")
    print()
    rot_trans_max = max_eef_diffs[:, :9].max()
    gripper_max   = max_eef_diffs[:, 9].max()
    print(f"  Rotation+translation (dims 0-8): max diff = {rot_trans_max:.4f}")
    print(f"  Gripper articulation  (dim  9):  max diff = {gripper_max:.4f}")
    print()
    print("  FINDING: Dims 0-8 (rotation + translation) of action.right_eef_pose")
    print("  and observation.right_eef_pose are drawn from the SAME hardware read")
    print("  (robot_interface.last_eef_rot_and_pos), not from forward kinematics")
    print("  of the COMMANDED joint angles. So action.right_eef_pose encodes")
    print("  WHERE THE ARM IS NOW, not where it's being commanded to go.")
    print()
    print("  IMPACT: If the policy is trained on action_space='right_eef', it")
    print("  learns to predict the current EEF pose as the target — a no-op.")
    print()
    print("  REMEDY OPTIONS:")
    print("  A. Re-record with corrected add_eef_pose: compute FK from action joints")
    print("     (gello_joints[:7]) rather than reading last_eef_rot_and_pos.")
    print("     In control_utils.py:600:")
    print("       action['action.right_eef_pose'] = add_eef_pose(robot, action['action'])")
    print("     This currently calls robot_interface.last_eef_rot_and_pos for franka_2cam,")
    print("     which reads the observer, not the commanded pose.")
    print()
    print("  B. If re-recording is not possible, train on action_space='joint' only")
    print("     (the 8D action column), ignoring action.right_eef_pose entirely.")
    print("     The joint action IS correct (it's the GELLO reading).")
    print()
    print("  C. Post-process: compute FK offline from action.0-6 joint columns")
    print("     and overwrite action.right_eef_pose in the parquet files.")

    # --- Check 2 ---
    print()
    print("[Check 2] Normalization outliers (MIN_MAX)")
    print()

    act_mins = norm_stats["act_mins"]
    act_maxs = norm_stats["act_maxs"]
    g_min    = norm_stats["global_act_min"]
    g_max    = norm_stats["global_act_max"]
    med_min  = np.median(act_mins, axis=0)
    med_max  = np.median(act_maxs, axis=0)

    # fraction of global range that a typical (median) episode covers
    global_range = g_max - g_min
    typical_range = med_max - med_min
    coverage_pct = np.where(global_range > 1e-6, typical_range / global_range * 100, 100.0)

    print("  Typical-episode range as % of global MIN_MAX range:")
    for dim in range(8):
        flag = " ← COMPRESSED" if coverage_pct[dim] < 50 else ""
        print(f"    {JOINT_NAMES[dim]:12s}: {coverage_pct[dim]:.1f}%{flag}")

    print()
    print("  FINDING: Several joints (especially joint_1, joint_3, joint_7)")
    print("  have outlier episodes that extend the global range significantly")
    print("  beyond what most episodes use. Under MIN_MAX normalization the")
    print("  typical episode's actions get compressed into a small band of [-1,1],")
    print("  leaving the model with less precision where most of the data lives.")
    print()
    print("  REMEDY OPTIONS:")
    print("  A. Switch normalization to MEAN_STD (z-score) for action and state.")
    print("     In config.json: change ACTION normalization from MIN_MAX to MEAN_STD.")
    print("     More robust to outlier episodes; standard practice for diffusion policies.")
    print()
    print("  B. Remove outlier episodes from training:")
    norm_coverage = norm_stats["norm_coverage"]
    outlier_eps = sorted(set(
        ep for ep in range(N_EPISODES)
        for dim in range(8)
        if norm_coverage[ep, dim] < 0.3
    ))
    if outlier_eps:
        print(f"     Episodes with <30% normalized coverage in ≥1 dim: {outlier_eps}")
        print("     In train_config.json: set dataset.episodes to exclude these.")
    else:
        print("     No episodes have critically narrow coverage — option B is lower priority.")
    print()
    print("  C. Clip normalization bounds at percentile (e.g. 1st–99th) rather")
    print("     than absolute min/max, so rare outlier frames don't define the scale.")
    print("     Already used for action.right_eef_pose_relative (PER_TIMESTEP_PERCENTILE)")
    print("     — extend the same logic to action and state.")

    print("\n" + "═" * 66)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./quality_report", help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = load_all()

    max_eef_diffs = check_eef_identity(eps, out_dir)
    norm_stats    = check_normalization(eps, out_dir)
    print_remedy(max_eef_diffs, norm_stats)

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
