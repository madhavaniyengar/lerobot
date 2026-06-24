#!/usr/bin/env python3
"""
Visualize and verify the franka_2cam_zed dataset.

Checks:
  - action vs observation.state alignment (temporal offset, gripper continuity)
  - action.right_eef_pose vs observation.right_eef_pose lag
  - video frame timestamps vs parquet timestamps
  - camera images rendered side-by-side with the action/state overlaid
  - end-effector trajectory projected onto each camera view using calibration

Usage:
  python visualize_franka_2cam_dataset.py [--episode N] [--output DIR] [--max-frames N]
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

DATA_ROOT = Path("/home/madhavan/lerobot/data/franka_2cam_zed_shifted")
CALIB_ROOT = Path("/home/madhavan/lerobot/lerobot/scripts/franka_2cam_calibration")

# Camera name -> calibration files
CAMERAS = {
    "cam_azure_kinect_front": {
        "intrinsics": CALIB_ROOT / "cam0_intrinsics.txt",
        "extrinsics": CALIB_ROOT / "cam0_extrinsics.txt",
        "video_key": "observation.images.cam_azure_kinect_front.color",
    },
    "cam_azure_kinect_left": {
        "intrinsics": CALIB_ROOT / "cam1_intrinsics.txt",
        "extrinsics": CALIB_ROOT / "cam1_extrinsics.txt",
        "video_key": "observation.images.cam_azure_kinect_left.color",
    },
    "cam_wrist": {
        "intrinsics": None,
        "extrinsics": None,
        "video_key": "observation.images.cam_wrist",
    },
}


def load_calib(intrinsics_path, extrinsics_path):
    K = np.loadtxt(intrinsics_path)          # 3x3
    T = np.loadtxt(extrinsics_path)          # 4x4  (cam_T_world)
    return K, T


def rot6d_to_matrix(r6d):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1 = r6d[:3]
    a2 = r6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # 3x3


def eef_pose_to_world_point(eef_pose):
    """Extract translation (world coords) from 10-D eef pose."""
    return np.array(eef_pose[6:9])  # trans_0, trans_1, trans_2


def project_point(world_pt, K, cam_T_world):
    """Project a 3D world point to pixel coords using camera calibration."""
    pt_h = np.append(world_pt, 1.0)
    cam_pt = cam_T_world @ pt_h
    if cam_pt[2] <= 0:
        return None
    px = K @ cam_pt[:3]
    px = px[:2] / px[2]
    return px.astype(int)


def load_parquet(episode_idx):
    chunk = episode_idx // 1000
    path = DATA_ROOT / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)

    # Expand array-valued columns into per-element scalar columns
    array_cols = [c for c in df.columns if isinstance(df[c].iloc[0], np.ndarray)]
    for col in array_cols:
        arr = np.stack(df[col].values)
        for i in range(arr.shape[1]):
            df[f"{col}.{i}"] = arr[:, i]
        df.drop(columns=[col], inplace=True)

    return df


def open_video(episode_idx, video_key):
    chunk = episode_idx // 1000
    path = DATA_ROOT / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_idx:06d}.mp4"
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    return cap if cap.isOpened() else None


def read_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ─── Temporal alignment check ────────────────────────────────────────────────

def check_temporal_alignment(df, episode_idx, output_dir):
    """Plot action vs state to detect temporal misalignment."""
    joint_names = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]

    action_cols = [c for c in df.columns if c.startswith("action") and not "eef" in c]
    state_cols  = [c for c in df.columns if c.startswith("observation.state")]

    fig, axes = plt.subplots(8, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(f"Episode {episode_idx} — Action vs State (temporal alignment)", fontsize=14)

    for i, (jname, ax) in enumerate(zip(joint_names, axes)):
        if i < len(action_cols):
            ax.plot(df["timestamp"], df[action_cols[i]], label="action", color="tab:orange", linewidth=1)
        if i < len(state_cols):
            ax.plot(df["timestamp"], df[state_cols[i]], label="obs.state", color="tab:blue", linewidth=1, alpha=0.7)
        ax.set_ylabel(jname, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("timestamp (s)")
    plt.tight_layout()
    out = output_dir / f"ep{episode_idx:03d}_action_vs_state.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")

    # Compute per-joint lag (cross-correlation peak)
    lags = {}
    for i, jname in enumerate(joint_names):
        if i >= len(action_cols) or i >= len(state_cols):
            continue
        a = df[action_cols[i]].values
        s = df[state_cols[i]].values
        a = (a - a.mean()) / (a.std() + 1e-8)
        s = (s - s.mean()) / (s.std() + 1e-8)
        corr = np.correlate(a, s, mode="full")
        lag = int(np.argmax(corr)) - (len(s) - 1)
        lags[jname] = lag

    return lags


def check_eef_lag(df, episode_idx, output_dir):
    """Plot action.right_eef_pose vs observation.right_eef_pose."""
    act_cols = [c for c in df.columns if c.startswith("action.right_eef_pose")]
    obs_cols = [c for c in df.columns if c.startswith("observation.right_eef_pose")]
    if not act_cols or not obs_cols:
        print("  No eef_pose columns found, skipping eef lag check.")
        return {}

    names = ["rot6d_0","rot6d_1","rot6d_2","rot6d_3","rot6d_4","rot6d_5",
             "tx","ty","tz","gripper_artic"]
    n = min(len(act_cols), len(obs_cols), 10)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Episode {episode_idx} — EEF Action vs Observation", fontsize=14)

    lags = {}
    for i, ax in enumerate(axes):
        if i >= n:
            break
        label = names[i] if i < len(names) else f"dim_{i}"
        ax.plot(df["timestamp"], df[act_cols[i]], label="action", color="tab:red", linewidth=1)
        ax.plot(df["timestamp"], df[obs_cols[i]], label="obs", color="tab:green", linewidth=1, alpha=0.7)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

        a = df[act_cols[i]].values
        s = df[obs_cols[i]].values
        a_n = (a - a.mean()) / (a.std() + 1e-8)
        s_n = (s - s.mean()) / (s.std() + 1e-8)
        corr = np.correlate(a_n, s_n, mode="full")
        lag = int(np.argmax(corr)) - (len(s_n) - 1)
        lags[label] = lag

    axes[-1].set_xlabel("timestamp (s)")
    plt.tight_layout()
    out = output_dir / f"ep{episode_idx:03d}_eef_lag.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")
    return lags


# ─── Video timestamp sync check ───────────────────────────────────────────────

def check_video_timestamps(df, episode_idx, output_dir):
    """Compare expected timestamps (from parquet fps) vs actual video frames."""
    fps = 30.0
    n = len(df)
    expected_ts = np.arange(n) / fps
    actual_ts   = df["timestamp"].values
    drift = actual_ts - expected_ts

    _fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(expected_ts, drift * 1000, linewidth=1, color="tab:purple")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("drift (ms) = actual − expected")
    ax.set_title(f"Episode {episode_idx} — Timestamp drift vs uniform {fps}fps")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = output_dir / f"ep{episode_idx:03d}_timestamp_drift.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")

    return {
        "max_drift_ms": float(np.abs(drift).max() * 1000),
        "mean_drift_ms": float(np.abs(drift).mean() * 1000),
        "monotonic": bool(np.all(np.diff(actual_ts) > 0)),
    }


# ─── Frame-level visualisation ────────────────────────────────────────────────

def visualize_frames(df, episode_idx, output_dir, calib, max_frames=8):
    """
    For sampled frames: show all camera views side by side, with EEF position
    projected onto calibrated cameras and action/state bar chart overlaid.
    """
    caps = {}
    for cam_name, info in CAMERAS.items():
        cap = open_video(episode_idx, info["video_key"])
        if cap is None:
            print(f"  WARNING: video not found for {cam_name}")
        caps[cam_name] = cap

    n = len(df)
    indices = np.linspace(0, n - 1, min(max_frames, n), dtype=int)

    act_eef_cols = [c for c in df.columns if c.startswith("action.right_eef_pose")]
    obs_eef_cols = [c for c in df.columns if c.startswith("observation.right_eef_pose")]
    action_cols  = [c for c in df.columns if c.startswith("action") and not "eef" in c]
    state_cols   = [c for c in df.columns if c.startswith("observation.state")]
    joint_names  = [f"j{i}" for i in range(1, 8)] + ["grp"]

    ncams = len(CAMERAS)
    out_frames_dir = output_dir / f"ep{episode_idx:03d}_frames"
    out_frames_dir.mkdir(exist_ok=True)

    for fi in indices:
        row = df.iloc[fi]

        # collect camera images
        imgs = {}
        for cam_name, info in CAMERAS.items():
            cap = caps[cam_name]
            if cap is None:
                imgs[cam_name] = np.zeros((360, 640, 3), dtype=np.uint8)
                continue
            frame = read_frame(cap, int(fi))
            if frame is None:
                imgs[cam_name] = np.zeros((360, 640, 3), dtype=np.uint8)
            else:
                imgs[cam_name] = frame

        # project EEF onto calibrated cameras
        act_eef  = row[act_eef_cols].values  if act_eef_cols  else None
        obs_eef  = row[obs_eef_cols].values  if obs_eef_cols  else None

        for cam_name, (K, T_cam_world) in calib.items():
            if cam_name not in imgs:
                continue
            img = imgs[cam_name].copy()
            h, w = img.shape[:2]

            # draw action eef (red) and obs eef (green)
            for eef_arr, color, label in [
                (obs_eef,  (0, 255, 0),   "obs"),
                (act_eef,  (255, 80, 0),  "act"),
            ]:
                if eef_arr is None or len(eef_arr) < 9:
                    continue
                pt_world = eef_pose_to_world_point(eef_arr)
                px = project_point(pt_world, K, T_cam_world)
                if px is not None and 0 <= px[0] < w and 0 <= px[1] < h:
                    cv2.circle(img, tuple(px), 8, color, -1)
                    cv2.putText(img, label, (px[0] + 10, px[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            imgs[cam_name] = img

        # layout: top row = cameras, bottom = action vs state bar
        fig = plt.figure(figsize=(7 * ncams, 7))
        gs  = gridspec.GridSpec(2, ncams, height_ratios=[3, 1], hspace=0.35)

        for ci, (cam_name, info) in enumerate(CAMERAS.items()):
            ax_img = fig.add_subplot(gs[0, ci])
            ax_img.imshow(imgs[cam_name])
            ax_img.set_title(cam_name.replace("cam_", ""), fontsize=9)
            ax_img.axis("off")

        # action vs state bar chart (bottom spanning full width)
        ax_bar = fig.add_subplot(gs[1, :])
        n_joints = min(len(action_cols), len(state_cols), 8)
        x = np.arange(n_joints)
        w_bar = 0.35
        if action_cols:
            act_vals = [row[action_cols[i]] for i in range(n_joints)]
            ax_bar.bar(x - w_bar / 2, act_vals, w_bar, label="action", color="tab:orange", alpha=0.8)
        if state_cols:
            st_vals = [row[state_cols[i]] for i in range(n_joints)]
            ax_bar.bar(x + w_bar / 2, st_vals, w_bar, label="obs.state", color="tab:blue", alpha=0.8)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(joint_names[:n_joints], fontsize=8)
        ax_bar.legend(fontsize=8)
        ax_bar.set_title(f"frame {fi}  t={row['timestamp']:.3f}s", fontsize=9)
        ax_bar.grid(True, alpha=0.3, axis="y")

        out = out_frames_dir / f"frame_{fi:05d}.png"
        plt.savefig(out, dpi=80, bbox_inches="tight")
        plt.close()

    for cap in caps.values():
        if cap is not None:
            cap.release()

    print(f"  Saved {len(indices)} frame images to {out_frames_dir}/")


# ─── Action continuity / anomaly check ───────────────────────────────────────

def check_action_continuity(df, episode_idx, output_dir):
    """Detect large jumps in action that might indicate dropped frames or bad data."""
    action_cols = [c for c in df.columns if c.startswith("action") and "eef" not in c]
    if not action_cols:
        return {}

    vals = df[action_cols].values.astype(float)
    diffs = np.diff(vals, axis=0)
    abs_diffs = np.abs(diffs)
    max_jump_per_joint = abs_diffs.max(axis=0)
    jump_threshold = 0.5  # radians / gripper width — tune as needed

    suspicious = {}
    for i, col in enumerate(action_cols):
        if max_jump_per_joint[i] > jump_threshold:
            suspicious[col] = float(max_jump_per_joint[i])

    # plot per-joint delta
    fig, axes = plt.subplots(len(action_cols), 1, figsize=(14, 2.2 * len(action_cols)), sharex=True)
    if len(action_cols) == 1:
        axes = [axes]
    fig.suptitle(f"Episode {episode_idx} — Action delta (frame-to-frame jumps)", fontsize=13)
    for i, (col, ax) in enumerate(zip(action_cols, axes)):
        ax.plot(df["timestamp"].values[1:], abs_diffs[:, i], linewidth=0.8, color="tab:red")
        ax.axhline(jump_threshold, color="k", linewidth=0.7, linestyle="--", label=f"thresh={jump_threshold}")
        ax.set_ylabel(col.split(".")[-1], fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    axes[-1].set_xlabel("timestamp (s)")
    plt.tight_layout()
    out = output_dir / f"ep{episode_idx:03d}_action_jumps.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")

    return suspicious


# ─── Action-image lag heatmap ─────────────────────────────────────────────────

def check_action_image_lag(df, episode_idx, output_dir):
    """
    Estimate if the action recorded at frame t corresponds to the image at frame t
    by comparing action vs shifted-state cross-correlation peaks.

    Returns estimated frame lag (positive = action leads state).
    """
    action_cols = [c for c in df.columns if c.startswith("action") and "eef" not in c]
    state_cols  = [c for c in df.columns if c.startswith("observation.state")]
    if not action_cols or not state_cols:
        return None

    lags = []
    for a_col, s_col in zip(action_cols, state_cols):
        a = df[a_col].values.astype(float)
        s = df[s_col].values.astype(float)
        a = (a - a.mean()) / (a.std() + 1e-8)
        s = (s - s.mean()) / (s.std() + 1e-8)
        corr = np.correlate(a, s, mode="full")
        lag = int(np.argmax(corr)) - (len(s) - 1)
        lags.append(lag)

    _fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(lags)), lags, color=["tab:green" if l == 0 else "tab:red" for l in lags])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels([c.split(".")[-1] for c in action_cols], rotation=30, fontsize=8)
    ax.set_ylabel("lag (frames)\n+ve = action leads state")
    ax.set_title(f"Episode {episode_idx} — Action→State cross-correlation lag")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = output_dir / f"ep{episode_idx:03d}_action_image_lag.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")

    return lags


# ─── Summary report ───────────────────────────────────────────────────────────

def print_summary(episode_idx, ts_stats, joint_lags, eef_lags, suspicious_jumps, action_state_lags):
    print("\n" + "═" * 60)
    print(f"  SUMMARY  Episode {episode_idx}")
    print("═" * 60)
    print(f"  Timestamp monotonic     : {ts_stats.get('monotonic')}")
    print(f"  Timestamp max drift     : {ts_stats.get('max_drift_ms', '?'):.1f} ms")
    print(f"  Timestamp mean drift    : {ts_stats.get('mean_drift_ms', '?'):.1f} ms")

    print("\n  Cross-correlation lag (action vs obs.state) — in frames:")
    for k, v in joint_lags.items():
        flag = " ← NON-ZERO" if v != 0 else ""
        print(f"    {k:20s}: {v:+d}{flag}")

    if eef_lags:
        print("\n  EEF action vs obs lag (frames):")
        for k, v in eef_lags.items():
            flag = " ← NON-ZERO" if v != 0 else ""
            print(f"    {k:20s}: {v:+d}{flag}")

    if suspicious_jumps:
        print("\n  LARGE ACTION JUMPS DETECTED:")
        for k, v in suspicious_jumps.items():
            print(f"    {k}: max jump = {v:.4f}")
    else:
        print("\n  No large action jumps detected.")

    if action_state_lags is not None:
        mean_lag = np.mean(action_state_lags)
        print(f"\n  Mean action→state lag   : {mean_lag:+.2f} frames")
        if abs(mean_lag) >= 1:
            print("  *** WARNING: Action and state appear to be misaligned by ≥1 frame ***")
    print("═" * 60 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize and verify franka_2cam_zed dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to inspect")
    parser.add_argument("--output", type=str, default="./viz_output", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=8, help="Number of sampled frames to render")
    parser.add_argument("--all-episodes", action="store_true", help="Run on all episodes (summary only, no frame images)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load calibration for projection
    calib = {}
    for cam_name, info in CAMERAS.items():
        if info["intrinsics"] and info["extrinsics"]:
            K, T = load_calib(info["intrinsics"], info["extrinsics"])
            calib[cam_name] = (K, T)

    # choose episodes
    if args.all_episodes:
        meta = json.loads((DATA_ROOT / "meta/info.json").read_text())
        episodes = list(range(meta["total_episodes"]))
    else:
        episodes = [args.episode]

    for episode_idx in episodes:
        print(f"\n{'─' * 60}")
        print(f"  Processing episode {episode_idx}")
        print(f"{'─' * 60}")

        try:
            df = load_parquet(episode_idx)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Frames: {len(df)}  |  Columns: {len(df.columns)}")

        ts_stats       = check_video_timestamps(df, episode_idx, output_dir)
        joint_lags     = check_temporal_alignment(df, episode_idx, output_dir)
        eef_lags       = check_eef_lag(df, episode_idx, output_dir)
        suspicious     = check_action_continuity(df, episode_idx, output_dir)
        a_s_lags       = check_action_image_lag(df, episode_idx, output_dir)

        if not args.all_episodes:
            visualize_frames(df, episode_idx, output_dir, calib, max_frames=args.max_frames)

        print_summary(episode_idx, ts_stats, joint_lags, eef_lags, suspicious, a_s_lags)

    print(f"All outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
