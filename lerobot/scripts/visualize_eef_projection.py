#!/usr/bin/env python3
"""
Project end-effector trajectory onto camera images using calibration.

For each frame renders:
  - Filled circle at current EEF position (green=open, red=closed)
  - Fading trail of the last TRAIL_LEN frames (past trajectory)
  - Dotted lookahead of the next LOOK_AHEAD frames (where the arm is going)
  - Gripper state bar and joint angles text overlay

Outputs a side-by-side video of all calibrated cameras.

Usage:
  python visualize_eef_projection.py [--episode N] [--output path.mp4] [--trail 20] [--look-ahead 20]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DATA_ROOT = Path("/home/madhavan/lerobot/data/franka_2cam_zed")
CALIB_ROOT = Path("/home/madhavan/lerobot/lerobot/scripts/franka_2cam_calibration")

CAMERAS = [
    {
        "name": "front",
        "video_key": "observation.images.cam_azure_kinect_front.color",
        "intrinsics": CALIB_ROOT / "cam0_intrinsics.txt",
        "extrinsics": CALIB_ROOT / "cam0_extrinsics.txt",
    },
    {
        "name": "left",
        "video_key": "observation.images.cam_azure_kinect_left.color",
        "intrinsics": CALIB_ROOT / "cam1_intrinsics.txt",
        "extrinsics": CALIB_ROOT / "cam1_extrinsics.txt",
    },
]

# Display resolution per camera panel (scale down from 1280x720)
DISPLAY_W, DISPLAY_H = 640, 360

TRAIL_COLOR     = (0, 200, 255)   # BGR yellow-ish for past trail
LOOKAHEAD_COLOR = (255, 180, 0)   # BGR blue for future
OPEN_COLOR      = (0, 220, 0)     # BGR green for gripper open
CLOSE_COLOR     = (0, 0, 220)     # BGR red for gripper closed
EEF_RADIUS      = 8


def load_calib(cam):
    K = np.loadtxt(cam["intrinsics"])             # 3x3
    world_T_cam = np.loadtxt(cam["extrinsics"])   # 4x4 — files store world_T_cam
    cam_T_world = np.linalg.inv(world_T_cam)      # invert to get projection transform
    return K, cam_T_world


def project(xyz_world, K, cam_T_world, img_w, img_h):
    """Project a 3D world point. Returns (px, py) or None if behind camera / out of frame."""
    pt_h = np.array([xyz_world[0], xyz_world[1], xyz_world[2], 1.0])
    cam_pt = cam_T_world @ pt_h
    if cam_pt[2] <= 0.01:
        return None
    px = K @ cam_pt[:3]
    x = int(px[0] / px[2])
    y = int(px[1] / px[2])
    # scale to display resolution
    x = int(x * DISPLAY_W / img_w)
    y = int(y * DISPLAY_H / img_h)
    if 0 <= x < DISPLAY_W and 0 <= y < DISPLAY_H:
        return (x, y)
    return None


def draw_crosshair(img, pt, color, size=10, thickness=2):
    x, y = pt
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def gripper_color(gripper_val):
    """gripper_val: 0=open, 1=closed. Returns BGR color."""
    return CLOSE_COLOR if gripper_val > 0.5 else OPEN_COLOR


def load_parquet(episode_idx):
    chunk = episode_idx // 1000
    path = DATA_ROOT / f"data/chunk-{chunk:03d}/episode_{episode_idx:06d}.parquet"
    df = pd.read_parquet(path)
    # expand array columns
    for col in df.columns:
        v = df[col].iloc[0]
        if hasattr(v, "__len__") and not isinstance(v, str):
            arr = np.stack(df[col].values)
            for i in range(arr.shape[1]):
                df[f"{col}.{i}"] = arr[:, i]
            df.drop(columns=[col], inplace=True)
    return df


def open_video(episode_idx, video_key):
    chunk = episode_idx // 1000
    path = DATA_ROOT / f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_idx:06d}.mp4"
    cap = cv2.VideoCapture(str(path))
    return cap if cap.isOpened() else None


def read_frame_bgr(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return frame if ret else None


def draw_info_bar(img, frame_idx, timestamp, joints, gripper_obs, gripper_act):
    """Draw a text bar at the bottom of the image."""
    bar_h = 60
    bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)

    j_str = " ".join(f"j{i+1}:{v:.2f}" for i, v in enumerate(joints[:7]))
    cv2.putText(bar, f"f={frame_idx:04d}  t={timestamp:.2f}s  grp_obs={gripper_obs:.2f}  grp_act={gripper_act:.0f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(bar, j_str, (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1)

    return np.vstack([img, bar])


def draw_legend(img):
    x, y = 8, 16
    for label, color in [("current EEF", (0, 220, 0)), ("trail (past)", TRAIL_COLOR),
                          ("look-ahead", LOOKAHEAD_COLOR), ("gripper closed", CLOSE_COLOR)]:
        cv2.circle(img, (x + 6, y), 5, color, -1)
        cv2.putText(img, label, (x + 16, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        y += 18


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: viz_eef_ep<N>.mp4)")
    parser.add_argument("--trail", type=int, default=20, help="Past frames to show as trail")
    parser.add_argument("--look-ahead", type=int, default=20, help="Future frames to show")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    episode_idx = args.episode
    out_path = args.output or f"viz_eef_ep{episode_idx:03d}.mp4"
    TRAIL = args.trail
    AHEAD = args.look_ahead

    print(f"Loading episode {episode_idx}...")
    df = load_parquet(episode_idx)
    N = len(df)

    # Pre-extract EEF translations and gripper columns
    # observation.right_eef_pose: dims 6,7,8 = tx,ty,tz; dim 9 = gripper_artic
    eef_xyz  = df[["observation.right_eef_pose.6",
                    "observation.right_eef_pose.7",
                    "observation.right_eef_pose.8"]].values.astype(float)  # (N,3)
    grp_obs  = df["observation.state.7"].values.astype(float)   # continuous gripper width
    grp_act  = df["action.7"].values.astype(float)              # binary action (0/1)
    joints   = df[[f"observation.state.{i}" for i in range(7)]].values.astype(float)  # (N,7)
    timestamps = df["timestamp"].values

    # Load calibration
    calibs = []
    for cam in CAMERAS:
        K, T = load_calib(cam)
        calibs.append((K, T))

    # Open video captures
    caps = [open_video(episode_idx, cam["video_key"]) for cam in CAMERAS]
    for cam, cap in zip(CAMERAS, caps):
        if cap is None:
            print(f"WARNING: video not found for {cam['name']}")

    # Probe source video dimensions
    src_w, src_h = 1280, 720
    for cap in caps:
        if cap is not None:
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            break

    # Output video: cameras side by side + info bar
    total_w = DISPLAY_W * len(CAMERAS)
    total_h = DISPLAY_H + 60  # 60px info bar
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(out_path, fourcc, args.fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {out_path}")

    print(f"Rendering {N} frames → {out_path}")

    for fi in range(N):
        panels = []

        for cam, cap, (K, T) in zip(CAMERAS, caps, calibs):
            # Grab video frame
            if cap is not None:
                frame = read_frame_bgr(cap, fi)
            else:
                frame = None

            if frame is None:
                panel = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                cv2.putText(panel, f"no video: {cam['name']}", (10, DISPLAY_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
            else:
                panel = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

            # ── past trail ────────────────────────────────────────────────
            start_trail = max(0, fi - TRAIL)
            for ti in range(start_trail, fi):
                alpha = (ti - start_trail + 1) / (fi - start_trail + 1)  # 0→1 as ti→fi
                pt = project(eef_xyz[ti], K, T, src_w, src_h)
                if pt is None:
                    continue
                radius = max(2, int(4 * alpha))
                color = tuple(int(c * alpha) for c in TRAIL_COLOR)
                cv2.circle(panel, pt, radius, color, -1)

            # ── look-ahead (future trajectory) ────────────────────────────
            end_ahead = min(N - 1, fi + AHEAD)
            for ti in range(fi + 1, end_ahead + 1):
                alpha = 1.0 - (ti - fi) / (AHEAD + 1)
                pt = project(eef_xyz[ti], K, T, src_w, src_h)
                if pt is None:
                    continue
                radius = max(2, int(4 * alpha))
                color = tuple(int(c * alpha) for c in LOOKAHEAD_COLOR)
                cv2.circle(panel, pt, radius, color, -1)

            # ── current EEF ───────────────────────────────────────────────
            cur_pt = project(eef_xyz[fi], K, T, src_w, src_h)
            cur_color = gripper_color(grp_act[fi])
            if cur_pt is not None:
                cv2.circle(panel, cur_pt, EEF_RADIUS, cur_color, -1)
                cv2.circle(panel, cur_pt, EEF_RADIUS + 2, (255, 255, 255), 1)  # white ring
                draw_crosshair(panel, cur_pt, (255, 255, 255), size=14, thickness=1)

            # ── gripper state bar (top-right corner) ──────────────────────
            bar_w, bar_h_px = 80, 10
            bx, by = DISPLAY_W - bar_w - 6, 6
            cv2.rectangle(panel, (bx, by), (bx + bar_w, by + bar_h_px), (60, 60, 60), -1)
            filled = int(bar_w * grp_obs[fi])
            cv2.rectangle(panel, (bx, by), (bx + filled, by + bar_h_px), cur_color, -1)
            cv2.rectangle(panel, (bx, by), (bx + bar_w, by + bar_h_px), (200, 200, 200), 1)
            cv2.putText(panel, f"grp", (bx - 30, by + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # ── camera label ──────────────────────────────────────────────
            cv2.putText(panel, cam["name"], (6, DISPLAY_H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            if fi == 0:
                draw_legend(panel)

            panels.append(panel)

        # ── assemble row + info bar ────────────────────────────────────────
        row = np.hstack(panels)
        row = draw_info_bar(row, fi, float(timestamps[fi]),
                            joints[fi], grp_obs[fi], grp_act[fi])
        writer.write(row)

        if fi % 100 == 0:
            print(f"  frame {fi}/{N}")

    writer.release()
    for cap in caps:
        if cap is not None:
            cap.release()

    print(f"Done → {out_path}")


if __name__ == "__main__":
    main()
