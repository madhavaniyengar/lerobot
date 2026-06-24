#!/usr/bin/env python3
"""
Reformat franka_2cam_zed dataset from {o_{t+1}, a_t} to {o_t, a_{t+1}}.

Current recording order in teleop_step:
  1. Read GELLO  → a_t
  2. Send a_t to Franka
  3. Read Franka state + cameras → o_{t+1}   (observation AFTER action was sent)
  4. Store (o_{t+1}, a_t) as row t

Desired format:
  Row t: (o_t, a_{t+1})  — observation NOW, action that will come NEXT

Transform:
  new_action[t]              = old_action[t+1]
  new_action.right_eef_pose[t] = old_action.right_eef_pose[t+1]
  all observation columns    = unchanged
  last row per episode       = dropped (no a_{t+1} exists)

Videos are symlinked (not copied) — the parquet only references frames 0..N-2,
so the last video frame is simply never accessed.

Usage:
  python reformat_dataset_shift_action.py [--src DIR] [--dst DIR]
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DEFAULT = Path("data/franka_2cam_zed")
DST_DEFAULT = Path("data/franka_2cam_zed_shifted")

ACTION_COLS = ["action", "action.right_eef_pose"]

VIDEO_KEYS = [
    "observation.images.cam_azure_kinect_front.color",
    "observation.images.cam_azure_kinect_front.transformed_depth",
    "observation.images.cam_azure_kinect_left.color",
    "observation.images.cam_azure_kinect_left.transformed_depth",
    "observation.images.cam_wrist",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_stats(arr: np.ndarray) -> dict:
    """Compute per-dim min/max/mean/std/count from a (N, D) or (N,) array."""
    if arr.ndim == 1:
        arr = arr[:, None]
    return {
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).tolist(),
        "count": [len(arr)],
    }


def compute_image_stats_stub(original_stats: dict) -> dict:
    """Image stats don't change (same frames, just last one dropped — negligible)."""
    return original_stats


def shift_episode(src_parquet: Path, dst_parquet: Path) -> tuple[int, dict]:
    """
    Shift action columns by +1 and drop the last row.
    Returns (new_length, {col: new_stats}).
    """
    df = pd.read_parquet(src_parquet)
    N  = len(df)

    # For array-valued columns we work with object arrays — shift via iloc
    df_new = df.iloc[:-1].copy()   # rows 0 .. N-2  (observations stay)

    for col in ACTION_COLS:
        if col not in df.columns:
            continue
        # rows 1..N-1 of the original become rows 0..N-2 of the new dataframe
        df_new[col] = df[col].iloc[1:].values

    # Rewrite frame_index and index cleanly
    df_new["frame_index"] = np.arange(N - 1, dtype=np.int64)
    # global index will be fixed by the caller after all episodes are processed
    # leave it as-is for now; caller will offset

    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_parquet(dst_parquet, index=False)

    # Compute stats for action columns (changed) and state/eef columns (unchanged)
    stats = {}
    for col in df_new.columns:
        v0 = df_new[col].iloc[0]
        if not hasattr(v0, "__len__") or isinstance(v0, str):
            continue   # skip scalar columns (timestamp, frame_index, etc.)
        arr = np.stack(df_new[col].values).astype(np.float64)
        stats[col] = compute_stats(arr)

    return N - 1, stats


def symlink_videos(src_root: Path, dst_root: Path, n_episodes: int, chunk: int = 0):
    """Create per-episode video symlinks pointing to original files."""
    for vkey in VIDEO_KEYS:
        src_dir = src_root / f"videos/chunk-{chunk:03d}" / vkey
        dst_dir = dst_root / f"videos/chunk-{chunk:03d}" / vkey
        dst_dir.mkdir(parents=True, exist_ok=True)

        for ep in range(n_episodes):
            src_file = src_dir / f"episode_{ep:06d}.mp4"
            dst_file = dst_dir / f"episode_{ep:06d}.mp4"
            if not src_file.exists():
                print(f"  WARNING: video not found: {src_file}")
                continue
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            dst_file.symlink_to(src_file.resolve())

        print(f"  Symlinked {n_episodes} videos: {vkey}")


def update_meta(src_root: Path, dst_root: Path, episode_lengths: list[int],
                episode_stats: list[dict]):
    """Write updated meta files to dst."""
    meta_dst = dst_root / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    meta_src = src_root / "meta"

    # ── info.json ────────────────────────────────────────────────────────
    info = json.loads((meta_src / "info.json").read_text())
    info["total_frames"] = sum(episode_lengths)
    (meta_dst / "info.json").write_text(json.dumps(info, indent=4))
    print(f"  info.json: total_frames = {info['total_frames']}")

    # ── tasks.jsonl ──────────────────────────────────────────────────────
    shutil.copy(meta_src / "tasks.jsonl", meta_dst / "tasks.jsonl")

    # ── episodes.jsonl ───────────────────────────────────────────────────
    src_episodes = [json.loads(l) for l in (meta_src / "episodes.jsonl").read_text().splitlines()]
    with open(meta_dst / "episodes.jsonl", "w") as f:
        for ep, (orig, new_len) in enumerate(zip(src_episodes, episode_lengths)):
            entry = {**orig, "length": new_len}
            f.write(json.dumps(entry) + "\n")
    print(f"  episodes.jsonl: {len(episode_lengths)} episodes written")

    # ── episodes_stats.jsonl ─────────────────────────────────────────────
    src_stats_lines = (meta_src / "episodes_stats.jsonl").read_text().splitlines()
    with open(meta_dst / "episodes_stats.jsonl", "w") as f:
        for ep, (line, new_stats) in enumerate(zip(src_stats_lines, episode_stats)):
            orig = json.loads(line)
            # Merge: replace action columns with recomputed stats, keep image stats
            merged = {**orig["stats"]}
            for col, s in new_stats.items():
                merged[col] = s
            entry = {"episode_index": ep, "stats": merged}
            f.write(json.dumps(entry) + "\n")
    print(f"  episodes_stats.jsonl: {len(episode_stats)} episodes written")


def fix_global_index(dst_root: Path, n_episodes: int, episode_lengths: list[int], chunk: int = 0):
    """Rewrite the global 'index' column so it's contiguous across episodes."""
    print("  Fixing global index column...")
    offset = 0
    for ep in range(n_episodes):
        path = dst_root / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        df = pd.read_parquet(path)
        df["index"] = np.arange(offset, offset + len(df), dtype=np.int64)
        df.to_parquet(path, index=False)
        offset += len(df)
    print(f"  Global index fixed: 0 .. {offset - 1}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(SRC_DEFAULT))
    parser.add_argument("--dst", default=str(DST_DEFAULT))
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if dst.exists():
        print(f"Destination {dst} already exists — removing and recreating.")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Discover episodes
    info = json.loads((src / "meta/info.json").read_text())
    n_episodes = info["total_episodes"]
    chunk      = 0   # single chunk dataset

    print(f"\nSource      : {src}")
    print(f"Destination : {dst}")
    print(f"Episodes    : {n_episodes}")
    print(f"Transform   : {{o_t, a_t}} → {{o_t, a_{{t+1}}}}  (drop last frame per episode)\n")

    # ── shift parquet files ───────────────────────────────────────────────
    episode_lengths = []
    episode_stats   = []

    for ep in range(n_episodes):
        src_parquet = src / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"
        dst_parquet = dst / f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet"

        new_len, stats = shift_episode(src_parquet, dst_parquet)
        episode_lengths.append(new_len)
        episode_stats.append(stats)
        print(f"  ep {ep:02d}: {new_len + 1} → {new_len} frames  (action shifted +1, last row dropped)")

    # ── fix global index ──────────────────────────────────────────────────
    fix_global_index(dst, n_episodes, episode_lengths, chunk)

    # ── symlink videos ────────────────────────────────────────────────────
    print("\nSymlinking videos...")
    symlink_videos(src, dst, n_episodes, chunk)

    # ── update meta ───────────────────────────────────────────────────────
    print("\nUpdating meta files...")
    update_meta(src, dst, episode_lengths, episode_stats)

    # ── verify spot-check ─────────────────────────────────────────────────
    print("\nVerification (episode 0, first 3 rows):")
    src_df = pd.read_parquet(src / f"data/chunk-{chunk:03d}/episode_000000.parquet")
    dst_df = pd.read_parquet(dst / f"data/chunk-{chunk:03d}/episode_000000.parquet")

    src_acts = np.stack(src_df["action"].values)
    dst_acts = np.stack(dst_df["action"].values)

    print(f"  src action[0]: {src_acts[0].round(4)}")
    print(f"  src action[1]: {src_acts[1].round(4)}")
    print(f"  dst action[0]: {dst_acts[0].round(4)}  ← should match src action[1]")
    match = np.allclose(dst_acts[0], src_acts[1])
    print(f"  dst[0] == src[1]: {match}")

    print(f"\n  src action[-2]: {src_acts[-2].round(4)}")
    print(f"  src action[-1]: {src_acts[-1].round(4)}")
    print(f"  dst action[-1]: {dst_acts[-1].round(4)}  ← should match src action[-1]")
    match_last = np.allclose(dst_acts[-1], src_acts[-1])
    print(f"  dst[-1] == src[-1]: {match_last}")

    print(f"\n  src length: {len(src_df)}  dst length: {len(dst_df)}  (dropped {len(src_df)-len(dst_df)} row)")

    total = sum(episode_lengths)
    print(f"\nDone. New dataset at: {dst.resolve()}")
    print(f"  Total frames: {info['total_frames']} → {total}  (−{info['total_frames'] - total} = 1 per episode)")


if __name__ == "__main__":
    main()
