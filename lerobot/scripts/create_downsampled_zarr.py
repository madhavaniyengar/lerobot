#!/usr/bin/env python
"""Convert a LeRobot dataset to a self-contained, downsampled zarr archive.

Applies two reductions:
  - Temporal: keeps every ``--temporal-stride``-th frame (default 2 → half frequency).
  - Spatial:  resizes all RGB images to (H // ``--scale-factor``, W // ``--scale-factor``)
              (default 2 → half linear resolution).
  - Optional crop: saves resized images as ``--crop-shape H W`` crops. By default
                   crops are centered; ``--wrist-crop left`` left-aligns the
                   wrist camera crop horizontally.

The output zarr is fully self-contained: it embeds episode boundaries, fps,
and feature metadata in its root attributes so it can be loaded without the
original LeRobot metadata files.

Loading the result is as simple as::

    import zarr, numpy as np
    store = zarr.open_group("dataset_downsampled.zarr", mode="r")
    # store.attrs contains: fps, total_frames, total_episodes,
    #   episode_data_index ({"from": [...], "to": [...]}),
    #   camera_keys, features_json
    frame = store["observation.state"][42]          # any vector feature
    jpg   = store["observation.images.cam_front/bytes"][42, :store["observation.images.cam_front/lengths"][42]]
    img   = cv2.imdecode(jpg, cv2.IMREAD_COLOR)     # BGR uint8, half resolution
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import zarr
from numcodecs import Blosc
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def _compressor():
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def _create_array(group, key: str, shape, dtype, chunks):
    return group.create_array(key, shape=shape, dtype=dtype, chunks=chunks, compressor=_compressor())


def _as_array(values, dtype: str, shape: tuple) -> np.ndarray:
    if shape == (1,):
        return np.asarray(values, dtype=dtype).reshape(-1)
    return np.stack(values).astype(dtype, copy=False)


def _crop_frame(
    frame: np.ndarray,
    crop_h: int,
    crop_w: int,
    mode: str = "center",
    left_override: int | None = None,
) -> np.ndarray:
    """Crop an HWC frame. ``left_override`` is an explicit x pixel in resized image space."""
    h, w = frame.shape[:2]
    if crop_h > h or crop_w > w:
        raise ValueError(f"Crop {crop_h}x{crop_w} does not fit inside frame {h}x{w}.")

    top = (h - crop_h) // 2
    if left_override is not None:
        left = int(left_override)
    elif mode == "left":
        left = 0
    elif mode == "center":
        left = (w - crop_w) // 2
    elif mode == "right":
        left = w - crop_w
    else:
        raise ValueError(f"Unknown crop mode: {mode}")

    left = max(0, min(left, w - crop_w))
    return frame[top : top + crop_h, left : left + crop_w]


def _crop_mode_for_key(camera_key: str, wrist_crop: str) -> str:
    return wrist_crop if "cam_wrist" in camera_key else "center"


def _crop_left_override_for_key(camera_key: str, wrist_crop_left: int | None) -> int | None:
    return wrist_crop_left if "cam_wrist" in camera_key else None


# ---------------------------------------------------------------------------
# Frame readers with optional downsampling
# ---------------------------------------------------------------------------

def _read_jpeg_video_strided(
    video_path: Path,
    keep_indices: np.ndarray,
    out_h: int,
    out_w: int,
    crop_shape: tuple[int, int] | None,
    crop_mode: str,
    crop_left_override: int | None,
    jpeg_quality: int,
    max_jpeg_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read selected frames from a video, resize, and JPEG-encode."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    n_out = len(keep_indices)
    encoded = np.zeros((n_out, max_jpeg_bytes), dtype=np.uint8)
    lengths = np.zeros((n_out,), dtype=np.int32)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    keep_set = set(keep_indices.tolist())
    out_idx = 0
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in keep_set:
            resized = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if crop_shape is not None:
                resized = _crop_frame(
                    resized, crop_shape[0], crop_shape[1], crop_mode, crop_left_override
                )
            ok2, jpg = cv2.imencode(".jpg", resized, encode_params)
            if not ok2:
                raise ValueError(f"JPEG encode failed for frame {frame_idx} in {video_path}")
            if len(jpg) > max_jpeg_bytes:
                raise ValueError(
                    f"Frame {frame_idx} in {video_path} is {len(jpg)} bytes > --max-jpeg-bytes={max_jpeg_bytes}"
                )
            encoded[out_idx, : len(jpg)] = jpg.reshape(-1)
            lengths[out_idx] = len(jpg)
            out_idx += 1
        frame_idx += 1
    cap.release()

    if out_idx != n_out:
        raise ValueError(f"{video_path}: expected {n_out} kept frames but got {out_idx}")
    return encoded, lengths


def _read_depth_video_strided(
    video_path: Path,
    keep_indices: np.ndarray,
    out_h: int,
    out_w: int,
    crop_shape: tuple[int, int] | None,
    crop_mode: str,
    crop_left_override: int | None,
) -> np.ndarray:
    """Read selected depth frames and resize."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    keep_set = set(keep_indices.tolist())
    frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in keep_set:
            resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if crop_shape is not None:
                resized = _crop_frame(
                    resized, crop_shape[0], crop_shape[1], crop_mode, crop_left_override
                )
            frames.append(np.transpose(resized, (2, 0, 1)))
        frame_idx += 1
    cap.release()
    return np.stack(frames).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def create_downsampled_zarr(
    root: Path,
    repo_id: str,
    output: Path,
    camera_keys: list[str] | None,
    include_depth: bool,
    temporal_stride: int,
    scale_factor: int,
    crop_shape: tuple[int, int] | None,
    wrist_crop: str,
    wrist_crop_left: int | None,
    viz_output: Path | None,
    chunk_frames: int,
    jpeg_quality: int,
    max_jpeg_bytes: int,
    episodes: list[int] | None,
    overwrite: bool,
) -> None:
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    # ---- resolve camera keys ----
    all_camera_keys = list(meta.camera_keys)
    if camera_keys is None:
        camera_keys = [k for k in all_camera_keys if include_depth or "depth" not in k]
    missing = sorted(set(camera_keys) - set(all_camera_keys))
    if missing:
        raise ValueError(f"Unknown camera key(s): {missing}. Available: {all_camera_keys}")

    # ---- compute output image dimensions ----
    # Use the first camera key to get canonical image shape (H, W, C)
    first_cam = camera_keys[0] if camera_keys else None
    if first_cam:
        orig_h, orig_w, _c = meta.features[first_cam]["shape"]
        resized_h, resized_w = orig_h // scale_factor, orig_w // scale_factor
        if crop_shape is not None:
            crop_h, crop_w = crop_shape
            if crop_h > resized_h or crop_w > resized_w:
                raise ValueError(
                    f"--crop-shape {crop_h} {crop_w} does not fit resized image "
                    f"{resized_h}x{resized_w}. Reduce crop or increase output size."
                )
            out_h, out_w = crop_h, crop_w
        else:
            out_h, out_w = resized_h, resized_w
    else:
        resized_h = resized_w = out_h = out_w = None

    episodes_to_convert = list(range(meta.total_episodes)) if episodes is None else episodes

    # ---- first pass: compute per-episode kept-frame counts ----
    ep_kept: dict[int, int] = {}
    for ep_idx in episodes_to_convert:
        ep_len = meta.episodes[ep_idx]["length"]
        ep_kept[ep_idx] = len(range(0, ep_len, temporal_stride))

    total_frames = sum(ep_kept.values())
    fps_out = meta.fps / temporal_stride

    # ---- build episode_data_index for the output zarr ----
    ep_from, ep_to = [], []
    cursor = 0
    for ep_idx in episodes_to_convert:
        ep_from.append(cursor)
        ep_to.append(cursor + ep_kept[ep_idx])
        cursor += ep_kept[ep_idx]

    # ---- pre-allocate zarr arrays ----
    group = zarr.open_group(str(output), mode="w", zarr_format=2)

    # update features_json to reflect new image shapes
    updated_features = dict(meta.features)
    if out_h is not None:
        for ck in camera_keys:
            ft = dict(updated_features[ck])
            _orig_h, _orig_w, c = ft["shape"]
            ft["shape"] = (out_h, out_w, c)
            updated_features[ck] = ft

    group.attrs["repo_id"] = repo_id
    group.attrs["root"] = str(root)
    group.attrs["format"] = "lerobot_zarr_downsampled_v1"
    group.attrs["features_json"] = json.dumps(updated_features)
    group.attrs["image_encoding"] = "jpeg"
    group.attrs["camera_keys"] = camera_keys
    group.attrs["fps"] = fps_out
    group.attrs["total_frames"] = total_frames
    group.attrs["total_episodes"] = len(episodes_to_convert)
    group.attrs["episode_data_index"] = {"from": ep_from, "to": ep_to}
    group.attrs["temporal_stride"] = temporal_stride
    group.attrs["scale_factor"] = scale_factor
    group.attrs["crop_shape"] = list(crop_shape) if crop_shape is not None else None
    group.attrs["wrist_crop"] = wrist_crop
    group.attrs["wrist_crop_left"] = wrist_crop_left
    group.attrs["original_fps"] = meta.fps

    vector_keys = [k for k, ft in meta.features.items() if ft["dtype"] not in ("video", "image")]
    arrays: dict[str, zarr.Array] = {}

    chk = min(chunk_frames, total_frames)
    for key in vector_keys:
        ft = meta.features[key]
        shape = tuple(ft["shape"])
        zarr_shape = (total_frames,) if shape == (1,) else (total_frames, *shape)
        zarr_chunks = (chk,) if shape == (1,) else (chk, *shape)
        arrays[key] = _create_array(group, key, zarr_shape, ft["dtype"], zarr_chunks)

    for ck in camera_keys:
        if "depth" in ck:
            _orig_h2, _orig_w2, dc = meta.features[ck]["shape"]
            _rh, _rw = _orig_h2 // scale_factor, _orig_w2 // scale_factor
            _oh, _ow = crop_shape if crop_shape is not None else (_rh, _rw)
            arrays[ck] = _create_array(
                group, ck, (total_frames, dc, _oh, _ow), np.float32, (chk, dc, _oh, _ow)
            )
        else:
            cam_group = group.create_group(ck)
            cam_group.attrs["encoding"] = "jpeg"
            arrays[f"{ck}/bytes"] = _create_array(
                cam_group, "bytes", (total_frames, max_jpeg_bytes), np.uint8, (chk, max_jpeg_bytes)
            )
            arrays[f"{ck}/lengths"] = _create_array(
                cam_group, "lengths", (total_frames,), np.int32, (chk,)
            )

    # ---- second pass: fill arrays ----
    write_offset = 0
    for ep_idx in tqdm(episodes_to_convert, desc="Converting episodes"):
        parquet_path = root / meta.get_data_file_path(ep_idx)
        df = pd.read_parquet(parquet_path)
        ep_len = len(df)
        keep_indices = np.arange(0, ep_len, temporal_stride)
        n_kept = len(keep_indices)
        sl = slice(write_offset, write_offset + n_kept)

        # downsample vector features
        df_kept = df.iloc[keep_indices].reset_index(drop=True)
        # rewrite frame_index to be contiguous 0..n_kept-1
        df_kept["frame_index"] = np.arange(n_kept, dtype=np.int64)

        for key in vector_keys:
            ft = meta.features[key]
            arrays[key][sl] = _as_array(df_kept[key].to_numpy(), ft["dtype"], tuple(ft["shape"]))

        # downsample camera frames
        for ck in camera_keys:
            video_path = root / meta.get_video_file_path(ep_idx, ck)
            cam_h, cam_w = out_h, out_w
            if "depth" in ck:
                _oh2, _ow2, _ = meta.features[ck]["shape"]
                cam_h, cam_w = _oh2 // scale_factor, _ow2 // scale_factor
                frames = _read_depth_video_strided(
                    video_path,
                    keep_indices,
                    cam_h,
                    cam_w,
                    crop_shape,
                    _crop_mode_for_key(ck, wrist_crop),
                    _crop_left_override_for_key(ck, wrist_crop_left),
                )
                arrays[ck][sl] = frames
            else:
                encoded, lengths = _read_jpeg_video_strided(
                    video_path,
                    keep_indices,
                    meta.features[ck]["shape"][0] // scale_factor,
                    meta.features[ck]["shape"][1] // scale_factor,
                    crop_shape,
                    _crop_mode_for_key(ck, wrist_crop),
                    _crop_left_override_for_key(ck, wrist_crop_left),
                    jpeg_quality,
                    max_jpeg_bytes,
                )
                arrays[f"{ck}/bytes"][sl] = encoded
                arrays[f"{ck}/lengths"][sl] = lengths

        write_offset += n_kept

    print(f"\nWrote downsampled zarr to {output}")
    print(f"  Original fps: {meta.fps}  →  Output fps: {fps_out}")
    print(
        f"  Original size: {orig_h}x{orig_w}  →  Resized: {resized_h}x{resized_w}"
        f"  →  Saved: {out_h}x{out_w}" if out_h else ""
    )
    print(f"  Total frames: {total_frames}  (out of {meta.total_frames} original)")
    print(f"  Camera keys: {camera_keys}")

    if viz_output is not None:
        _write_crop_viz(output, camera_keys, viz_output)
        print(f"  Wrote viz: {viz_output}")


def _read_zarr_jpeg_rgb(group: zarr.Group, key: str, frame_idx: int = 0) -> np.ndarray:
    node = group[key]
    encoded = np.asarray(node["bytes"][frame_idx])
    length = int(node["lengths"][frame_idx])
    bgr = cv2.imdecode(encoded[:length], cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to decode JPEG frame {frame_idx} for {key}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_crop_viz(zarr_path: Path, camera_keys: list[str], viz_output: Path) -> None:
    """Write a side-by-side PNG of the first saved RGB frame for quick inspection."""
    from PIL import Image, ImageDraw

    group = zarr.open_group(str(zarr_path), mode="r")
    rgb_keys = [k for k in camera_keys if "depth" not in k]
    crops = []
    labels = []
    for key in rgb_keys:
        crops.append(Image.fromarray(_read_zarr_jpeg_rgb(group, key, frame_idx=0)))
        labels.append(key.replace("observation.images.", ""))

    if not crops:
        return

    w, h = crops[0].size
    label_h = 28
    canvas = Image.new("RGB", (w * len(crops), h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (crop, label) in enumerate(zip(crops, labels)):
        x = i * w
        canvas.paste(crop, (x, label_h))
        draw.text((x + 8, 7), label, fill=(0, 0, 0))
    viz_output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(viz_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a temporally and spatially downsampled zarr from a LeRobot dataset."
    )
    parser.add_argument("--root", type=Path, required=True, help="LeRobot dataset root directory.")
    parser.add_argument("--repo-id", type=str, default="local/dataset",
                        help="Repo-id string (used only for metadata loading).")
    parser.add_argument("--output", type=Path, default=Path("dataset_downsampled.zarr"),
                        help="Output zarr path.")
    parser.add_argument("--temporal-stride", type=int, default=2,
                        help="Keep every Nth frame (2 = half frequency).")
    parser.add_argument("--scale-factor", type=int, default=2,
                        help="Divide image H and W by this factor (2 = half resolution).")
    parser.add_argument("--crop-shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="After resizing, save an HxW crop instead of the full resized image.")
    parser.add_argument("--wrist-crop", choices=["left", "center", "right"], default="left",
                        help="Horizontal crop mode for cam_wrist when --crop-shape is set.")
    parser.add_argument("--wrist-crop-left", type=int, default=None,
                        help="Explicit left x pixel for cam_wrist crop after resizing. Overrides --wrist-crop.")
    parser.add_argument("--viz-output", type=Path, default=None,
                        help="Optional PNG path for a side-by-side visualization of the first saved RGB crops.")
    parser.add_argument("--camera-keys", nargs="*", default=None,
                        help="Camera keys to materialise. Defaults to all non-depth cameras.")
    parser.add_argument("--include-depth", action="store_true",
                        help="Also convert depth video keys.")
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-jpeg-bytes", type=int, default=500_000,
                        help="Max bytes per JPEG frame. Can be smaller than the original "
                             "since images are downsampled.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="Subset of episode indices to convert.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    create_downsampled_zarr(
        root=args.root,
        repo_id=args.repo_id,
        output=args.output,
        camera_keys=args.camera_keys,
        include_depth=args.include_depth,
        temporal_stride=args.temporal_stride,
        scale_factor=args.scale_factor,
        crop_shape=tuple(args.crop_shape) if args.crop_shape is not None else None,
        wrist_crop=args.wrist_crop,
        wrist_crop_left=args.wrist_crop_left,
        viz_output=args.viz_output,
        chunk_frames=args.chunk_frames,
        jpeg_quality=args.jpeg_quality,
        max_jpeg_bytes=args.max_jpeg_bytes,
        episodes=args.episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
