#!/usr/bin/env python

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
from lerobot.common.datasets.video_utils import decode_video_frames


def _as_array(values, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if shape == (1,):
        arr = np.asarray(values, dtype=dtype)
        return arr.reshape(-1)
    return np.stack(values).astype(dtype, copy=False)


def _create_array(group, key: str, shape: tuple[int, ...], dtype, chunks: tuple[int, ...], overwrite: bool):
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    if key in group:
        if not overwrite:
            raise FileExistsError(f"{key} already exists in {group.store.path}")
        del group[key]
    return group.create_array(
        key,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=compressor,
    )


def _read_rgb_video(video_path: Path, expected_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(np.transpose(frame_rgb, (2, 0, 1)))
    cap.release()

    if len(frames) != expected_frames:
        raise ValueError(f"{video_path} has {len(frames)} frames, expected {expected_frames}")
    return np.stack(frames).astype(np.uint8, copy=False)


def _read_jpeg_video(video_path: Path, expected_frames: int, jpeg_quality: int, max_jpeg_bytes: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    encoded = np.zeros((expected_frames, max_jpeg_bytes), dtype=np.uint8)
    lengths = np.zeros((expected_frames,), dtype=np.int32)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        ok, jpg = cv2.imencode(".jpg", frame_bgr, encode_params)
        if not ok:
            raise ValueError(f"Failed to JPEG encode frame {frame_idx} in {video_path}")
        if len(jpg) > max_jpeg_bytes:
            raise ValueError(
                f"JPEG frame {frame_idx} in {video_path} is {len(jpg)} bytes, "
                f"larger than --max-jpeg-bytes={max_jpeg_bytes}."
            )
        encoded[frame_idx, : len(jpg)] = jpg.reshape(-1)
        lengths[frame_idx] = len(jpg)
        frame_idx += 1
    cap.release()

    if frame_idx != expected_frames:
        raise ValueError(f"{video_path} has {frame_idx} frames, expected {expected_frames}")
    return encoded, lengths


def _read_depth_video(video_path: Path, timestamps: np.ndarray, tolerance_s: float, backend: str):
    frames = decode_video_frames(video_path, timestamps.tolist(), tolerance_s, backend=backend)
    return frames.numpy().astype(np.float32, copy=False)


def convert_lerobot_to_zarr(
    root: Path,
    repo_id: str,
    output: Path,
    camera_keys: list[str] | None,
    include_depth: bool,
    chunk_frames: int,
    tolerance_s: float,
    video_backend: str,
    image_encoding: str,
    jpeg_quality: int,
    max_jpeg_bytes: int,
    episodes: list[int] | None,
    overwrite: bool,
) -> None:
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    output = output if output.is_absolute() else root / output

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)

    group = zarr.open_group(str(output), mode="w", zarr_format=2)
    group.attrs["repo_id"] = repo_id
    group.attrs["root"] = str(root)
    group.attrs["format"] = "lerobot_zarr_v1"
    group.attrs["features_json"] = json.dumps(meta.features)
    group.attrs["image_encoding"] = image_encoding

    all_camera_keys = list(meta.camera_keys)
    if camera_keys is None:
        camera_keys = [key for key in all_camera_keys if include_depth or "depth" not in key]
    missing = sorted(set(camera_keys) - set(all_camera_keys))
    if missing:
        raise ValueError(f"Unknown camera key(s): {missing}. Available: {all_camera_keys}")
    group.attrs["camera_keys"] = camera_keys

    vector_keys = [key for key, ft in meta.features.items() if ft["dtype"] not in ("video", "image")]
    arrays = {}
    for key in vector_keys:
        ft = meta.features[key]
        shape = tuple(ft["shape"])
        zarr_shape = (meta.total_frames,) if shape == (1,) else (meta.total_frames, *shape)
        zarr_chunks = (min(chunk_frames, meta.total_frames),) if shape == (1,) else (min(chunk_frames, meta.total_frames), *shape)
        arrays[key] = _create_array(group, key, zarr_shape, ft["dtype"], zarr_chunks, overwrite=False)

    for key in camera_keys:
        ft = meta.features[key]
        h, w, c = ft["shape"]
        if image_encoding == "jpeg" and "depth" not in key:
            cam_group = group.create_group(key)
            cam_group.attrs["encoding"] = "jpeg"
            arrays[f"{key}/bytes"] = _create_array(
                cam_group,
                "bytes",
                (meta.total_frames, max_jpeg_bytes),
                np.uint8,
                (min(chunk_frames, meta.total_frames), max_jpeg_bytes),
                overwrite=False,
            )
            arrays[f"{key}/lengths"] = _create_array(
                cam_group,
                "lengths",
                (meta.total_frames,),
                np.int32,
                (min(chunk_frames, meta.total_frames),),
                overwrite=False,
            )
        elif "depth" in key:
            dtype = np.float32
            arrays[key] = _create_array(
                group,
                key,
                (meta.total_frames, c, h, w),
                dtype,
                (min(chunk_frames, meta.total_frames), c, h, w),
                overwrite=False,
            )
        else:
            arrays[key] = _create_array(
                group,
                key,
                (meta.total_frames, c, h, w),
                np.uint8,
                (min(chunk_frames, meta.total_frames), c, h, w),
                overwrite=False,
            )

    global_offsets = {}
    running_offset = 0
    for ep_idx in range(meta.total_episodes):
        global_offsets[ep_idx] = running_offset
        running_offset += meta.episodes[ep_idx]["length"]

    episodes_to_convert = list(range(meta.total_episodes)) if episodes is None else episodes
    for ep_idx in tqdm(episodes_to_convert, desc="Converting episodes"):
        parquet_path = root / meta.get_data_file_path(ep_idx)
        df = pd.read_parquet(parquet_path)
        ep_len = len(df)
        offset = global_offsets[ep_idx]
        sl = slice(offset, offset + ep_len)

        for key in vector_keys:
            ft = meta.features[key]
            arrays[key][sl] = _as_array(df[key].to_numpy(), ft["dtype"], tuple(ft["shape"]))

        timestamps = np.asarray(df["timestamp"], dtype=np.float32)
        for key in camera_keys:
            video_path = root / meta.get_video_file_path(ep_idx, key)
            if "depth" in key:
                frames = _read_depth_video(video_path, timestamps, tolerance_s, video_backend)
                arrays[key][sl] = frames
            elif image_encoding == "jpeg":
                encoded, lengths = _read_jpeg_video(video_path, ep_len, jpeg_quality, max_jpeg_bytes)
                arrays[f"{key}/bytes"][sl] = encoded
                arrays[f"{key}/lengths"][sl] = lengths
            else:
                frames = _read_rgb_video(video_path, ep_len)
                arrays[key][sl] = frames

    converted_frames = sum(meta.episodes[ep_idx]["length"] for ep_idx in episodes_to_convert)
    if episodes is None and converted_frames != meta.total_frames:
        raise RuntimeError(f"Converted {offset} frames, but metadata expected {meta.total_frames}")

    print(f"Wrote zarr dataset to {output}")
    print("Camera keys:")
    for key in camera_keys:
        print(f"  - {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a LeRobot parquet/video dataset to zarr.")
    parser.add_argument("--root", type=Path, required=True, help="Path to the LeRobot dataset root.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/zarr_dataset",
        help="Repo id used only for LeRobot metadata loading when --root is local.",
    )
    parser.add_argument("--output", type=Path, default=Path("dataset.zarr"))
    parser.add_argument(
        "--camera-keys",
        nargs="*",
        default=None,
        help="Camera keys to materialize. Defaults to all non-depth cameras.",
    )
    parser.add_argument("--include-depth", action="store_true", help="Also convert depth video keys.")
    parser.add_argument("--chunk-frames", type=int, default=64)
    parser.add_argument("--tolerance-s", type=float, default=4e-4)
    parser.add_argument("--video-backend", type=str, default="pyav")
    parser.add_argument("--image-encoding", choices=["jpeg", "raw"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-jpeg-bytes", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_lerobot_to_zarr(
        root=args.root,
        repo_id=args.repo_id,
        output=args.output,
        camera_keys=args.camera_keys,
        include_depth=args.include_depth,
        chunk_frames=args.chunk_frames,
        tolerance_s=args.tolerance_s,
        video_backend=args.video_backend,
        image_encoding=args.image_encoding,
        jpeg_quality=args.jpeg_quality,
        max_jpeg_bytes=args.max_jpeg_bytes,
        episodes=args.episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
