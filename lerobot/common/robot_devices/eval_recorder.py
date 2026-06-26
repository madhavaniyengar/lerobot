"""Record eval rollouts to zarr in the same schema as the training dataset.

Usage inside control_loop (see control_utils.py patch):

    recorder = EvalRecorder("/tmp/eval_recordings")
    # per step:
    recorder.add_step(observation, action, timestamp, episode_index, frame_index, task_index=0)
    # at episode end:
    recorder.save_episode()

Then visualise with the existing script:
    python scripts/visualize_zarr_timestep.py --zarr /tmp/eval_recordings/eval.zarr --episode 0
    python scripts/visualize_zarr_timestep.py --zarr /tmp/eval_recordings/eval.zarr --episode 0 --out /tmp/ep0.mp4
"""

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image


CAMERAS = [
    "observation.images.cam_azure_kinect_left.color",
    "observation.images.cam_azure_kinect_front.color",
    "observation.images.cam_wrist",
]

JPEG_QUALITY = 85
# Match the training dataset's pre-allocated buffer size.
BYTES_BUF = 500_000


def _to_numpy_rgb(img) -> np.ndarray:
    """Accept torch tensor (H,W,3) uint8 or numpy array; return (H,W,3) uint8."""
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img


def _jpeg_encode(img_np: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


class EvalRecorder:
    """Accumulates per-step data and flushes to zarr at episode boundaries."""

    def __init__(self, out_dir: str, zarr_name: str = "eval.zarr"):
        self.out_path = Path(out_dir) / zarr_name
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode_buf: list[dict] = []
        self._all_steps: list[dict] = []
        self._episode_boundaries: list[tuple[int, int]] = []  # (start, end) indices into _all_steps

    # ------------------------------------------------------------------
    def add_step(
        self,
        observation: dict,
        action: dict | None,
        timestamp: float,
        episode_index: int,
        frame_index: int,
        task_index: int = 0,
    ):
        """Call once per control-loop step."""
        step = {
            "timestamp": float(timestamp),
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "task_index": int(task_index),
        }

        # raw camera images (full-res, as captured by the robot)
        for cam_key in CAMERAS:
            if cam_key in observation:
                img_np = _to_numpy_rgb(observation[cam_key])
                step[cam_key] = _jpeg_encode(img_np)
            else:
                step[cam_key] = None

        # pre-normalisation crops — exactly what the model received as input
        debug_keys = [k for k in observation if k.startswith("debug.")]
        step["_debug_keys"] = debug_keys
        for dk in debug_keys:
            img_np = _to_numpy_rgb(observation[dk])
            step[dk] = _jpeg_encode(img_np)

        # state
        state = observation.get("observation.state")
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        step["observation.state"] = np.array(state, dtype=np.float32) if state is not None else None

        # eef obs
        eef_obs = observation.get("observation.right_eef_pose")
        if isinstance(eef_obs, torch.Tensor):
            eef_obs = eef_obs.numpy()
        step["observation.right_eef_pose"] = np.array(eef_obs, dtype=np.float32) if eef_obs is not None else None

        # joint action
        act = None
        if action is not None:
            act = action.get("action")
            if isinstance(act, torch.Tensor):
                act = act.numpy()
        step["action"] = np.array(act, dtype=np.float32) if act is not None else None

        # eef action
        act_eef = None
        if action is not None:
            act_eef = action.get("action.right_eef_pose")
            if isinstance(act_eef, torch.Tensor):
                act_eef = act_eef.numpy()
        step["action.right_eef_pose"] = np.array(act_eef, dtype=np.float32) if act_eef is not None else None

        self._episode_buf.append(step)

    # ------------------------------------------------------------------
    def save_episode(self):
        """Flush the current episode buffer and append to zarr."""
        if not self._episode_buf:
            return
        start = len(self._all_steps)
        self._all_steps.extend(self._episode_buf)
        end = len(self._all_steps)
        self._episode_boundaries.append((start, end))
        self._episode_buf = []
        self._flush_zarr()
        print(f"[EvalRecorder] episode {len(self._episode_boundaries) - 1} saved "
              f"({end - start} frames) → {self.out_path}")

    # ------------------------------------------------------------------
    def _flush_zarr(self):
        steps = self._all_steps
        n = len(steps)
        if n == 0:
            return

        z = zarr.open(str(self.out_path), mode="w")

        def save(name, arr):
            z[name] = arr

        # scalar arrays
        save("index", np.arange(n, dtype=np.int64))
        save("episode_index", np.array([s["episode_index"] for s in steps], dtype=np.int64))
        save("frame_index",   np.array([s["frame_index"]   for s in steps], dtype=np.int64))
        save("task_index",    np.array([s["task_index"]    for s in steps], dtype=np.int64))
        save("timestamp",     np.array([s["timestamp"]     for s in steps], dtype=np.float32))

        # state / action
        for key in ("observation.state", "action", "observation.right_eef_pose", "action.right_eef_pose"):
            vals = [s[key] for s in steps if s[key] is not None]
            if vals:
                arr = np.stack(vals).astype(np.float32)
                if len(arr) < n:
                    pad = np.zeros((n - len(arr), arr.shape[1]), dtype=np.float32)
                    arr = np.concatenate([arr, pad], axis=0)
                save(key, arr)

        # raw camera images
        for cam_key in CAMERAS:
            jpeg_list = [s[cam_key] for s in steps]
            lengths = np.array([len(b) if b is not None else 0 for b in jpeg_list], dtype=np.int32)
            byte_arr = np.zeros((n, BYTES_BUF), dtype=np.uint8)
            for i, b in enumerate(jpeg_list):
                if b is not None:
                    byte_arr[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
            z[f"{cam_key}/lengths"] = lengths
            z[f"{cam_key}/bytes"] = byte_arr

        # pre-normalisation crops (what the model actually saw)
        all_debug_keys = set()
        for s in steps:
            all_debug_keys.update(s.get("_debug_keys", []))
        for dk in sorted(all_debug_keys):
            jpeg_list = [s.get(dk) for s in steps]
            lengths = np.array([len(b) if b is not None else 0 for b in jpeg_list], dtype=np.int32)
            byte_arr = np.zeros((n, BYTES_BUF), dtype=np.uint8)
            for i, b in enumerate(jpeg_list):
                if b is not None:
                    byte_arr[i, :len(b)] = np.frombuffer(b, dtype=np.uint8)
            z[f"{dk}/lengths"] = lengths
            z[f"{dk}/bytes"] = byte_arr
