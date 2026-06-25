#!/usr/bin/env python

from pathlib import Path

import numpy as np
import torch
import zarr
import cv2

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
    get_episode_data_index,
)


class ZarrLeRobotDataset(torch.utils.data.Dataset):
    """Zarr-backed LeRobot dataset for fast random access during chunked policy training.

    The converter stores every feature under its original LeRobot key, so this
    class intentionally mirrors ``LeRobotDataset.__getitem__`` as closely as
    possible. RGB camera frames may be stored either as raw channel-first arrays
    or as JPEG bytes plus lengths, and are returned as float tensors in ``[0, 1]``.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        zarr_path: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.delta_indices = None

        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        if self.episodes is not None:
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        self.zarr_path = Path(zarr_path) if zarr_path else self.root / "dataset.zarr"
        if not self.zarr_path.exists():
            raise FileNotFoundError(
                f"Zarr dataset not found at {self.zarr_path}. "
                "Run `python lerobot/scripts/convert_lerobot_to_zarr.py --root ...` first."
            )
        self.store = zarr.open_group(str(self.zarr_path), mode="r")
        self.zarr_camera_keys = list(self.store.attrs.get("camera_keys", []))

        self.selected_episodes = (
            list(range(self.meta.total_episodes)) if self.episodes is None else list(self.episodes)
        )
        self.global_episode_data_index = get_episode_data_index(self.meta.episodes)
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.selected_episodes)
        self.episode_to_local_index = {ep_idx: i for i, ep_idx in enumerate(self.selected_episodes)}
        self.local_to_global_index = self._make_local_to_global_index()

        timestamps = np.asarray(self.store["timestamp"][:])[self.local_to_global_index]
        episode_indices = np.asarray(self.store["episode_index"][:])[self.local_to_global_index]
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def _make_local_to_global_index(self) -> np.ndarray:
        ranges = []
        starts = self.global_episode_data_index["from"]
        ends = self.global_episode_data_index["to"]
        for ep_idx in self.selected_episodes:
            ranges.append(np.arange(starts[ep_idx].item(), ends[ep_idx].item(), dtype=np.int64))
        if len(ranges) == 0:
            return np.asarray([], dtype=np.int64)
        return np.concatenate(ranges)

    @property
    def fps(self) -> int:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.local_to_global_index)

    @property
    def num_episodes(self) -> int:
        return len(self.selected_episodes)

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    def _get_query_indices(self, local_idx: int, ep_idx: int) -> tuple[dict[str, list[int]], dict]:
        local_ep_idx = self.episode_to_local_index[ep_idx]
        ep_start = self.episode_data_index["from"][local_ep_idx]
        ep_end = self.episode_data_index["to"][local_ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, local_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [
                    (local_idx + delta < ep_start.item()) | (local_idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _decode_jpeg(self, encoded: np.ndarray, length: int) -> torch.Tensor:
        frame = cv2.imdecode(encoded[:length], cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG frame from zarr.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1))
        return torch.from_numpy(frame).float().div_(255.0)

    def _read_image(self, key: str, global_idx: int) -> torch.Tensor:
        node = self.store[key]
        if isinstance(node, zarr.Group) and node.attrs.get("encoding") == "jpeg":
            encoded = np.asarray(node["bytes"][global_idx])
            length = int(node["lengths"][global_idx])
            return self._decode_jpeg(encoded, length)

        data = np.asarray(node[global_idx])
        tensor = torch.from_numpy(data)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        else:
            tensor = tensor.float()
        return tensor

    def _read_array(self, key: str, global_indices) -> torch.Tensor:
        if key in self.zarr_camera_keys:
            return torch.stack([self._read_image(key, int(global_idx)) for global_idx in global_indices])

        arr = self.store[key]
        data = arr.get_orthogonal_selection((global_indices,))
        tensor = torch.from_numpy(np.asarray(data))
        return tensor

    def _read_scalar_or_vector(self, key: str, global_idx: int) -> torch.Tensor:
        data = np.asarray(self.store[key][global_idx])
        return torch.as_tensor(data)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        global_idx = int(self.local_to_global_index[idx])
        item = {key: self._read_scalar_or_vector(key, global_idx) for key in self.store.array_keys()}
        for key in self.zarr_camera_keys:
            item[key] = self._read_image(key, global_idx)
        ep_idx = int(item["episode_index"].item())

        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            item.update(padding)
            for key, local_indices in query_indices.items():
                global_indices = self.local_to_global_index[np.asarray(local_indices, dtype=np.int64)]
                item[key] = self._read_array(key, global_indices)

        if self.image_transforms is not None:
            for cam in self.meta.camera_keys:
                if cam in item and "depth" not in cam:
                    item[cam] = self.image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Zarr path: '{self.zarr_path}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )


# ---------------------------------------------------------------------------
# Standalone zarr support (downsampled / self-contained zarr, no parquet files)
# ---------------------------------------------------------------------------

class StandaloneZarrMeta:
    """Metadata stub backed entirely by zarr root attrs.

    Mimics the interface of LeRobotDatasetMetadata that is consumed by
    resolve_delta_timestamps and make_policy (features, fps, stats, tasks).
    Stats and tasks are optionally loaded from the original dataset root so
    that policy normalisation still works.
    """

    def __init__(
        self,
        store: zarr.Group,
        root=None,
        repo_id: str | None = None,
        revision: str | None = None,
        load_original_stats: bool = True,
    ):
        import json as _json

        attrs = dict(store.attrs)
        self.features = _json.loads(attrs["features_json"])
        self._fps = float(attrs["fps"])
        self._total_frames = int(attrs["total_frames"])
        self._total_episodes = int(attrs["total_episodes"])

        ep_idx = attrs["episode_data_index"]
        froms = ep_idx["from"]
        tos = ep_idx["to"]
        self._episodes = [{"length": int(tos[i] - froms[i])} for i in range(len(froms))]
        self._episode_data_index = {
            "from": torch.tensor(froms, dtype=torch.int64),
            "to": torch.tensor(tos, dtype=torch.int64),
        }
        self.camera_keys = list(attrs.get("camera_keys", []))

        # defaults — overwritten below if original metadata is accessible
        self.stats: dict = {}
        self.tasks: dict = {0: ""}

        if load_original_stats and root is not None and repo_id is not None:
            try:
                orig = LeRobotDatasetMetadata(repo_id, Path(root), revision)
                self.stats = orig.stats
                self.tasks = orig.tasks
            except Exception:
                pass

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def episodes(self) -> list[dict]:
        return self._episodes


class StandaloneZarrDataset(torch.utils.data.Dataset):
    """Dataset backed by a self-contained downsampled zarr (no parquet files required).

    Use this instead of ZarrLeRobotDataset when the zarr was created with
    ``create_downsampled_zarr.py`` and therefore has a different frame count
    than the original dataset.

    The zarr root attrs must contain: fps, total_frames, total_episodes,
    episode_data_index ({"from": [...], "to": [...]}), camera_keys, features_json.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        root: str | Path | None = None,
        repo_id: str | None = None,
        episodes: list[int] | None = None,
        image_transforms=None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
    ):
        super().__init__()
        self.zarr_path = Path(zarr_path)
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.delta_indices = None

        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Standalone zarr not found: {self.zarr_path}")

        self.store = zarr.open_group(str(self.zarr_path), mode="r")
        self.zarr_camera_keys = list(self.store.attrs.get("camera_keys", []))

        self.meta = StandaloneZarrMeta(
            self.store, root=root, repo_id=repo_id, revision=revision
        )

        total_ep = self.meta.total_episodes
        self.selected_episodes = list(range(total_ep)) if episodes is None else list(episodes)

        # Build local episode_data_index (0-based into local_to_global_index)
        all_froms = self.meta._episode_data_index["from"]
        all_tos = self.meta._episode_data_index["to"]
        local_froms, local_tos, cursor = [], [], 0
        for ep_idx in self.selected_episodes:
            length = int((all_tos[ep_idx] - all_froms[ep_idx]).item())
            local_froms.append(cursor)
            local_tos.append(cursor + length)
            cursor += length
        self.episode_data_index = {
            "from": torch.tensor(local_froms, dtype=torch.int64),
            "to": torch.tensor(local_tos, dtype=torch.int64),
        }
        self.episode_to_local_index = {ep: i for i, ep in enumerate(self.selected_episodes)}
        self.local_to_global_index = self._make_local_to_global_index()

        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.fps)

    def _make_local_to_global_index(self) -> np.ndarray:
        all_froms = self.meta._episode_data_index["from"]
        all_tos = self.meta._episode_data_index["to"]
        ranges = [
            np.arange(int(all_froms[ep].item()), int(all_tos[ep].item()), dtype=np.int64)
            for ep in self.selected_episodes
        ]
        return np.concatenate(ranges) if ranges else np.asarray([], dtype=np.int64)

    @property
    def fps(self) -> float:
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        return len(self.local_to_global_index)

    @property
    def num_episodes(self) -> int:
        return len(self.selected_episodes)

    @property
    def features(self) -> dict:
        return self.meta.features

    def _decode_jpeg(self, encoded: np.ndarray, length: int) -> torch.Tensor:
        frame = cv2.imdecode(encoded[:length], cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode JPEG frame from zarr.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (2, 0, 1))
        return torch.from_numpy(frame).float().div_(255.0)

    def _read_image(self, key: str, global_idx: int) -> torch.Tensor:
        node = self.store[key]
        if isinstance(node, zarr.Group) and node.attrs.get("encoding") == "jpeg":
            encoded = np.asarray(node["bytes"][global_idx])
            length = int(node["lengths"][global_idx])
            return self._decode_jpeg(encoded, length)
        data = np.asarray(node[global_idx])
        tensor = torch.from_numpy(data)
        return tensor.float().div_(255.0) if tensor.dtype == torch.uint8 else tensor.float()

    def _read_array(self, key: str, global_indices) -> torch.Tensor:
        if key in self.zarr_camera_keys:
            return torch.stack([self._read_image(key, int(i)) for i in global_indices])
        data = self.store[key].get_orthogonal_selection((global_indices,))
        return torch.from_numpy(np.asarray(data))

    def _read_scalar_or_vector(self, key: str, global_idx: int) -> torch.Tensor:
        return torch.as_tensor(np.asarray(self.store[key][global_idx]))

    def _get_query_indices(self, local_idx: int, ep_idx: int):
        local_ep_idx = self.episode_to_local_index[ep_idx]
        ep_start = self.episode_data_index["from"][local_ep_idx].item()
        ep_end = self.episode_data_index["to"][local_ep_idx].item()
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, local_idx + d)) for d in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(local_idx + d < ep_start) or (local_idx + d >= ep_end) for d in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        global_idx = int(self.local_to_global_index[idx])
        item = {key: self._read_scalar_or_vector(key, global_idx) for key in self.store.array_keys()}
        for key in self.zarr_camera_keys:
            item[key] = self._read_image(key, global_idx)

        ep_idx = int(item["episode_index"].item())
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            item.update(padding)
            for key, local_indices in query_indices.items():
                global_indices = self.local_to_global_index[np.asarray(local_indices, dtype=np.int64)]
                item[key] = self._read_array(key, global_indices)

        if self.image_transforms is not None:
            for cam in self.zarr_camera_keys:
                if cam in item and "depth" not in cam:
                    item[cam] = self.image_transforms(item[cam])

        task_idx = int(item["task_index"].item())
        item["task"] = self.meta.tasks.get(task_idx, "")
        return item

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Zarr path: '{self.zarr_path}',\n"
            f"    FPS: {self.fps},\n"
            f"    Episodes: {self.num_episodes},\n"
            f"    Frames: {self.num_frames},\n"
            "}})"
        )
