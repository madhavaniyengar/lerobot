"""
EgoVerse checkpoint inference server.

Run this in the EgoVerse environment before running control_robot.py:

    /home/madhavan/EgoVerse/emimic/bin/python \\
        lerobot/common/policies/polaris_policy/egoverse/egoverse_server.py \\
        --ckpt_path /home/madhavan/EgoVerse/logs/custom_franka_2cam/hpt_bc_flow_franka_2cam_2026-06-15_14-29-24/checkpoints/last.ckpt \\
        --egoverse_root /home/madhavan/EgoVerse \\
        --port 5557 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys
import traceback

import numpy as np
import torch
import zmq


DEFAULT_CKPT_PATH = (
    "/home/madhavan/EgoVerse/logs/custom_franka_2cam/"
    "hpt_bc_flow_franka_2cam_2026-06-15_14-29-24/checkpoints/last.ckpt"
)
DEFAULT_EGOVERSE_ROOT = "/home/madhavan/EgoVerse"
DOMAIN = "franka_right_arm"
EMBODIMENT_ID = 16
ACTION_KEY = "actions_cartesian"
STATE_KEY = "observations.state.ee_pose"
def _enc(v):
    if isinstance(v, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "data": v.tobytes(),
        }
    if isinstance(v, dict):
        return {k2: _enc(v2) for k2, v2 in v.items()}
    return v


def _dec(v):
    if isinstance(v, dict) and v.get("__ndarray__") is True:
        return np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"]).copy()
    if isinstance(v, dict):
        return {k2: _dec(v2) for k2, v2 in v.items()}
    return v


def _serialize(obj) -> bytes:
    return pickle.dumps(_enc(obj))


def _deserialize(data: bytes):
    return _dec(pickle.loads(data))


def _load_calib(calib_dir: str) -> list[dict]:
    """Load calibration from JSON+TXT. Returns list of {'name', 'K', 'T_cam_to_world'}."""
    calib_dir = pathlib.Path(calib_dir)
    json_files = list(calib_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON calibration file in {calib_dir}")
    with open(json_files[0]) as f:
        meta = json.load(f)
    # Paths in the JSON are relative to the parent of the calibration dir (lerobot/scripts/).
    scripts_dir = calib_dir.parent
    cameras = []
    for cam_name, cam_data in meta.items():
        K = np.loadtxt(scripts_dir / cam_data["intrinsics"])
        T = np.loadtxt(scripts_dir / cam_data["extrinsics"])
        cameras.append({"name": cam_name, "K": K, "T_cam_to_world": T})
        print(f"[calib] {cam_name}: f={K[0,0]:.1f}px  pos=[{T[0,3]:.3f},{T[1,3]:.3f},{T[2,3]:.3f}]m")
    return cameras


def _project_xyz(xyz_world: np.ndarray, K: np.ndarray, T_cam_to_world: np.ndarray):
    """Project a robot-base-frame xyz point to image (u,v). Returns None if behind camera."""
    T_world_to_cam = np.linalg.inv(T_cam_to_world)
    xyz_h = np.array([xyz_world[0], xyz_world[1], xyz_world[2], 1.0])
    xyz_cam = T_world_to_cam @ xyz_h
    if xyz_cam[2] <= 0.01:
        return None
    u = K[0, 0] * xyz_cam[0] / xyz_cam[2] + K[0, 2]
    v = K[1, 1] * xyz_cam[1] / xyz_cam[2] + K[1, 2]
    return int(round(float(u))), int(round(float(v)))


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Convert CHW or HWC image (float or uint8) to HWC uint8."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _draw_traj_on_img(
    img_hwc_rgb: np.ndarray,
    xyzs: np.ndarray,
    K: np.ndarray,
    T_cam_to_world: np.ndarray,
    state_xyz: np.ndarray,
) -> np.ndarray:
    """Draw predicted trajectory dots on image (BGR output for cv2 saving)."""
    import cv2
    img_bgr = cv2.cvtColor(img_hwc_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    T = len(xyzs)
    for i, xyz in enumerate(xyzs):
        pt = _project_xyz(xyz, K, T_cam_to_world)
        if pt is None:
            continue
        u, v = pt
        if not (0 <= u < w and 0 <= v < h):
            continue
        ratio = i / max(T - 1, 1)
        # BGR: red→green over the trajectory
        color = (0, int(255 * ratio), int(255 * (1 - ratio)))
        radius = max(2, 8 - i * 6 // T)
        cv2.circle(img_bgr, (u, v), radius, color, -1)
    # White dot = current EEF position
    pt = _project_xyz(state_xyz, K, T_cam_to_world)
    if pt is not None:
        u, v = pt
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_bgr, (u, v), 10, (255, 255, 255), -1)
            cv2.circle(img_bgr, (u, v), 10, (0, 0, 0), 2)
    return img_bgr


def _add_egoverse_to_path(root: str) -> None:
    root_path = pathlib.Path(root).expanduser().resolve()
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))


def load_model(ckpt_path: str, egoverse_root: str, device: str):
    _add_egoverse_to_path(egoverse_root)

    import egomimic.utils.hydra_resolvers  # noqa: F401
    from egomimic.pl_utils.pl_model import ModelWrapper

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = ckpt["hyper_parameters"]
    wrapper = ModelWrapper(
        config_tree=hparams["config_tree"],
        norm_stats_state=hparams["norm_stats_state"],
        scheduler_interval=hparams.get("scheduler_interval", "step"),
        scheduler_frequency=hparams.get("scheduler_frequency", 1),
        enable_grad_norm=hparams.get("enable_grad_norm", False),
    )
    wrapper.load_state_dict(ckpt["state_dict"], strict=True)
    wrapper.eval()
    wrapper.to(device)
    wrapper.model.device = torch.device(device)
    wrapper.model.nets["policy"].device = torch.device(device)
    return wrapper


class EgoVerseRunner:
    def __init__(self, ckpt_path: str, egoverse_root: str, device: str, calib_path: str | None = None):
        self.device = torch.device(device)
        self.wrapper = load_model(ckpt_path, egoverse_root, device)
        self.algo = self.wrapper.model
        self.policy = self.algo.nets["policy"]
        self.image_keys = tuple(self.algo.camera_keys[EMBODIMENT_ID])
        print(
            "[EgoVerseRunner] loaded "
            f"domains={self.algo.domains} camera_keys={self.algo.camera_keys} "
            f"proprio_keys={self.algo.proprio_keys} ac_keys={self.algo.ac_keys}"
        )
        self.cameras = _load_calib(calib_path) if calib_path else None
        self._viz_counter = 0
        if self.cameras:
            print(f"[EgoVerseRunner] visualization enabled — output → /tmp/ego_viz/")

    @torch.no_grad()
    def predict_chunk(self, obs_np: dict) -> np.ndarray:
        raw_batch = self._build_raw_batch(obs_np)
        norm_batch = self.algo.norm_stats.normalize(raw_batch, EMBODIMENT_ID)
        batch = self._add_runtime_keys(norm_batch)
        data = self._build_hpt_data(batch)
        actions = self.policy.forward(DOMAIN, data)
        pred = actions[DOMAIN]
        unnorm = self.algo.norm_stats.unnormalize({ACTION_KEY: pred}, EMBODIMENT_ID)[ACTION_KEY]
        chunk = unnorm.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if chunk.ndim != 2 or chunk.shape[-1] != 7:
            raise ValueError(f"Expected action chunk (T, 7), got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("Model produced non-finite actions")
        state = obs_np["state_ee_pose"]
        image_shapes = ",".join(
            f"{self._short_key(key)}={obs_np[self._short_key(key)].shape}" for key in self.image_keys
        )
        print(
            f"[predict] state=[{state[0]:.3f},{state[1]:.3f},{state[2]:.3f},"
            f"{state[3]:.3f},{state[4]:.3f},{state[5]:.3f},{state[6]:.3f}]  "
            f"act[0]=[{chunk[0,0]:.3f},{chunk[0,1]:.3f},{chunk[0,2]:.3f},"
            f"{chunk[0,3]:.3f},{chunk[0,4]:.3f},{chunk[0,5]:.3f},{chunk[0,6]:.3f}]  "
            f"act[-1]=[{chunk[-1,0]:.3f},{chunk[-1,1]:.3f},{chunk[-1,2]:.3f}]  "
            f"img_shapes={image_shapes}"
        )
        if self.cameras is not None:
            self._save_viz(obs_np, chunk, state)
        return chunk

    def _save_viz(self, obs_np: dict, chunk: np.ndarray, state: np.ndarray) -> None:
        """Project predicted action trajectory onto front camera images and save to /tmp/ego_viz/."""
        try:
            import cv2
        except ImportError:
            print("[viz] cv2 not available, skipping visualization")
            self.cameras = None  # disable future attempts
            return
        out_dir = pathlib.Path("/tmp/ego_viz")
        out_dir.mkdir(exist_ok=True)
        step = self._viz_counter
        self._viz_counter += 1
        state_xyz = state[:3]
        xyzs = chunk[:, :3]  # (T, 3)
        img_obs_keys = ["front_img_1", "front_img_2"]
        saved = []
        for cam_idx, cam in enumerate(self.cameras):
            if cam_idx >= len(img_obs_keys):
                break
            raw = obs_np.get(img_obs_keys[cam_idx])
            if raw is None:
                continue
            img_hwc = _to_hwc_uint8(raw)
            img_bgr = _draw_traj_on_img(img_hwc, xyzs, cam["K"], cam["T_cam_to_world"], state_xyz)
            # text overlay: state + first/last action xyz
            h = img_bgr.shape[0]
            cv2.putText(
                img_bgr,
                f"state xyz=[{state_xyz[0]:.3f},{state_xyz[1]:.3f},{state_xyz[2]:.3f}] g={state[6]:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr,
                f"act[0] xyz=[{xyzs[0,0]:.3f},{xyzs[0,1]:.3f},{xyzs[0,2]:.3f}] g={chunk[0,6]:.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.putText(
                img_bgr,
                f"act[-1] xyz=[{xyzs[-1,0]:.3f},{xyzs[-1,1]:.3f},{xyzs[-1,2]:.3f}] g={chunk[-1,6]:.2f}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 128, 255), 1, cv2.LINE_AA,
            )
            # white = current EEF  |  green→red = predicted trajectory
            cv2.putText(img_bgr, "white=current  green->red=pred", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            out_path = out_dir / f"{cam['name']}_step{step:04d}.jpg"
            cv2.imwrite(str(out_path), img_bgr)
            saved.append(str(out_path))
        if saved:
            print(f"[viz] step={step}  saved: {saved}")

    def _build_raw_batch(self, obs_np: dict) -> dict[str, torch.Tensor]:
        required = tuple(self._short_key(key) for key in self.image_keys) + ("state_ee_pose",)
        missing = [key for key in required if key not in obs_np]
        if missing:
            raise KeyError(f"Missing EgoVerse obs keys: {missing}; got {list(obs_np.keys())}")

        batch = {
            key: self._image_tensor(obs_np[self._short_key(key)])
            for key in self.image_keys
        }
        batch[STATE_KEY] = torch.as_tensor(
            obs_np["state_ee_pose"], dtype=torch.float32, device=self.device
        ).view(1, 7)
        return batch

    @staticmethod
    def _short_key(key: str) -> str:
        return key.rsplit(".", 1)[-1]

    def _image_tensor(self, value: np.ndarray) -> torch.Tensor:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            tensor = torch.from_numpy(arr)
        elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
        else:
            raise ValueError(f"Expected image CHW or HWC, got {arr.shape}")
        if tensor.max() > 2.0:
            tensor = tensor / 255.0
        return tensor.to(self.device).float().unsqueeze(0).contiguous()

    def _add_runtime_keys(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = dict(batch)
        out["pad_mask"] = torch.ones(1, 100, 1, device=self.device)
        out["embodiment"] = torch.tensor([EMBODIMENT_ID], device=self.device, dtype=torch.int64)
        # The diffusion/flow head only needs this for shape/device when building data.
        out[ACTION_KEY] = torch.zeros(1, 100, 7, device=self.device)
        return out

    def _build_hpt_data(self, batch: dict[str, torch.Tensor]) -> dict:
        data = {}
        state = batch[STATE_KEY]
        data["state_ee_pose"] = state.unsqueeze(1)

        for full_key in self.image_keys:
            short = self._short_key(full_key)
            image = batch[full_key]
            if self.algo.eval_image_augs and short in self.algo.encoders:
                image = self.algo.eval_image_augs(image)
            data[short] = image.unsqueeze(1).unsqueeze(1)

        data["is_6dof"] = self.algo.is_6dof
        data["pad_mask"] = batch["pad_mask"]
        data["embodiment"] = batch["embodiment"]
        data["action"] = batch[ACTION_KEY]
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--egoverse_root", type=str, default=DEFAULT_EGOVERSE_ROOT)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--calib_path", type=str, default=None,
        help="Path to calibration dir (e.g. lerobot/scripts/franka_2cam_calibration). "
             "Enables projection of predicted trajectories onto front camera images → /tmp/ego_viz/",
    )
    args = parser.parse_args()

    runner = EgoVerseRunner(args.ckpt_path, args.egoverse_root, args.device, calib_path=args.calib_path)

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"[EgoVerseRunner] server listening on port {args.port}")

    while True:
        raw = socket.recv()
        try:
            request = _deserialize(raw)
            if request.get("reset"):
                socket.send(_serialize({"status": "ok"}))
                continue

            chunk = runner.predict_chunk(request["obs"])
            horizon = int(request.get("open_loop_horizon", 10))
            horizon = max(1, min(horizon, len(chunk)))
            socket.send(_serialize({"actions": chunk[:horizon]}))
        except Exception as exc:
            traceback.print_exc()
            socket.send(_serialize({"error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
