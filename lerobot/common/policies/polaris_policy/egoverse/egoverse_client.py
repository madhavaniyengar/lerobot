"""
EgoVersePolicyClient: thin ZMQ client for EgoVerse HPT/flow policies.

Run egoverse_server.py in the EgoVerse environment before starting
control_robot.py in the lerobot environment.

Example:
    python lerobot/scripts/control_robot.py \\
        --robot.type=franka_2cam \\
        --robot.use_eef=true \\
        --control.type=record \\
        --control.policy.type=egoverse \\
        --control.policy.host=localhost \\
        --control.policy.port=5557 \\
        --control.policy.image_resize='[240, 426]' \\
        --control.policy.open_loop_horizon=10 \\
        --control.policy.use_ik=true
"""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from torch import Tensor

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig


DEFAULT_OBS_KEY_MAP = {
    "observation.images.cam_azure_kinect_front.color": "front_img_1",
    "observation.images.cam_azure_kinect_left.color": "front_img_2",
    "observation.right_eef_pose": "state_ee_pose",
}


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


@PreTrainedConfig.register_subclass("egoverse")
@dataclass
class EgoVerseConfig(PreTrainedConfig):
    """Configuration for the EgoVerse remote policy client."""

    host: str = "localhost"
    port: int = 5557
    obs_key_map: dict = field(default_factory=lambda: dict(DEFAULT_OBS_KEY_MAP))
    image_resize: list | None = field(default_factory=lambda: [720, 1280])
    open_loop_horizon: int = 10
    use_ik: bool = True
    transform_eef_to_agent_pos: bool = True
    gripper_latch_enabled: bool = False
    gripper_close_threshold: float = 0.5
    gripper_reopen_threshold: float = 0.8
    enable_goal_conditioning: bool = False

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self):
        return None

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        pass


class EgoVersePolicyClient(PreTrainedPolicy):
    """Inference-only client that forwards observations to egoverse_server.py."""

    config_class = EgoVerseConfig
    name = "egoverse"

    def __init__(self, config: EgoVerseConfig, **kwargs):
        super().__init__(config)
        self.config = config

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{config.host}:{config.port}")
        print(f"[EgoVersePolicyClient] Connected to server at {config.host}:{config.port}")

        if config.use_ik:
            from lerobot.common.policies.robot_adapters import DroidAdapter

            self._droid_adapter = DroidAdapter(action_space="right_eef")
        else:
            self._droid_adapter = None

        self._action_buf: deque[np.ndarray] = deque()
        self._gripper_latched_closed = False

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path=None, *, config=None, **kwargs):
        if config is None:
            from lerobot.configs.policies import PreTrainedConfig

            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy = cls(config)
        policy.eval()
        return policy

    def reset(self):
        self._action_buf.clear()
        self._gripper_latched_closed = False
        self.socket.send(_serialize({"reset": True}))
        response = _deserialize(self.socket.recv())
        if "error" in response:
            raise RuntimeError(f"[EgoVersePolicyClient] Server reset error: {response['error']}")

    def select_action(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if not self._action_buf:
            obs_np = self._batch_to_numpy(batch)
            request = {
                "obs": obs_np,
                "open_loop_horizon": int(self.config.open_loop_horizon),
            }
            self.socket.send(_serialize(request))
            response = _deserialize(self.socket.recv())
            if "error" in response:
                raise RuntimeError(f"[EgoVersePolicyClient] Server error: {response['error']}")

            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"Expected EgoVerse action chunk (T, 7), got {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("[EgoVersePolicyClient] Server returned non-finite actions")
            for action in actions:
                self._action_buf.append(action)

        action_xyzypr = self._action_buf.popleft()
        action_xyzypr = self._apply_gripper_latch(action_xyzypr)
        action_eef_lerobot = self._xyzypr_to_lerobot_eef(action_xyzypr)

        if self._droid_adapter is not None:
            state = batch.get("observation.state", torch.zeros(1, 8)).squeeze(0).cpu()
            action = self._droid_adapter._eef_to_joints(action_eef_lerobot.squeeze(0), state).unsqueeze(0)
        else:
            action = action_eef_lerobot

        return action.float(), action_eef_lerobot.float()

    def _apply_gripper_latch(self, action_xyzypr: np.ndarray) -> np.ndarray:
        if not self.config.gripper_latch_enabled:
            return action_xyzypr

        action = np.asarray(action_xyzypr, dtype=np.float32).copy()
        raw_gripper = float(action[6])

        if self._gripper_latched_closed:
            if raw_gripper > self.config.gripper_reopen_threshold:
                self._gripper_latched_closed = False
            else:
                action[6] = 0.0
                return action

        if raw_gripper < self.config.gripper_close_threshold:
            self._gripper_latched_closed = True
            action[6] = 0.0
        else:
            action[6] = float(np.clip(raw_gripper, 0.0, 1.0))

        return action

    def _batch_to_numpy(self, batch: dict[str, Tensor]) -> dict[str, np.ndarray]:
        obs_np: dict[str, np.ndarray] = {}
        key_map = self.config.obs_key_map or DEFAULT_OBS_KEY_MAP

        for lerobot_key, server_key in key_map.items():
            if lerobot_key not in batch:
                raise KeyError(
                    f"[EgoVersePolicyClient] Missing '{lerobot_key}' in observation. "
                    f"Available keys: {list(batch.keys())}"
                )

            val = batch[lerobot_key]
            if not isinstance(val, Tensor):
                continue

            if "image" in lerobot_key:
                img = val
                if self.config.image_resize is not None:
                    h, w = self.config.image_resize
                    img = F.interpolate(img, size=(h, w), mode="bilinear", align_corners=False)
                obs_np[server_key] = img.squeeze(0).detach().cpu().numpy().astype(np.float32)
            elif lerobot_key.endswith("eef_pose") or server_key == "state_ee_pose":
                obs_np[server_key] = self._lerobot_eef_to_xyzypr(val.squeeze(0)).astype(np.float32)
            else:
                obs_np[server_key] = val.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return obs_np

    @staticmethod
    def _lerobot_eef_to_xyzypr(eef: Tensor) -> np.ndarray:
        """[rot6d(6), xyz(3), gripper(1)] -> [xyz, yaw, pitch, roll, gripper]."""
        import pytorch3d.transforms as p3d

        eef_cpu = eef.detach().cpu().float()
        rot6d = eef_cpu[:6]
        trans = eef_cpu[6:9]
        gripper = eef_cpu[9:10]
        rot_mat = p3d.rotation_6d_to_matrix(rot6d.unsqueeze(0))
        ypr = p3d.matrix_to_euler_angles(rot_mat, "ZYX").squeeze(0)
        return torch.cat([trans, ypr, gripper], dim=0).numpy()

    @staticmethod
    def _xyzypr_to_lerobot_eef(action: np.ndarray) -> Tensor:
        """[xyz, yaw, pitch, roll, gripper] -> (1, 10) [rot6d, xyz, gripper]."""
        import pytorch3d.transforms as p3d

        arr = torch.as_tensor(action, dtype=torch.float32)
        trans = arr[:3]
        ypr = arr[3:6]
        gripper = arr[6:7]
        rot_mat = p3d.euler_angles_to_matrix(ypr.unsqueeze(0), "ZYX")
        rot6d = p3d.matrix_to_rotation_6d(rot_mat).squeeze(0)
        return torch.cat([rot6d, trans, gripper], dim=0).unsqueeze(0)

    def get_optim_params(self) -> dict:
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        raise NotImplementedError("EgoVersePolicyClient is inference-only; use select_action().")
