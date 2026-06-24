"""Franka2CamRobot: Franka Panda + stock hand + GELLO + two Azure Kinects."""

import glob
import os
import time

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import Franka2CamRobotConfig


class Franka2CamRobot:
    """Robot class for a Franka Panda controlled via deoxys with GELLO teleoperation.

    The follower arm is a Franka Panda controlled through the deoxys C++ controller
    interface over ZMQ. The leader arm is a GELLO (7-DOF Dynamixel) accessed through
    the gello Python package.
    """

    robot_type = "franka_2cam"

    def __init__(self, config: Franka2CamRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = Franka2CamRobotConfig(**kwargs)
        else:
            self.config = config

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.robot_interface = None
        self.gello = None
        self.gello_home = None
        self.robot_home = None
        self.is_connected = False
        self.logs = {}
        self.robot_type = self.config.type
        self.use_eef = self.config.use_eef
        self.joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
            "gripper",
        ]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            features = cam.config.get_feature_specs(cam_key)

            # Datasets recorded with this robot historically expose a color-only
            # ZED stream as ``observation.images.<camera>``.  ZedCameraConfig now
            # describes that stream as ``...<camera>.color``; retain the old key
            # here so prerecorded datasets and their policy normalization buffers
            # remain compatible at rollout time.
            if cam.config.__class__.__name__ == "ZedCameraConfig" and not cam.config.use_depth:
                base_key = f"observation.images.{cam_key}"
                color_key = f"{base_key}.color"
                if color_key in features:
                    features = dict(features)
                    features[base_key] = features.pop(color_key)

            cam_ft.update(features)
        return cam_ft

    def _camera_observation_key(self, cam_name: str, stream_name: str) -> str:
        """Return the dataset-compatible observation key for a camera stream."""
        cam = self.cameras[cam_name]
        if (
            cam.config.__class__.__name__ == "ZedCameraConfig"
            and not cam.config.use_depth
            and stream_name == "color"
        ):
            return f"observation.images.{cam_name}"
        return f"observation.images.{cam_name}.{stream_name}"

    @property
    def motor_features(self) -> dict:
        motor_features = {
            "action": {
                "dtype": "float32",
                "shape": (len(self.joint_names),),
                "names": list(self.joint_names),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.joint_names),),
                "names": list(self.joint_names),
            },
        }

        if self.use_eef:
            eef_names = ["rot_6d_0", "rot_6d_1", "rot_6d_2", "rot_6d_3", "rot_6d_4", "rot_6d_5", "trans_0", "trans_1", "trans_2", "gripper_articulation"]
            motor_features["observation.right_eef_pose"] = {
                "dtype": "float32",
                "shape": (len(eef_names),),
                "names": eef_names,
            }
            motor_features["action.right_eef_pose"] = {
                "dtype": "float32",
                "shape": (len(eef_names),),
                "names": eef_names,
            }

        return motor_features

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    def connect(self):
        if self.is_connected:
            raise RuntimeError("Franka2CamRobot is already connected. Do not run `robot.connect()` twice.")

        # Lazy import deoxys
        from deoxys.franka_interface import FrankaInterface
        from deoxys.utils import YamlConfig

        # Initialize Franka interface
        self.robot_interface = FrankaInterface(
            self.config.deoxys_general_cfg_file,
            use_visualizer=False,
        )
        self.controller_cfg = YamlConfig(
            self.config.deoxys_controller_cfg_file
        ).as_easydict()

        # Wait for state buffer to populate
        print("Waiting for Franka state buffer...")
        timeout = 30.0
        start_t = time.time()
        while len(self.robot_interface._state_buffer) == 0:
            time.sleep(0.1)
            if time.time() - start_t > timeout:
                raise TimeoutError(
                    "Timed out waiting for Franka state buffer. "
                    "Check that the deoxys controller is running."
                )
        print("Franka state buffer ready.")

        # Initialize GELLO
        self._init_gello()

        # The stock Franka hand is controlled directly through deoxys and needs
        # no extra Python gripper client.

        # Connect cameras (two-phase init for multi-Azure Kinect setups)
        from threading import Thread

        azure_kinect_cameras = []
        for name, camera in self.cameras.items():
            if camera.__class__.__name__ == "AzureKinectCamera":
                camera.connect(start_cameras=False)
                azure_kinect_cameras.append(camera)
            else:
                camera.connect()

        if len(azure_kinect_cameras) > 0:
            camera_start_errors = []

            def start_camera(cam):
                try:
                    cam.start()
                except Exception as exc:
                    camera_start_errors.append((cam, exc))

            threads = [Thread(target=start_camera, args=(cam,)) for cam in azure_kinect_cameras]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if camera_start_errors:
                errors = ", ".join(
                    f"AzureKinectCamera({cam.device_id}): {exc}" for cam, exc in camera_start_errors
                )
                raise RuntimeError(f"Failed to start Azure Kinect camera(s): {errors}")

        self.is_connected = True

        # Run interactive home calibration (skip during policy inference)
        if not self.config.skip_gello_calibration:
            self.run_calibration()

    def _init_gello(self):
        """Initialize the GELLO leader arm."""
        from gello.robots.dynamixel import DynamixelRobot

        port = self.config.gello_port
        if port is None:
            # GELLO uses a generic FTDI adapter whose by-id name contains
            # "Serial_Converter". Match on that to disambiguate.
            matches = [
                p for p in glob.glob("/dev/serial/by-id/*")
                if "Serial_Converter" in p
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one GELLO serial device, found {matches}. "
                    "Pass --robot.gello_port=... to override."
                )
            port = matches[0]
            print(f"Auto-detected GELLO on {port}")

        self._gello_port = port
        self._check_gello_port_access(port)

        self.gello = DynamixelRobot(
            joint_ids=list(self.config.gello_joint_ids),
            joint_offsets=list(self.config.gello_joint_offsets),
            real=True,
            joint_signs=list(self.config.gello_joint_signs),
            port=port,
            gripper_config=(
                self.config.gello_gripper_joint_id,
                self.config.gello_gripper_open_degrees,
                self.config.gello_gripper_close_degrees,
            ),
        )

        driver = getattr(self.gello, "_driver", None)
        if getattr(driver, "_is_fake", False):
            raise RuntimeError(
                "GELLO Dynamixel driver fell back to a fake driver. "
                f"Check that the real GELLO is connected and accessible at {port}."
            )

    def _check_gello_port_access(self, port: str):
        """Fail early if the GELLO serial device cannot be opened by this user."""
        real_port = os.path.realpath(port)
        if not os.path.exists(port):
            raise FileNotFoundError(f"GELLO serial port does not exist: {port}")
        if not os.path.exists(real_port):
            raise FileNotFoundError(
                f"GELLO serial port target does not exist: {port} -> {real_port}"
            )
        if os.access(real_port, os.R_OK | os.W_OK):
            return

        stat_info = os.stat(real_port)
        user = os.environ.get("USER", "<unknown>")
        raise PermissionError(
            f"Cannot read/write GELLO serial port {port} -> {real_port} as user {user}. "
            f"Device uid={stat_info.st_uid}, gid={stat_info.st_gid}, "
            f"mode={oct(stat_info.st_mode & 0o777)}. "
            "Add your user to the device's group, install a udev rule, or run once with "
            f"`sudo chmod a+rw {real_port}` for a temporary test, then unplug/replug the GELLO."
        )

    def _smooth_move_to(self, target_joints: np.ndarray, step_rad: float = 0.01):
        """Smoothly interpolate Franka from its current pose to target_joints."""
        franka_current = self._get_franka_joints()
        max_delta = np.max(np.abs(target_joints - franka_current))
        num_steps = max(int(max_delta / step_rad), 1)

        gripper_action = getattr(self, '_last_gripper_action', self.config.gripper_open_action)

        for i in range(num_steps):
            alpha = (i + 1) / num_steps
            waypoint = franka_current + alpha * (target_joints - franka_current)
            deoxys_action = list(waypoint) + [self._deoxys_gripper_action(gripper_action)]
            self.robot_interface.control(
                controller_type=self.config.deoxys_controller_type,
                action=deoxys_action,
                controller_cfg=self.controller_cfg,
            )
        return max_delta, num_steps

    def run_calibration(self):
        """Move Franka to match the GELLO's current pose for absolute control."""
        input(
            "\n[Franka2CamRobot] Position the GELLO at your desired starting pose.\n"
            "Press Enter when ready — the Franka will move to match..."
        )

        gello_joints = np.array(self.gello.get_joint_state())
        gello_target = gello_joints[:7]
        franka_current = self._get_franka_joints()

        print(f"  Franka current: {np.round(franka_current, 4)}")
        print(f"  GELLO target:   {np.round(gello_target, 4)}")
        print(f"  Moving Franka to match GELLO...")

        self._smooth_move_to(gello_target)
        print("[Franka2CamRobot] Franka aligned to GELLO. Ready for teleop.")

    def _get_franka_joints(self) -> np.ndarray:
        """Read current 7-DOF joint positions from Franka."""
        return np.array(self.robot_interface._state_buffer[-1].q)

    def _deoxys_gripper_action(self, gripper_action: float) -> float:
        """Convert LeRobot gripper convention to deoxys' Franka-hand command."""
        return -1.0 if gripper_action == self.config.gripper_open_action else 1.0

    def _command_gripper(self, gripper_action: float):
        """Command the configured gripper when its logical state changes."""
        if hasattr(self, "_last_gripper_action") and gripper_action == self._last_gripper_action:
            return

        self._last_gripper_action = gripper_action

    def _get_gripper_width(self) -> float:
        """Read current gripper position normalized to [0, 1].

        Returns 1.0 for fully open, 0.0 for fully closed.
        """
        width = self.robot_interface.last_gripper_q
        if width is None:
            return getattr(self, "_last_gripper_width", self.config.gripper_open_action)
        normalized_width = float(np.clip(width / 0.08, 0.0, 1.0))
        self._last_gripper_width = normalized_width
        return normalized_width

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RuntimeError("Franka2CamRobot is not connected. Run `robot.connect()` first.")

        # Read GELLO state (7 arm joints + 1 gripper)
        before_lread_t = time.perf_counter()
        gello_joints = np.array(self.gello.get_joint_state())
        self.logs["read_leader_dt_s"] = time.perf_counter() - before_lread_t

        # Absolute control: GELLO is a kinematic replica of Franka, so joint
        # angles map 1:1 (offsets/signs already applied by DynamixelRobot)
        robot_target = gello_joints[:7]

        # Gripper thresholding
        if gello_joints[-1] > self.config.gripper_threshold:
            gripper_action = self.config.gripper_close_action
        else:
            gripper_action = self.config.gripper_open_action

        # Cache the logical gripper state; the Franka hand command is sent with
        # the arm command below.
        self._command_gripper(gripper_action)

        # Send joint target to Franka via deoxys — nothing runs between the
        # GELLO read and this call so teleoperation timing is unchanged.
        deoxys_action = list(robot_target) + [self._deoxys_gripper_action(gripper_action)]
        action = list(robot_target) + [gripper_action]
        before_fwrite_t = time.perf_counter()
        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )
        deoxys_dt = time.perf_counter() - before_fwrite_t
        self.logs["write_follower_dt_s"] = deoxys_dt

        if not record_data:
            return

        # Read Franka state and cameras AFTER the deoxys tick, then carry them
        # forward to the NEXT step as that step's "obs before action".  This
        # matches the gello run_control_loop convention:
        #   obs_t = env.step(action_{t-1})   ← read after prev action completes
        #   action_t = agent.act(obs_t)       ← GELLO reading at top of step t
        #   save(obs_t, action_t)             ← correct (pre-action) pairing
        #
        # On the very first call _teleop_prev_obs is not set, so we capture
        # the initial state here and return it immediately (one-step bootstrap).
        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        gripper_width = self._get_gripper_width()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        current_state = torch.tensor(
            list(franka_joints) + [gripper_width], dtype=torch.float32
        )
        current_images = self._read_cameras()

        action_tensor = torch.tensor(action, dtype=torch.float32)

        # Retrieve the obs captured at the end of the PREVIOUS step.
        prev_state = getattr(self, "_teleop_prev_state", None)
        prev_images = getattr(self, "_teleop_prev_images", None)

        # Store current obs for the next step.
        self._teleop_prev_state = current_state
        self._teleop_prev_images = current_images

        # Bootstrap: no previous obs yet, use current obs for this first frame.
        if prev_state is None:
            prev_state = current_state
            prev_images = current_images

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = prev_state
        action_dict["action"] = action_tensor
        for cam_name, img_dict in prev_images.items():
            for stream_name, tensor in img_dict.items():
                obs_dict[self._camera_observation_key(cam_name, stream_name)] = tensor

        return obs_dict, action_dict

    def _read_cameras(self) -> dict[str, dict[str, torch.Tensor]]:
        """Read all cameras and return {cam_name: {stream_name: tensor}}.

        Normalises every camera to dict form so observation keys are always
        `observation.images.<cam>.<stream>` (e.g. `.color`, `.transformed_depth`).
        Cameras that return a bare array (ZED without depth) are wrapped as
        {"color": tensor} to match AzureKinect convention.
        """
        images: dict[str, dict[str, torch.Tensor]] = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            raw = self.cameras[name].async_read()
            if isinstance(raw, dict):
                images[name] = {k: torch.from_numpy(v) for k, v in raw.items()}
            else:
                images[name] = {"color": torch.from_numpy(raw)}
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t
        return images

    def capture_observation(self) -> dict:
        """Read Franka joint state + gripper + camera images. For policy inference."""
        if not self.is_connected:
            raise RuntimeError("Franka2CamRobot is not connected. Run `robot.connect()` first.")

        before_fread_t = time.perf_counter()
        franka_joints = self._get_franka_joints()
        gripper_width = self._get_gripper_width()
        self.logs["read_follower_dt_s"] = time.perf_counter() - before_fread_t

        state = torch.tensor(
            list(franka_joints) + [gripper_width], dtype=torch.float32
        )

        images = self._read_cameras()

        obs_dict = {"observation.state": state}
        for cam_name, img_dict in images.items():
            for stream_name, tensor in img_dict.items():
                obs_dict[self._camera_observation_key(cam_name, stream_name)] = tensor

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Send an 8-value action tensor (7 joints + gripper) to Franka via deoxys."""
        if not self.is_connected:
            raise RuntimeError("Franka2CamRobot is not connected. Run `robot.connect()` first.")

        action_list = action.tolist()
        joint_target = np.array(action_list[:7])

        # Threshold continuous gripper value into open/close
        print(f"[action] joints={[round(v,4) for v in action_list[:7]]}  gripper={action_list[7]:.4f} threshold={self.config.gripper_threshold}")
        if action_list[7] < self.config.gripper_threshold:
            gripper_action = self.config.gripper_close_action
        else:
            gripper_action = self.config.gripper_open_action

        # Send gripper command only on change
        self._command_gripper(gripper_action)

        # Send joint target to Franka via deoxys
        deoxys_gripper_action = self._deoxys_gripper_action(gripper_action)
        print(
            f"[gripper] logical={'close' if gripper_action == self.config.gripper_close_action else 'open'} "
            f"deoxys={deoxys_gripper_action:.1f}"
        )
        deoxys_action = list(joint_target) + [deoxys_gripper_action]

        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )
        self.logs["write_follower_dt_s"] = 0.0

        return action

    def open_gripper(self):
        """Open the gripper and reset the cached gripper state."""
        current_joints = self._get_franka_joints()
        deoxys_action = list(current_joints) + [self._deoxys_gripper_action(self.config.gripper_open_action)]
        self.robot_interface.control(
            controller_type=self.config.deoxys_controller_type,
            action=deoxys_action,
            controller_cfg=self.controller_cfg,
        )
        self._last_gripper_action = self.config.gripper_open_action

    def print_logs(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            return

        if self.robot_interface is not None:
            self.robot_interface.close()
            self.robot_interface = None

        self.gello = None
        self.gello_home = None
        self.robot_home = None

        for name, camera in self.cameras.items():
            if getattr(camera, "is_connected", False):
                camera.disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
