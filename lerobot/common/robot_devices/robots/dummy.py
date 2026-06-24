"""DummyRobot: Camera-only recording with no physical robot connection.

Mimics the DroidRobot interface but skips all hardware (Franka, GELLO, Robotiq).
Returns zero-filled state/action tensors so the dataset schema stays compatible
with real Droid datasets.
"""

import time
from threading import Thread

import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import DummyRobotConfig


class DummyRobot:
    """Dummy robot that only connects cameras. No robot hardware needed."""

    robot_type = "dummy"

    def __init__(self, config: DummyRobotConfig | None = None, **kwargs):
        if config is None:
            self.config = DummyRobotConfig(**kwargs)
        else:
            self.config = config

        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}
        self.robot_type = self.config.type
        self.use_eef = self.config.use_eef

        self.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4",
            "joint_5", "joint_6", "joint_7", "gripper",
        ]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            cam_ft.update(cam.config.get_feature_specs(cam_key))
        return cam_ft

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
            eef_names = [
                "rot_6d_0", "rot_6d_1", "rot_6d_2", "rot_6d_3", "rot_6d_4", "rot_6d_5",
                "trans_0", "trans_1", "trans_2", "gripper_articulation",
            ]
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
            raise RuntimeError("DummyRobot is already connected.")

        azure_kinect_cameras = []
        try:
            for camera in self.cameras.values():
                if camera.__class__.__name__ == "AzureKinectCamera":
                    camera.connect(start_cameras=False)
                    azure_kinect_cameras.append(camera)
                else:
                    camera.connect()

            if len(azure_kinect_cameras) > 0:
                self._start_azure_kinect_cameras(azure_kinect_cameras)
        except Exception:
            self._disconnect_cameras()
            raise

        self.is_connected = True
        print("[DummyRobot] Connected cameras only (no robot hardware).")

    def _start_azure_kinect_cameras(self, cameras):
        subordinates = [cam for cam in cameras if cam.wired_sync_mode == "subordinate"]
        masters = [cam for cam in cameras if cam.wired_sync_mode == "master"]
        standalone = [cam for cam in cameras if cam.wired_sync_mode is None]

        for camera_group in (subordinates, masters, standalone):
            self._start_azure_kinect_camera_group(camera_group)

    def _start_azure_kinect_camera_group(self, cameras):
        if len(cameras) == 0:
            return

        camera_start_errors = []

        def start_camera(cam):
            try:
                cam.start()
            except Exception as exc:
                camera_start_errors.append((cam, exc))

        threads = [Thread(target=start_camera, args=(cam,)) for cam in cameras]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if camera_start_errors:
            errors = ", ".join(
                f"AzureKinectCamera({cam.device_id}): {exc}" for cam, exc in camera_start_errors
            )
            raise RuntimeError(f"Failed to start Azure Kinect camera(s): {errors}")

    def _disconnect_cameras(self):
        for camera in self.cameras.values():
            if getattr(camera, "is_connected", False):
                camera.disconnect()
            elif (
                camera.__class__.__name__ == "AzureKinectCamera"
                and getattr(camera, "camera", None) is not None
            ):
                if getattr(camera.camera, "opened", False):
                    camera.camera.close()
                camera.camera = None

    def run_calibration(self):
        pass

    def teleop_step(self, record_data=False):
        if not self.is_connected:
            raise RuntimeError("DummyRobot is not connected.")

        if not record_data:
            return

        state = torch.zeros(len(self.joint_names), dtype=torch.float32)
        action_tensor = torch.zeros(len(self.joint_names), dtype=torch.float32)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    images[name][img_name] = torch.from_numpy(images[name][img_name])
            else:
                images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action_tensor
        for name in self.cameras:
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    obs_dict[f"observation.images.{name}.{img_name}"] = images[name][img_name]
            else:
                obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self) -> dict:
        if not self.is_connected:
            raise RuntimeError("DummyRobot is not connected.")

        state = torch.zeros(len(self.joint_names), dtype=torch.float32)

        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    images[name][img_name] = torch.from_numpy(images[name][img_name])
            else:
                images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            if type(images[name]) == dict:
                for img_name in images[name].keys():
                    obs_dict[f"observation.images.{name}.{img_name}"] = images[name][img_name]
            else:
                obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def teleop_safety_stop(self):
        pass

    def disconnect(self):
        if not self.is_connected:
            return
        self._disconnect_cameras()
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
