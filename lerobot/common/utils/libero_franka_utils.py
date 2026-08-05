import numpy as np
from scipy.spatial.transform import Rotation as R
import os
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import re

def get_4_points_from_gripper_pos_orient(gripper_pos, gripper_orn, cur_joint_angle, world_to_cam_mat=None):
    """
    From https://github.com/NakuraMino/articubot-on-mimicgen/blob/main/third_party/robogen/robogen_utils.py
    Analytically calculates 4-points on the Franka gripper

    Args:
        gripper_pos (np.ndarray): 3D position of gripper end-effector [x, y, z]
        gripper_orn (np.ndarray): Quaternion orientation of gripper [x, y, z, w]
        cur_joint_angle (float): Current gripper joint angle (0 = closed, 0.04 = open)
        world_to_cam_mat (np.ndarray, optional): 4x4 world-to-camera transform matrix.
                                               If provided, returns points in camera frame.

    Returns:
        np.ndarray: 4x3 array of gripper point cloud coordinates (world or camera frame)
    """
    original_gripper_pcd = np.array([[ 0.5648266,   0.05482348,  0.34434554],
        [ 0.5642125,   0.02702148,  0.2877661 ],
        [ 0.53906703,  0.01263776,  0.38347825],
        [ 0.54250515, -0.00441092,  0.32957944]]
    )
    original_gripper_orn = np.array([0.21120763,  0.75430543, -0.61925177, -0.05423936])

    gripper_pcd_right_finger_closed = np.array([ 0.55415434,  0.02126799,  0.32605097])
    gripper_pcd_left_finger_closed = np.array([ 0.54912525,  0.01839125,  0.3451934 ])
    gripper_pcd_closed_finger_angle = 2.6652539383870777e-05

    original_gripper_pcd[1] = gripper_pcd_right_finger_closed + (original_gripper_pcd[1] - gripper_pcd_right_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)
    original_gripper_pcd[2] = gripper_pcd_left_finger_closed + (original_gripper_pcd[2] - gripper_pcd_left_finger_closed) / (0.04 - gripper_pcd_closed_finger_angle) * (cur_joint_angle - gripper_pcd_closed_finger_angle)

    goal_R = R.from_quat(gripper_orn)
    original_R = R.from_quat(original_gripper_orn)
    rotation_transfer = goal_R * original_R.inv()
    original_pcd = original_gripper_pcd - original_gripper_pcd[3]
    rotated_pcd = rotation_transfer.apply(original_pcd)
    gripper_pcd = rotated_pcd + gripper_pos

    # Transform to camera frame if transformation matrix is provided
    if world_to_cam_mat is not None:
        # Convert to homogeneous coordinates
        gripper_pcd_hom = np.hstack([gripper_pcd, np.ones((gripper_pcd.shape[0], 1))])
        # Transform to camera frame
        gripper_pcd_cam = world_to_cam_mat @ gripper_pcd_hom.T
        gripper_pcd = gripper_pcd_cam[:3].T  # Drop homogeneous coordinate

    return gripper_pcd.astype(np.float32)

def get_libero_caption(h5_fpath):
    """
    Some hacky string processing to extract captions from the demo fname....
    """
    h5_fname = os.path.basename(h5_fpath)
    # Remove .hdf5 and _demo suffix
    base = h5_fname.replace('.hdf5', '').replace('_demo', '')
    if '_SCENE' in base:
        # Find last occurrence of SCENE[digit]_ pattern
        base = re.sub(r'^[A-Z_]+_SCENE\d+_', '', base)

    # Convert underscores to spaces
    caption = base.replace('_', ' ')
    return caption

def setup_libero_env(task_bddl_file, img_shape):
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_shape[0],
        "camera_widths": img_shape[1],
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    return env

def get_scene_point_cloud(
    depth,
    intrinsic_matrix,
    cam_to_world_mat,
    num_points=4500,
    workspace_bounds=((-0.5, 0.5), (-0.5, 0.5), (-0.05, 1.0)),
    rng=None,
):
    """Unproject a depth image into a fixed-size world-frame scene point cloud.

    DP3Policy's `forward`/`select_action` (lerobot/common/policies/dp3/modeling_dp3.py) reads
    `observation.points.point_cloud` unconditionally -- it's core to the DP3 ("3D Diffusion
    Policy") architecture, not just for goal-conditioning. `create_libero_dataset.py` doesn't
    produce this feature upstream (only gripper/goal point clouds), so this fills that gap:
    used both at dataset-conversion time (create_libero_dataset.py, --new_features point_cloud)
    and at closed-loop eval time (eval_hierarchical.py) so the two stay consistent.

    Args:
        depth: (H, W) metric depth in meters.
        intrinsic_matrix: (3, 3) camera intrinsics.
        cam_to_world_mat: (4, 4) camera-to-world extrinsics (as returned by
            robosuite.utils.camera_utils.get_camera_extrinsic_matrix).
        num_points: fixed output point count (randomly subsampled or padded via
            with-replacement resampling to hit this count).
        workspace_bounds: ((xmin,xmax),(ymin,ymax),(zmin,zmax)) crop in world frame, to
            discard points off the table / outside the workspace. Approximate defaults for
            the LIBERO tabletop setup -- validate against a real scene before relying on this
            for a trained policy.
        rng: optional np.random.Generator for deterministic subsampling.

    Returns:
        (num_points, 3) float32 array of world-frame points.
    """
    rng = rng or np.random.default_rng()
    depth = np.asarray(depth).squeeze()  # drop trailing channel dim, e.g. (H, W, 1) -> (H, W)
    h, w = depth.shape[:2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    zs = depth.astype(np.float32)
    valid = zs > 0
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    xs = (us - cx) * zs / fx
    ys = (vs - cy) * zs / fy
    # Camera convention here follows robosuite's OpenGL-style camera (matches
    # project_points_to_image / get_camera_intrinsic_matrix usage elsewhere in this file).
    pts_cam = np.stack([xs, ys, zs], axis=-1)[valid]  # (M, 3)
    pts_cam_hom = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)
    pts_world = (cam_to_world_mat @ pts_cam_hom.T).T[:, :3]

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = workspace_bounds
    in_bounds = (
        (pts_world[:, 0] >= xmin) & (pts_world[:, 0] <= xmax)
        & (pts_world[:, 1] >= ymin) & (pts_world[:, 1] <= ymax)
        & (pts_world[:, 2] >= zmin) & (pts_world[:, 2] <= zmax)
    )
    pts_world = pts_world[in_bounds]

    if len(pts_world) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    replace = len(pts_world) < num_points
    idx = rng.choice(len(pts_world), size=num_points, replace=replace)
    return pts_world[idx].astype(np.float32)


def prepare_caption_to_bddl_mapping():
    """Map caption extracted through fname processing to a BDDL file of the environment"""
    mapping_dict = {}
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite in ["libero_goal", "libero_object", "libero_spatial", "libero_90", "libero_10"]:
        task_suite = benchmark_dict[suite]()
        for task_id in range(len(task_suite.tasks)):
            task = task_suite.get_task(task_id)
            caption = task.language
            task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            mapping_dict[caption] = task_bddl_file
    return mapping_dict
