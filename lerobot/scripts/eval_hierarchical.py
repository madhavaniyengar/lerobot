"""Closed-loop hierarchical evaluation for GHOST-on-LIBERO.

Composes a high-level subgoal predictor (`HighLevelWrapper`, loading a GHOST checkpoint
from WandB) with a low-level `DP3Policy` (diffusion policy) in a receding-horizon rollout
loop against a live `LiberoEnv`, and reports per-task / per-suite success rate.

This is genuinely new code. As of the pinned r-pad/lerobot rev this repo was cloned at
(825eaf8...), `DP3Policy` instantiates a `HighLevelWrapper` internally when
`config.enable_goal_conditioning` is set, but never calls it -- the actual composition of
"high-level predicts a subgoal every N steps -> low-level acts toward it" was not wired up
anywhere in this codebase. This script is that missing piece, and is meant to be the seam
for swapping in different high-level/low-level architectures later (a different
`high_level.model_type`, or a different low-level policy class, without touching the
rollout loop itself).

Sibling to `eval_suite.py` (reuses its suite-enumeration / result-aggregation conventions),
but that script evaluates a single monolithic policy; this one drives two independently
loaded checkpoints per episode.

NOTE -- pieces flagged below as best-effort still need validation against a real trained
low-level checkpoint (none exists yet in this session):
  1. `_build_observation_state`: the exact 8-dim `observation.state` vector construction
     (ee_pos + axis-angle orientation + 2-dim gripper state) mirrors
     `create_libero_dataset.py`'s `prep_ee_pose`, but `LiberoEnv`'s live observation only
     exposes a quaternion + a single scalar gripper angle, so this reconstructs the
     axis-angle and duplicates the gripper scalar into 2 dims as an approximation.
  2. `get_scene_point_cloud`'s `workspace_bounds` (in `libero_franka_utils.py`) are a rough
     guess at the LIBERO tabletop region -- validate against a rendered point cloud before
     trusting it for a trained policy.

Usage:
    python lerobot/scripts/eval_hierarchical.py \\
        --high_level.model_type=dino_3dgp --high_level.run_id=<wandb_run_id> \\
        --low_level_checkpoint=outputs/train/libero_lowlevel_run1/checkpoints/last/pretrained_model \\
        --suite_name=libero_goal --n_episodes=10 --replan_every=20 \\
        --output_dir=outputs/eval/libero_hierarchical_run1
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from lerobot.common.envs.libero_env import LiberoEnv
from lerobot.common.policies.dp3.modeling_dp3 import DP3Policy
from lerobot.common.policies.high_level.high_level_wrapper import HighLevelConfig, HighLevelWrapper
from lerobot.common.utils.libero_franka_utils import get_scene_point_cloud
from lerobot.common.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser

from libero.libero import benchmark


@dataclass
class EvalHierarchicalConfig:
    high_level: HighLevelConfig = field(default_factory=HighLevelConfig)
    low_level_checkpoint: str = ""  # path to a DP3Policy.from_pretrained-loadable dir
    suite_name: str = "libero_goal"
    task_ids: Optional[str] = None  # comma-separated, default = all tasks in suite
    n_episodes: int = 10
    max_steps: int = 300
    replan_every: int = 20  # receding-horizon: re-run the high-level every N env steps
    settle_steps: int = 5
    camera_heights: int = 256
    camera_widths: int = 256
    seed: int = 0
    output_dir: str = "outputs/eval/libero_hierarchical"
    device: str = "cuda"


def get_available_suites() -> List[str]:
    benchmark_dict = benchmark.get_benchmark_dict()
    return [s for s in benchmark_dict.keys() if s != "libero_100"]


def _build_observation_state(robot_data: Dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct the 8-dim `observation.state` vector DP3 was trained on
    (ee_pos[3] + ee_ori_axis_angle[3] + gripper_state[2]) from LiberoEnv's live obs
    (ee_pos[3] + ee_quat[4] + gripper_angle[scalar]). See module docstring note (1)."""
    ee_pos = np.asarray(robot_data["ee_pos"], dtype=np.float32)
    axis_angle = R.from_quat(robot_data["ee_quat"]).as_rotvec().astype(np.float32)
    g = float(robot_data["gripper_angle"])
    gripper_state = np.array([g, -g], dtype=np.float32)  # mirrors symmetric finger qpos
    return np.concatenate([ee_pos, axis_angle, gripper_state])


def rollout_episode(
    env: LiberoEnv,
    high_level: HighLevelWrapper,
    low_level: DP3Policy,
    task_language: str,
    cfg: EvalHierarchicalConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> bool:
    """Run one closed-loop episode: replan the high-level subgoal every `cfg.replan_every`
    steps, step the low-level policy every step. Returns whether the episode succeeded."""
    obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    low_level.reset()

    for _ in range(cfg.settle_steps):
        obs, _, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
        if terminated or truncated:
            return bool(info.get("is_success", False))

    # Static agentview calibration for this env instance (constant across the episode).
    agentview_ext = get_camera_extrinsic_matrix(env.env.sim, "agentview")
    agentview_int = get_camera_intrinsic_matrix(
        env.env.sim, "agentview", cfg.camera_widths, cfg.camera_heights
    )

    goal_gripper_pcd_world = None
    for step in range(cfg.max_steps):
        agentview_rgb = obs["pixels"]["agentview"]
        agentview_depth = obs["depth"]["agentview"]
        robot_data = obs["robot_data"]  # {"ee_pos", "ee_quat", "gripper_angle"}

        if step % cfg.replan_every == 0:
            camera_obs = {"agentview": {"rgb": agentview_rgb, "depth": agentview_depth}}
            goal_gripper_pcd_world = high_level.predict(
                text=task_language,
                camera_obs=camera_obs,
                robot_type="libero_franka",
                robot_kwargs=robot_data,
            )

        point_cloud = get_scene_point_cloud(
            agentview_depth, agentview_int, agentview_ext, rng=rng
        )
        obs_state = _build_observation_state(robot_data)

        batch = {
            "observation.state": torch.from_numpy(obs_state).float().unsqueeze(0).to(device),
            "observation.points.point_cloud": torch.from_numpy(point_cloud)
            .float()
            .unsqueeze(0)
            .to(device),
            "observation.points.goal_gripper_pcds": torch.from_numpy(goal_gripper_pcd_world)
            .float()
            .unsqueeze(0)
            .to(device),
        }
        with torch.no_grad():
            action, _ = low_level.select_action(batch)
        action_np = action.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        if info.get("is_success"):
            return True
        if terminated or truncated:
            return False

    return False


def eval_hierarchical(cfg: EvalHierarchicalConfig, device: torch.device) -> Dict[str, Any]:
    high_level = HighLevelWrapper(cfg.high_level)
    low_level = DP3Policy.from_pretrained(cfg.low_level_checkpoint)
    low_level.to(device)
    low_level.eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    if cfg.suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown suite {cfg.suite_name}, available: {get_available_suites()}"
        )
    task_suite = benchmark_dict[cfg.suite_name]()
    task_ids = (
        [int(t) for t in cfg.task_ids.split(",")]
        if cfg.task_ids is not None
        else list(range(len(task_suite.tasks)))
    )

    rng = np.random.default_rng(cfg.seed)
    results: Dict[str, Any] = {"suite_name": cfg.suite_name, "tasks": {}}
    all_success_rates = []

    for task_id in tqdm(task_ids, desc=f"Evaluating {cfg.suite_name}"):
        task = task_suite.get_task(task_id)
        env = LiberoEnv(
            task_suite_name=cfg.suite_name,
            task_id=task_id,
            camera_heights=cfg.camera_heights,
            camera_widths=cfg.camera_widths,
            max_episode_steps=cfg.max_steps,
        )
        successes = [
            rollout_episode(env, high_level, low_level, task.language, cfg, device, rng)
            for _ in range(cfg.n_episodes)
        ]
        env.close()

        success_rate = float(np.mean(successes))
        all_success_rates.append(success_rate)
        results["tasks"][f"{cfg.suite_name}_{task_id}"] = {
            "name": task.name,
            "language": task.language,
            "success_rate": success_rate,
            "n_episodes": cfg.n_episodes,
        }
        logging.info(f"Task {cfg.suite_name}_{task_id} ({task.language}): {success_rate:.1%}")

    results["overall_success_rate"] = float(np.mean(all_success_rates))
    return results


@parser.wrap()
def main(cfg: EvalHierarchicalConfig):
    init_logging()
    logging.info("GHOST-on-LIBERO hierarchical evaluation")
    logging.info(json.dumps(asdict(cfg), indent=2, default=str))

    device = get_safe_torch_device(cfg.device, log=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = eval_hierarchical(cfg, device)

    print("\n" + "=" * 60)
    print("GHOST-ON-LIBERO HIERARCHICAL EVAL RESULTS")
    print("=" * 60)
    print(f"Suite: {results['suite_name']}")
    print(f"Overall success rate: {results['overall_success_rate']:.1%}")
    for task_key, task_result in results["tasks"].items():
        print(f"  {task_key} ({task_result['language']}): {task_result['success_rate']:.1%}")
    print("=" * 60)

    with open(output_dir / "eval_hierarchical_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {output_dir / 'eval_hierarchical_results.json'}")


if __name__ == "__main__":
    main()
