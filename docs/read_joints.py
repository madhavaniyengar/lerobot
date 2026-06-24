"""Read and print joint positions from both GELLO and Franka in a loop.

Usage:
    python docs/read_joints.py --robot-type franka_2cam

Use this to verify that GELLO joint offsets are correct. When the GELLO and
Franka are physically in the same pose, their printed joint values should match.
If they don't, adjust gello_joint_offsets and gello_joint_signs in lerobot/common/robot_devices/robots/configs.py.
"""

import argparse
import glob
import time

import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Read GELLO and Franka joint positions.")
    parser.add_argument("--robot-type", type=str, default="franka_2cam", help="Robot config type to use.")
    parser.add_argument("--config", type=str, default=None, help="Optional path to deoxys YAML config file.")
    args = parser.parse_args()

    from deoxys.franka_interface import FrankaInterface
    from gello.robots.dynamixel import DynamixelRobot
    from lerobot.common.robot_devices.robots.utils import make_robot_config

    robot_cfg = make_robot_config(args.robot_type)
    deoxys_config = args.config or robot_cfg.deoxys_general_cfg_file

    # --- Franka setup ---
    interface = FrankaInterface(
        deoxys_config,
        use_visualizer=False,
    )
    print("Waiting for Franka state buffer...")
    while len(interface._state_buffer) == 0:
        time.sleep(0.1)
    print("Franka connected.")

    matches = [p for p in glob.glob("/dev/serial/by-id/*") if "Serial_Converter" in p]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one GELLO serial device, found {matches}.")
    port = matches[0]
    gello = DynamixelRobot(
        joint_ids=list(robot_cfg.gello_joint_ids),
        joint_offsets=list(robot_cfg.gello_joint_offsets),
        real=True,
        joint_signs=list(robot_cfg.gello_joint_signs),
        port=port,
        gripper_config=(
            robot_cfg.gello_gripper_joint_id,
            robot_cfg.gello_gripper_open_degrees,
            robot_cfg.gello_gripper_close_degrees,
        ),
    )
    print("GELLO connected.")

    # --- Print loop ---
    print("\nReading joints (Ctrl+C to stop)...\n")
    try:
        while True:
            franka_q = np.array(interface._state_buffer[-1].q)
            gello_q = np.array(gello.get_joint_state())[:7]
            diff = franka_q - gello_q

            print(f"Franka: {np.array2string(franka_q, precision=4, suppress_small=True)}")
            print(f"GELLO:  {np.array2string(gello_q, precision=4, suppress_small=True)}")
            print(f"Diff:   {np.array2string(diff, precision=4, suppress_small=True)}")
            print(f"Max diff: {np.max(np.abs(diff)):.4f} rad")
            print("-" * 70)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
