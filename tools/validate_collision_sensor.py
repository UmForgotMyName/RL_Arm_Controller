# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate contact sensing by forcing an obstacle overlap with the robot."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Validate obstacle contact sensing.")
parser.add_argument("--task", type=str, default="Isaac-Reach-Fanuc-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=10, help="Number of steps to run for validation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import RL_Arm_Controller.tasks  # noqa: F401


def _summarize_contact_forces(sensor, threshold: float):
    if sensor.cfg.filter_prim_paths_expr:
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            return None, None, []
        norms = torch.norm(force_matrix, dim=-1)
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()
        contact_mask = norms > threshold
        contact_bodies = contact_mask.any(dim=-1)
    else:
        net_forces = sensor.data.net_forces_w
        if net_forces is None:
            return None, None, []
        norms = torch.norm(net_forces, dim=-1)
        max_norm = norms.max().item()
        mean_norm = norms.mean().item()
        contact_bodies = norms > threshold
    body_names = sensor.body_names
    in_contact = []
    if len(body_names) > 0:
        env0_mask = contact_bodies[0]
        if env0_mask.numel() == len(body_names):
            in_contact = [body_names[i] for i in range(len(body_names)) if env0_mask[i].item()]
    return max_norm, mean_norm, in_contact


def main():
    """Force an obstacle overlap and verify the contact sensor reports it."""
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    env_unwrapped = env.unwrapped
    sensor = env_unwrapped._robot_contact_sensor
    if sensor is None:
        sensor = env_unwrapped.scene.sensors.get("robot_contact_sensor")
    if sensor is None:
        raise RuntimeError("Robot contact sensor is not initialized.")
    threshold = env_unwrapped.cfg.collision_force_threshold

    if env_unwrapped.num_envs != 1:
        print(f"[WARN] Expected num_envs=1; got {env_unwrapped.num_envs}. Validation will use env 0.")

    env_ids = torch.arange(env_unwrapped.num_envs, device=env_unwrapped.device)
    tcp_pos, _ = env_unwrapped._get_tcp_pose(env_ids)
    obstacle_pos = env_unwrapped._obstacle_pos.clone()
    obstacle_pos[:, 0, :] = tcp_pos
    env_unwrapped._obstacle_pos[:] = obstacle_pos
    env_unwrapped._write_obstacle_positions(env_ids, obstacle_pos)
    env_unwrapped.scene.write_data_to_sim()
    env_unwrapped.sim.forward()
    env_unwrapped.scene.update(dt=env_unwrapped.physics_dt)

    print(f"[INFO] Collision force threshold: {threshold}")
    print(f"[INFO] Contact sensor bodies: {sensor.body_names}")

    collision_seen = False
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env_unwrapped.device)
            env.step(actions)
        collision = env_unwrapped._check_obstacle_collision()
        max_norm, mean_norm, bodies = _summarize_contact_forces(sensor, threshold)
        print(
            f"[STEP {step}] collision={bool(collision[0].item())} "
            f"max_norm={max_norm} mean_norm={mean_norm} bodies={bodies}"
        )
        if collision.any():
            collision_seen = True

    if not collision_seen:
        raise RuntimeError("Collision was not detected within the validation window.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
