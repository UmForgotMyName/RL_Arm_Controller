# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import normalize, quat_apply, sample_uniform

from .rl_arm_controller_env_cfg import RlArmControllerEnvCfg


class RlArmControllerEnv(DirectRLEnv):
    cfg: RlArmControllerEnvCfg

    def __init__(self, cfg: RlArmControllerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- Robot handles and kinematics ----
        self._joint_ids, _ = self._robot.find_joints(self.cfg.joint_names, preserve_order=True)
        self._tcp_body_idx = self._robot.find_bodies(self.cfg.tcp_body_name)[0][0]

        self._default_joint_pos = self._robot.data.default_joint_pos[:, self._joint_ids].clone()
        limits = self._robot.data.soft_joint_pos_limits[0, self._joint_ids].to(self.device)
        self._dof_lower_limits = limits[:, 0]
        self._dof_upper_limits = limits[:, 1]
        if self._robot.is_fixed_base:
            self._jacobi_body_idx = self._tcp_body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._tcp_body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # ---- Action buffers ----
        num_joints = len(self._joint_ids)
        self._actions = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._joint_targets = self._default_joint_pos.clone()

        # ---- Task state buffers ----
        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._obstacle_pos = torch.zeros((self.num_envs, self.cfg.num_obstacles, 3), device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._has_prev_dist = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_stats_initialized = False
        self._invalid_reset_count = 0
        self._total_reset_count = 0
        self._degraded_reset_count = 0
        self._degraded_target_count = 0
        self._degraded_obstacle_count = 0
        self._repair_attempts_total = 0
        self._repair_attempts_count = 0
        self._invalid_reset_ema = 0.0
        self._invalid_reset_window = torch.zeros(
            self.cfg.invalid_fraction_window, dtype=torch.bool, device=self.device
        )
        self._invalid_reset_window_idx = 0
        self._invalid_reset_window_filled = 0
        self._degraded_target_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._degraded_obstacle_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._best_dist = torch.zeros(self.num_envs, device=self.device)
        self._stuck_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stuck_terminated_count = 0

        # ---- Geometry helpers ----
        self._tcp_forward_axis = torch.tensor(self.cfg.tcp_forward_axis, device=self.device).repeat(self.num_envs, 1)
        self._base_pos_local = torch.tensor(self.cfg.robot_cfg.init_state.pos, device=self.device)
        obstacle_diag = torch.tensor(self.cfg.obstacle_size, device=self.device)
        self._obstacle_clearance_radius = 0.5 * torch.linalg.norm(obstacle_diag) + self.cfg.los_clearance_margin
        self._obstacle_inactive_pos = torch.tensor(self.cfg.obstacle_inactive_pos, device=self.device)

        # ---- Visualization ----
        self._target_markers = VisualizationMarkers(self.cfg.target_marker_cfg)
        # ---- Curriculum ----
        self._curriculum_cfg = self.cfg.curriculum
        self._curriculum_enabled = (
            self._curriculum_cfg is not None
            and self._curriculum_cfg.enabled
            and len(self._curriculum_cfg.stages) > 0
        )
        self._curriculum_stage = 0
        self._curr_target_pos_range = self.cfg.target_pos_range
        self._curr_obstacle_pos_range = self.cfg.obstacle_pos_range
        self._curr_success_tolerance = self.cfg.success_tolerance
        self._active_obstacle_count = self.cfg.num_obstacles
        if self._curriculum_enabled:
            self._apply_curriculum_stage(0)
        else:
            self._update_target_bounds_tensors()
        self._episode_success_history = torch.zeros(self._curriculum_cfg.window_size, device=self.device)
        self._episode_history_idx = 0
        self._episode_history_filled = 0
        self._success_rate = 0.0
        stage_count = len(self._curriculum_cfg.stages) if self._curriculum_cfg is not None else 0
        self._reset_count_by_stage = [0 for _ in range(stage_count)]
        self._invalid_count_by_stage = [0 for _ in range(stage_count)]
        self._degraded_count_by_stage = [0 for _ in range(stage_count)]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # ---- Obstacles and contact sensors ----
        self._obstacles: list[RigidObject] = []
        self._obstacle_contact_sensors: list[ContactSensor] = []
        for i in range(self.cfg.num_obstacles):
            obstacle_cfg = self.cfg.obstacle_cfg.replace(prim_path=f"/World/envs/env_.*/Obstacle_{i}")
            obstacle = RigidObject(obstacle_cfg)
            self._obstacles.append(obstacle)
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle

            sensor_cfg = self.cfg.obstacle_contact_sensor.replace(prim_path=f"/World/envs/env_.*/Obstacle_{i}")
            sensor = ContactSensor(sensor_cfg)
            self._obstacle_contact_sensors.append(sensor)
            self.scene.sensors[f"obstacle_contact_sensor_{i}"] = sensor

        # ---- Static world ----
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # ---- Clone environments ----
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # ---- Lighting ----
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Clamp to action space and map to joint targets.
        self._actions = actions.clamp(-1.0, 1.0)
        targets = self._default_joint_pos + self.cfg.action_scale * self._actions
        self._joint_targets = torch.clamp(targets, self._dof_lower_limits, self._dof_upper_limits)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # ---- State terms ----
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids] * self.cfg.dof_velocity_scale

        tcp_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx] - self.scene.env_origins
        target_rel = self._target_pos - tcp_pos

        # ---- Assemble observation ----
        obs_terms = [self._scale_joint_pos(joint_pos), joint_vel, target_rel]

        if self.cfg.include_obstacle_obs:
            obstacle_rel = self._obstacle_pos - tcp_pos.unsqueeze(1)
            obs_terms.append(obstacle_rel.view(self.num_envs, -1))

        if self.cfg.include_prev_actions:
            obs_terms.append(self._prev_actions)

        obs = torch.cat(obs_terms, dim=-1)
        obs = torch.clamp(obs, -self.cfg.obs_clip, self.cfg.obs_clip)

        self._prev_actions.copy_(self._actions)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # ---- Distance and progress ----
        tcp_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx] - self.scene.env_origins
        to_target = self._target_pos - tcp_pos
        dist = torch.norm(to_target, dim=-1)

        reward = -self.cfg.rew_scale_dist * dist
        reward += (dist < self._curr_success_tolerance).float() * self.cfg.rew_scale_success

        progress = self._prev_dist - dist
        progress = torch.where(self._has_prev_dist, progress, torch.zeros_like(progress))
        reward += self.cfg.rew_scale_progress * progress

        # ---- Action regularization ----
        action_penalty = torch.sum(self._actions**2, dim=-1)
        reward += self.cfg.rew_scale_action * action_penalty

        action_rate_penalty = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        reward += self.cfg.rew_scale_action_rate * action_rate_penalty

        joint_vel_penalty = torch.sum(self._robot.data.joint_vel[:, self._joint_ids] ** 2, dim=-1)
        reward += self.cfg.rew_scale_joint_vel * joint_vel_penalty

        # ---- Optional orientation shaping ----
        if self.cfg.rew_scale_approach != 0.0:
            tcp_rot = self._robot.data.body_quat_w[:, self._tcp_body_idx]
            tcp_forward = quat_apply(tcp_rot, self._tcp_forward_axis)
            to_target_dir = normalize(to_target)
            reward += self.cfg.rew_scale_approach * torch.sum(tcp_forward * to_target_dir, dim=-1)

        # ---- Collision penalty ----
        if self.cfg.rew_scale_collision != 0.0:
            collision = self._check_obstacle_collision()
            reward = torch.where(collision, reward + self.cfg.rew_scale_collision, reward)
        reward = torch.where(self._invalid_reset_buf, torch.zeros_like(reward), reward)

        self._prev_dist = dist
        self._has_prev_dist[:] = True

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- Termination conditions ----
        tcp_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx] - self.scene.env_origins
        dist = torch.norm(self._target_pos - tcp_pos, dim=-1)
        success = dist < self._curr_success_tolerance
        self._success_buf = success

        terminated = success
        if self.cfg.terminate_on_collision:
            collision = self._check_obstacle_collision()
            terminated = terminated | collision

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.enable_stuck_termination:
            # Track progress and truncate if there is no improvement for a while.
            improved = dist < (self._best_dist - self.cfg.stuck_min_progress)
            self._best_dist = torch.minimum(self._best_dist, dist)
            self._stuck_steps = torch.where(improved, torch.zeros_like(self._stuck_steps), self._stuck_steps + 1)
            stuck = (self._stuck_steps >= self.cfg.stuck_steps) & ~terminated
            time_out = time_out | stuck
            if stuck.any():
                self._stuck_terminated_count += stuck.sum().item()
        self.extras["Stats/stuck_terminated_count"] = float(self._stuck_terminated_count)
        time_out = time_out | self._invalid_reset_buf
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # ---- Reset pipeline ----
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if self._episode_stats_initialized:
            self._update_success_history(env_ids)
        else:
            self._episode_stats_initialized = True

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        noise = sample_uniform(-1.0, 1.0, (len(env_ids), len(self._joint_ids)), device=self.device)
        joint_pos[:, self._joint_ids] = self._default_joint_pos[env_ids] + noise * self.cfg.reset_joint_pos_noise
        joint_pos[:, self._joint_ids] = torch.clamp(
            joint_pos[:, self._joint_ids], self._dof_lower_limits, self._dof_upper_limits
        )
        joint_vel[:, self._joint_ids] = 0.0

        self._default_joint_pos[env_ids] = joint_pos[:, self._joint_ids]

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(joint_pos[:, self._joint_ids], joint_ids=self._joint_ids, env_ids=env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(dt=self.physics_dt)

        # ---- Sampling: targets then obstacles ----
        target_valid, target_degraded = self._reset_targets(env_ids)
        obstacle_valid, obstacle_degraded = self._reset_obstacles(env_ids)
        invalid_mask = ~(target_valid & obstacle_valid)
        degraded_mask = target_degraded | obstacle_degraded
        initial_invalid_mask = invalid_mask.clone()

        # ---- Repair loop: resample only the failing envs ----
        attempts_used = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        pending_mask = invalid_mask.clone()
        if self.cfg.reset_repair_attempts > 0 and pending_mask.any():
            for attempt in range(self.cfg.reset_repair_attempts):
                if not pending_mask.any():
                    break
                pending_env_ids = env_ids[pending_mask]
                target_valid_r, target_degraded_r = self._reset_targets(pending_env_ids)
                obstacle_valid_r, obstacle_degraded_r = self._reset_obstacles(pending_env_ids)
                invalid_r = ~(target_valid_r & obstacle_valid_r)
                degraded_r = target_degraded_r | obstacle_degraded_r

                invalid_mask[pending_mask] = invalid_r
                target_degraded[pending_mask] = target_degraded_r
                obstacle_degraded[pending_mask] = obstacle_degraded_r
                degraded_mask[pending_mask] = degraded_r
                attempts_used[pending_mask] = attempt + 1
                pending_mask[pending_mask] = invalid_r

        # ---- Hard fallback: last resort to keep episodes usable ----
        if pending_mask.any():
            pending_env_ids = env_ids[pending_mask]
            self._apply_degraded_reset(pending_env_ids)
            invalid_mask[pending_mask] = False
            target_degraded[pending_mask] = True
            obstacle_degraded[pending_mask] = True
            degraded_mask[pending_mask] = True

        self._invalid_reset_buf[env_ids] = invalid_mask
        self._degraded_target_buf[env_ids] = target_degraded
        self._degraded_obstacle_buf[env_ids] = obstacle_degraded

        raw_invalid_count = initial_invalid_mask.sum().item()
        degraded_count = degraded_mask.sum().item()
        degraded_target_count = target_degraded.sum().item()
        degraded_obstacle_count = obstacle_degraded.sum().item()

        self._total_reset_count += len(env_ids)
        self._invalid_reset_count += raw_invalid_count
        self._degraded_reset_count += degraded_count
        self._degraded_target_count += degraded_target_count
        self._degraded_obstacle_count += degraded_obstacle_count

        used_mask = attempts_used > 0
        if used_mask.any():
            self._repair_attempts_total += attempts_used[used_mask].sum().item()
            self._repair_attempts_count += used_mask.sum().item()

        # ---- Invalid stats: EMA + rolling window ----
        invalid_fraction_batch = raw_invalid_count / max(1, len(env_ids))
        alpha = self.cfg.invalid_fraction_ema_alpha
        if alpha > 0.0:
            self._invalid_reset_ema = (1.0 - alpha) * self._invalid_reset_ema + alpha * invalid_fraction_batch

        invalid_window_fraction = 0.0
        window_size = self._invalid_reset_window.numel()
        if window_size > 0:
            num = len(env_ids)
            idxs = (torch.arange(num, device=self.device) + self._invalid_reset_window_idx) % window_size
            self._invalid_reset_window[idxs] = initial_invalid_mask
            self._invalid_reset_window_idx = (self._invalid_reset_window_idx + num) % window_size
            self._invalid_reset_window_filled = min(window_size, self._invalid_reset_window_filled + num)
            if self._invalid_reset_window_filled < window_size:
                invalid_window_fraction = (
                    self._invalid_reset_window[: self._invalid_reset_window_filled].float().mean().item()
                )
            else:
                invalid_window_fraction = self._invalid_reset_window.float().mean().item()

        # ---- Per-stage counters (curriculum) ----
        if self._reset_count_by_stage:
            stage_idx = self._curriculum_stage
            self._reset_count_by_stage[stage_idx] += len(env_ids)
            self._invalid_count_by_stage[stage_idx] += raw_invalid_count
            self._degraded_count_by_stage[stage_idx] += degraded_count
            for idx in range(len(self._reset_count_by_stage)):
                self.extras[f"Stats/reset_count_stage_{idx}"] = float(self._reset_count_by_stage[idx])
                self.extras[f"Stats/invalid_count_stage_{idx}"] = float(self._invalid_count_by_stage[idx])
                self.extras[f"Stats/degraded_count_stage_{idx}"] = float(self._degraded_count_by_stage[idx])

        self.extras["Stats/invalid_env_fraction"] = float(self._invalid_reset_ema)
        self.extras["Stats/invalid_env_fraction_window"] = float(invalid_window_fraction)
        self.extras["Stats/invalid_env_count"] = float(self._invalid_reset_count)
        self.extras["Stats/degraded_env_count"] = float(self._degraded_reset_count)
        self.extras["Stats/degraded_env_fraction"] = float(
            self._degraded_reset_count / max(1, self._total_reset_count)
        )
        self.extras["Stats/degraded_target_count"] = float(self._degraded_target_count)
        self.extras["Stats/degraded_obstacle_count"] = float(self._degraded_obstacle_count)
        self.extras["Stats/repair_attempts_used_avg"] = self._repair_attempts_total / max(
            1, self._repair_attempts_count
        )
        self.extras["Stats/reset_count"] = float(self._total_reset_count)

        # ---- Buffer resets for new episode ----
        self._joint_targets[env_ids] = joint_pos[:, self._joint_ids]
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_dist[env_ids] = 0.0
        self._has_prev_dist[env_ids] = False
        tcp_pos = self._robot.data.body_pos_w[env_ids, self._tcp_body_idx] - self.scene.env_origins[env_ids]
        self._best_dist[env_ids] = torch.norm(self._target_pos[env_ids] - tcp_pos, dim=-1)
        self._stuck_steps[env_ids] = 0

    def _apply_degraded_reset(self, env_ids: torch.Tensor) -> None:
        # Fallback: target near TCP and inactive obstacles to keep episode usable.
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        tcp_pos = self._robot.data.body_pos_w[env_ids, self._tcp_body_idx] - self.scene.env_origins[env_ids]
        offset = self._sample_uniform_pos(
            len(env_ids), ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        ) * self.cfg.invalid_target_fallback_radius
        target_pos = tcp_pos + offset
        target_pos = torch.max(torch.min(target_pos, self._curr_target_max), self._curr_target_min)
        self._target_pos[env_ids] = target_pos
        self._update_target_markers()

        inactive_pos = self._obstacle_inactive_pos
        obstacle_pos = inactive_pos.view(1, 1, 3).expand(len(env_ids), self.cfg.num_obstacles, 3).clone()
        self._obstacle_pos[env_ids] = obstacle_pos
        self._write_obstacle_positions(env_ids, obstacle_pos)

    def _reset_targets(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Target sampling with optional IK reachability check.
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        target_pos = torch.zeros((num_envs, 3), device=self.device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        degraded_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        target_min = self._curr_target_min
        target_max = self._curr_target_max
        if self.cfg.use_ik_reachability:
            ee_pos_w = self._robot.data.body_pos_w[env_ids, self._tcp_body_idx]
            jacobian_pos = self._get_tcp_jacobian(env_ids)[:, :3, :]
            joint_pos = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
            env_origins = self.scene.env_origins[env_ids]
            ee_pos_local = ee_pos_w - env_origins
        else:
            ee_pos_local = self._robot.data.body_pos_w[env_ids, self._tcp_body_idx] - self.scene.env_origins[env_ids]
        for i in range(num_envs):
            candidate = None
            is_valid = False
            for _ in range(self.cfg.target_resample_attempts):
                candidate = self._sample_uniform_pos(1, self._curr_target_pos_range)[0]
                if not self._is_target_valid(candidate):
                    continue
                if self.cfg.use_ik_reachability:
                    if not self._is_target_reachable(
                        candidate,
                        ee_pos_w[i : i + 1],
                        jacobian_pos[i : i + 1],
                        joint_pos[i : i + 1],
                        env_origins[i : i + 1],
                    ):
                        continue
                is_valid = True
                break
            if candidate is None or not is_valid:
                valid_mask[i] = True
                degraded_mask[i] = True
                center = ee_pos_local[i : i + 1]
                offset = self._sample_uniform_pos(1, ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))[0]
                offset = offset * self.cfg.invalid_target_fallback_radius
                candidate = center.squeeze(0) + offset
                candidate = torch.max(torch.min(candidate, target_max), target_min)
            target_pos[i] = candidate
        self._target_pos[env_ids] = target_pos
        self._update_target_markers()
        return valid_mask, degraded_mask

    def _reset_obstacles(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Obstacle sampling with distance constraints and optional LOS clearance.
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        obstacle_pos = torch.zeros((num_envs, self.cfg.num_obstacles, 3), device=self.device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        degraded_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        inactive_pos = self._obstacle_inactive_pos
        active_count = max(0, min(self._active_obstacle_count, self.cfg.num_obstacles))
        tcp_pos = self._robot.data.body_pos_w[env_ids, self._tcp_body_idx] - self.scene.env_origins[env_ids]
        for env_idx in range(num_envs):
            target_pos = self._target_pos[env_ids[env_idx]]
            env_valid = True
            for obs_idx in range(active_count):
                candidate = None
                for _ in range(self.cfg.obstacle_resample_attempts):
                    candidate = self._sample_uniform_pos(1, self._curr_obstacle_pos_range)[0]
                    existing = obstacle_pos[env_idx, :obs_idx]
                    if self._is_obstacle_valid(candidate, target_pos, existing):
                        break
                if candidate is None or not self._is_obstacle_valid(candidate, target_pos, obstacle_pos[env_idx, :obs_idx]):
                    env_valid = False
                    break
                obstacle_pos[env_idx, obs_idx] = candidate
            if env_valid and self.cfg.enable_los_clearance_check and active_count > 0:
                if not self._is_line_of_sight_clear(tcp_pos[env_idx], target_pos, obstacle_pos[env_idx, :active_count]):
                    env_valid = False
            if not env_valid:
                valid_mask[env_idx] = True
                degraded_mask[env_idx] = True
                obstacle_pos[env_idx, :active_count] = inactive_pos
            if active_count < self.cfg.num_obstacles:
                obstacle_pos[env_idx, active_count:] = inactive_pos
        self._obstacle_pos[env_ids] = obstacle_pos

        self._write_obstacle_positions(env_ids, obstacle_pos)
        return valid_mask, degraded_mask

    def _write_obstacle_positions(self, env_ids: torch.Tensor, obstacle_pos: torch.Tensor) -> None:
        # Write obstacle root poses and velocities in one place for reuse.
        for i, obstacle in enumerate(self._obstacles):
            root_state = obstacle.data.default_root_state[env_ids].clone()
            root_state[:, :3] = obstacle_pos[:, i, :] + self.scene.env_origins[env_ids]
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            root_state[:, 7:] = 0.0
            obstacle.write_root_pose_to_sim(root_state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

    def _update_target_markers(self):
        target_pos_w = self._target_pos + self.scene.env_origins
        self._target_markers.visualize(translations=target_pos_w)

    def _check_obstacle_collision(self) -> torch.Tensor:
        # Aggregate contact from obstacle contact sensors.
        if not self._obstacle_contact_sensors:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        contact_flags = []
        for sensor in self._obstacle_contact_sensors:
            if sensor.cfg.filter_prim_paths_expr:
                force_matrix = sensor.data.force_matrix_w
                contact = torch.norm(force_matrix, dim=-1) > self.cfg.collision_force_threshold
                contact_any = torch.any(contact, dim=(1, 2))
            else:
                net_forces = sensor.data.net_forces_w
                contact = torch.norm(net_forces, dim=-1) > self.cfg.collision_force_threshold
                contact_any = torch.any(contact, dim=1)
            contact_flags.append(contact_any)
        return torch.any(torch.stack(contact_flags, dim=0), dim=0)

    def _scale_joint_pos(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return 2.0 * (joint_pos - self._dof_lower_limits) / (self._dof_upper_limits - self._dof_lower_limits) - 1.0

    def _apply_curriculum_stage(self, stage_idx: int) -> None:
        # Update sampling ranges and tolerances when curriculum advances.
        stage = self._curriculum_cfg.stages[stage_idx]
        self._curriculum_stage = stage_idx
        self._curr_target_pos_range = stage.target_pos_range
        self._curr_success_tolerance = stage.success_tolerance
        if stage.obstacle_pos_range is None:
            self._curr_obstacle_pos_range = self.cfg.obstacle_pos_range
        else:
            self._curr_obstacle_pos_range = stage.obstacle_pos_range
        self._active_obstacle_count = max(0, min(stage.active_obstacles, self.cfg.num_obstacles))
        self._update_target_bounds_tensors()
        self.extras["Stats/curriculum_stage"] = float(self._curriculum_stage)
        self.extras["Stats/curriculum_success_tolerance"] = float(self._curr_success_tolerance)

    def _update_target_bounds_tensors(self) -> None:
        self._curr_target_min = torch.tensor(
            [
                self._curr_target_pos_range[0][0],
                self._curr_target_pos_range[1][0],
                self._curr_target_pos_range[2][0],
            ],
            device=self.device,
        )
        self._curr_target_max = torch.tensor(
            [
                self._curr_target_pos_range[0][1],
                self._curr_target_pos_range[1][1],
                self._curr_target_pos_range[2][1],
            ],
            device=self.device,
        )

    def _reset_success_history(self) -> None:
        self._episode_success_history.zero_()
        self._episode_history_idx = 0
        self._episode_history_filled = 0
        self._success_rate = 0.0

    def _update_success_history(self, env_ids: torch.Tensor) -> None:
        # Track success history for curriculum gating.
        if self._episode_success_history.numel() == 0:
            return
        valid_env_ids = env_ids[~self._invalid_reset_buf[env_ids]]
        num = len(valid_env_ids)
        if num == 0:
            return
        window_size = self._episode_success_history.numel()
        success_values = self._success_buf[valid_env_ids].float()
        idxs = (torch.arange(num, device=self.device) + self._episode_history_idx) % window_size
        self._episode_success_history[idxs] = success_values
        self._episode_history_idx = (self._episode_history_idx + num) % window_size
        self._episode_history_filled = min(window_size, self._episode_history_filled + num)
        if self._episode_history_filled > 0:
            if self._episode_history_filled < window_size:
                self._success_rate = (
                    self._episode_success_history[: self._episode_history_filled].mean().item()
                )
            else:
                self._success_rate = self._episode_success_history.mean().item()
        else:
            self._success_rate = 0.0
        self.extras["Stats/success_rate"] = float(self._success_rate)
        if (
            self._curriculum_enabled
            and self._episode_history_filled >= window_size
            and self._success_rate >= self._curriculum_cfg.min_success_rate
            and self._curriculum_stage < len(self._curriculum_cfg.stages) - 1
        ):
            self._apply_curriculum_stage(self._curriculum_stage + 1)
            self._reset_success_history()

    def _get_tcp_jacobian(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        jacobians = self._robot.root_physx_view.get_jacobians()
        jacobian = jacobians[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        if env_ids is None:
            return jacobian
        return jacobian[env_ids]

    def _compute_dls_delta(self, jacobian_pos: torch.Tensor, delta_pos: torch.Tensor) -> torch.Tensor:
        jacobian_t = torch.transpose(jacobian_pos, 1, 2)
        lambda_val = self.cfg.ik_reachability_dls_lambda
        eye = torch.eye(jacobian_pos.shape[1], device=self.device).unsqueeze(0).expand(
            jacobian_pos.shape[0], -1, -1
        )
        mat = jacobian_pos @ jacobian_t + (lambda_val**2) * eye
        delta_q = jacobian_t @ torch.linalg.solve(mat, delta_pos.unsqueeze(-1))
        return delta_q.squeeze(-1)

    def _is_target_reachable(
        self,
        target_pos: torch.Tensor,
        ee_pos_w: torch.Tensor,
        jacobian_pos: torch.Tensor,
        joint_pos: torch.Tensor,
        env_origin: torch.Tensor,
    ) -> bool:
        if not self.cfg.use_ik_reachability:
            return True
        target_pos_w = target_pos + env_origin
        current_pos = ee_pos_w.clone()
        current_joint_pos = joint_pos.clone()
        for _ in range(self.cfg.ik_reachability_iters):
            delta_pos = target_pos_w - current_pos
            if torch.norm(delta_pos, dim=-1).item() <= self.cfg.ik_reachability_tolerance:
                return True
            delta_q = self._compute_dls_delta(jacobian_pos, delta_pos)
            max_delta = self.cfg.ik_reachability_max_delta
            delta_q = torch.clamp(delta_q, -max_delta, max_delta)
            new_joint_pos = torch.clamp(current_joint_pos + delta_q, self._dof_lower_limits, self._dof_upper_limits)
            delta_q = new_joint_pos - current_joint_pos
            current_joint_pos = new_joint_pos
            current_pos = current_pos + torch.bmm(jacobian_pos, delta_q.unsqueeze(-1)).squeeze(-1)
        return False

    def _sample_uniform_pos(self, num: int, pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]):
        pos = torch.zeros((num, 3), device=self.device)
        pos[:, 0] = sample_uniform(pos_range[0][0], pos_range[0][1], (num,), device=self.device)
        pos[:, 1] = sample_uniform(pos_range[1][0], pos_range[1][1], (num,), device=self.device)
        pos[:, 2] = sample_uniform(pos_range[2][0], pos_range[2][1], (num,), device=self.device)
        return pos

    def _is_target_valid(self, target_pos: torch.Tensor) -> bool:
        dist_base = torch.norm(target_pos - self._base_pos_local)
        if dist_base < self.cfg.target_min_distance_from_base:
            return False
        if dist_base > self.cfg.target_max_distance_from_base:
            return False
        return True

    def _is_line_of_sight_clear(
        self, tcp_pos: torch.Tensor, target_pos: torch.Tensor, obstacle_pos: torch.Tensor
    ) -> bool:
        # Segment-to-sphere distance test for cheap clearance checks.
        if obstacle_pos.numel() == 0:
            return True
        direction = target_pos - tcp_pos
        dir_norm_sq = torch.sum(direction * direction)
        if dir_norm_sq.item() <= 1e-6:
            return True
        to_obs = obstacle_pos - tcp_pos
        t = torch.sum(to_obs * direction, dim=-1) / dir_norm_sq
        t = torch.clamp(t, 0.0, 1.0)
        closest = tcp_pos + t.unsqueeze(-1) * direction
        dist = torch.norm(obstacle_pos - closest, dim=-1)
        return torch.all(dist > self._obstacle_clearance_radius).item()

    def _is_obstacle_valid(
        self, obstacle_pos: torch.Tensor, target_pos: torch.Tensor, existing_obstacles: torch.Tensor
    ) -> bool:
        dist_base = torch.norm(obstacle_pos - self._base_pos_local)
        if dist_base < self.cfg.obstacle_min_distance_from_base:
            return False
        dist_target = torch.norm(obstacle_pos - target_pos)
        if dist_target < self.cfg.obstacle_min_distance_from_target:
            return False
        if existing_obstacles.numel() > 0:
            dist_obs = torch.norm(existing_obstacles - obstacle_pos, dim=-1)
            if torch.any(dist_obs < self.cfg.obstacle_min_distance_between):
                return False
        return True
