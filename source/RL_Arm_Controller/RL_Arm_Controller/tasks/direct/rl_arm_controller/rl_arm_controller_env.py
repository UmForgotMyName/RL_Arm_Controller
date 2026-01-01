# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import normalize, quat_apply, quat_mul, sample_uniform

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
        self._exclude_from_curriculum_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_once_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stuck_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_stats_initialized = False
        self._invalid_reset_count = 0
        self._total_reset_count = 0
        self._degraded_reset_count = 0
        self._degraded_target_count = 0
        self._degraded_obstacle_count = 0
        self._exclude_from_curriculum_count = 0
        self._target_sample_invalid_distance_total = 0
        self._target_sample_unreachable_total = 0
        self._target_sample_success_total = 0
        self._target_sample_attempts_total = 0
        self._target_sample_attempts_count = 0
        self._target_sample_invalid_distance_batch = 0
        self._target_sample_unreachable_batch = 0
        self._target_sample_success_batch = 0
        self._target_sample_attempts_batch = 0
        self._target_sample_attempts_batch_count = 0
        self._repair_attempts_total = 0
        self._repair_attempts_count = 0
        self._invalid_reset_ema = 0.0
        self._invalid_reset_window = torch.zeros(self.cfg.invalid_fraction_window, dtype=torch.bool, device=self.device)
        self._invalid_reset_window_idx = 0
        self._invalid_reset_window_filled = 0
        self._degraded_target_window = torch.zeros(self.cfg.invalid_fraction_window, dtype=torch.bool, device=self.device)
        self._degraded_obstacle_window = torch.zeros(self.cfg.invalid_fraction_window, dtype=torch.bool, device=self.device)
        self._exclude_from_curriculum_window = torch.zeros(self.cfg.invalid_fraction_window, dtype=torch.bool, device=self.device)
        self._degraded_target_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._degraded_obstacle_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._best_dist = torch.zeros(self.num_envs, device=self.device)
        self._stuck_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._stuck_terminated_count = 0

        # ---- Geometry helpers ----
        self._tcp_forward_axis = torch.tensor(self.cfg.tcp_forward_axis, device=self.device).repeat(self.num_envs, 1)
        self._tcp_offset_pos = torch.tensor(self.cfg.tcp_offset_pos, device=self.device).repeat(self.num_envs, 1)
        self._tcp_offset_rot = torch.tensor(self.cfg.tcp_offset_rot, device=self.device).repeat(self.num_envs, 1)
        self._base_pos_local = torch.tensor(self.cfg.robot_cfg.init_state.pos, device=self.device)
        obstacle_diag = torch.tensor(self.cfg.obstacle_size, device=self.device)
        self._obstacle_half_diag = 0.5 * torch.linalg.norm(obstacle_diag)
        self._curr_los_clearance_margin = self.cfg.los_clearance_margin
        self._obstacle_clearance_radius = self._obstacle_half_diag + self._curr_los_clearance_margin
        self._obstacle_inactive_pos = torch.tensor(self.cfg.obstacle_inactive_pos, device=self.device)

        # ---- Visualization ----
        self._target_markers = VisualizationMarkers(self.cfg.target_marker_cfg)
        # ---- Curriculum ----
        self._curriculum_cfg = self.cfg.curriculum
        self._curriculum_enabled = self._curriculum_cfg is not None and self._curriculum_cfg.enabled and len(self._curriculum_cfg.stages) > 0
        self._curriculum_stage = 0
        self._curriculum_min_success_rate = self._curriculum_cfg.min_success_rate if self._curriculum_cfg is not None else 0.0
        self._curriculum_window_size = self._curriculum_cfg.window_size if self._curriculum_cfg is not None else 0
        self._curr_use_ik_reachability = self.cfg.use_ik_reachability
        self._curr_target_pos_range = self.cfg.target_pos_range
        self._curr_obstacle_pos_range = self.cfg.obstacle_pos_range
        self._curr_success_tolerance = self.cfg.success_tolerance
        self._active_obstacle_count = self.cfg.num_obstacles
        self._curr_enable_los_check = self.cfg.enable_los_clearance_check
        self._curr_force_path_obstacle = self.cfg.force_path_obstacle
        if self._curriculum_enabled:
            self._apply_curriculum_stage(0)
        else:
            self._update_target_bounds_tensors()
            self._update_obstacle_bounds_tensors()
        self._reset_success_history(self._curriculum_window_size)
        stage_count = len(self._curriculum_cfg.stages) if self._curriculum_cfg is not None else 0
        self._reset_count_by_stage = [0 for _ in range(stage_count)]
        self._invalid_count_by_stage = [0 for _ in range(stage_count)]
        self._degraded_count_by_stage = [0 for _ in range(stage_count)]
        self._exclude_count_by_stage = [0 for _ in range(stage_count)]
        self._success_count_by_stage = [0 for _ in range(stage_count)]
        self._timeout_count_by_stage = [0 for _ in range(stage_count)]
        self._stuck_count_by_stage = [0 for _ in range(stage_count)]
        self._collision_count_by_stage = [0 for _ in range(stage_count)]
        self._last_total_reset_count = 0
        self._last_invalid_reset_count = 0
        self._last_degraded_reset_count = 0
        self._last_degraded_target_count = 0
        self._last_degraded_obstacle_count = 0
        self._last_exclude_from_curriculum_count = 0
        self._last_target_sample_invalid_distance_total = 0
        self._last_target_sample_unreachable_total = 0
        self._last_target_sample_success_total = 0
        self._last_target_sample_attempts_total = 0
        self._last_reset_count_by_stage = [0 for _ in range(stage_count)]
        self._last_invalid_count_by_stage = [0 for _ in range(stage_count)]
        self._last_degraded_count_by_stage = [0 for _ in range(stage_count)]
        self._last_exclude_count_by_stage = [0 for _ in range(stage_count)]
        self._last_success_count_by_stage = [0 for _ in range(stage_count)]
        self._last_timeout_count_by_stage = [0 for _ in range(stage_count)]
        self._last_stuck_count_by_stage = [0 for _ in range(stage_count)]
        self._last_collision_count_by_stage = [0 for _ in range(stage_count)]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # ---- Obstacles and contact sensors ----
        self._obstacles: list[RigidObject] = []
        self._robot_contact_sensor = None
        for i in range(self.cfg.num_obstacles):
            obstacle_cfg = self.cfg.obstacle_cfg.replace(prim_path=f"/World/envs/env_.*/Obstacle_{i}")
            obstacle = RigidObject(obstacle_cfg)
            self._obstacles.append(obstacle)
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle

        self._robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor)
        self.scene.sensors["robot_contact_sensor"] = self._robot_contact_sensor

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
        # Interpret actions as delta joint-target commands.
        self._joint_targets = self._joint_targets + self.cfg.action_scale * self._actions
        self._joint_targets = torch.clamp(self._joint_targets, self._dof_lower_limits, self._dof_upper_limits)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        # ---- State terms ----
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids] * self.cfg.dof_velocity_scale

        tcp_pos, _ = self._get_tcp_pose()
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
        tcp_pos, tcp_quat = self._get_tcp_pose()
        to_target = self._target_pos - tcp_pos
        dist = torch.norm(to_target, dim=-1)
        self.extras["Stats/mean_dist_to_target"] = float(dist.mean().item())
        self.extras["Stats/min_dist_to_target"] = float(dist.min().item())

        reward = -self.cfg.rew_scale_dist * dist
        success = dist < self._curr_success_tolerance
        if self.cfg.rew_scale_success_time != 0.0:
            time_bonus = 1.0 - (self.episode_length_buf.float() / self.max_episode_length)
            success_reward = self.cfg.rew_scale_success + self.cfg.rew_scale_success_time * time_bonus
        else:
            success_reward = self.cfg.rew_scale_success
        reward += success.float() * success_reward

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
            tcp_forward = quat_apply(tcp_quat, self._tcp_forward_axis)
            to_target_dir = normalize(to_target)
            reward += self.cfg.rew_scale_approach * torch.sum(tcp_forward * to_target_dir, dim=-1)

        # ---- Proximity shaping to encourage detours around obstacles ----
        if self.cfg.rew_scale_proximity != 0.0:
            active_count = max(0, min(self._active_obstacle_count, self.cfg.num_obstacles))
            if active_count > 0:
                obstacle_pos = self._obstacle_pos[:, :active_count, :]
                obs_dist = torch.norm(obstacle_pos - tcp_pos.unsqueeze(1), dim=-1)
                min_dist = obs_dist.min(dim=1).values
                proximity_term = min_dist - self.cfg.proximity_radius
                reward += self.cfg.rew_scale_proximity * torch.where(
                    min_dist < self.cfg.proximity_radius, proximity_term, torch.zeros_like(min_dist)
                )

        # ---- Collision penalty ----
        if self.cfg.rew_scale_collision != 0.0:
            collision = self._check_obstacle_collision()
            self._collision_buf.copy_(collision)
            if self.cfg.collision_penalty_once:
                new_collision = collision & ~self._collision_once_buf
                self._collision_once_buf |= collision
                reward = torch.where(new_collision, reward + self.cfg.rew_scale_collision, reward)
            else:
                reward = torch.where(collision, reward + self.cfg.rew_scale_collision, reward)
        elif not self.cfg.terminate_on_collision:
            self._collision_buf.zero_()
        reward = torch.where(self._invalid_reset_buf, torch.zeros_like(reward), reward)

        self._prev_dist = dist
        self._has_prev_dist[:] = True

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- Termination conditions ----
        tcp_pos, _ = self._get_tcp_pose()
        dist = torch.norm(self._target_pos - tcp_pos, dim=-1)
        success = dist < self._curr_success_tolerance
        self._success_buf = success

        terminated = success
        if self.cfg.terminate_on_collision:
            collision = self._check_obstacle_collision()
            self._collision_buf.copy_(collision)
            terminated = terminated | collision
        elif self.cfg.rew_scale_collision == 0.0:
            self._collision_buf.zero_()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        stuck = torch.zeros_like(time_out)
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
        if self.cfg.enable_stuck_termination:
            self._stuck_buf.copy_(stuck)
        else:
            self._stuck_buf.zero_()
        self._time_out_buf.copy_(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # ---- Reset pipeline ----
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        has_prev_episode = self._episode_stats_initialized
        if has_prev_episode:
            self._update_success_history(env_ids)
        else:
            self._episode_stats_initialized = True
        prev_invalid_mask = self._invalid_reset_buf[env_ids].clone()
        prev_exclude_mask = self._exclude_from_curriculum_buf[env_ids].clone()
        self._target_sample_invalid_distance_batch = 0
        self._target_sample_unreachable_batch = 0
        self._target_sample_success_batch = 0
        self._target_sample_attempts_batch = 0
        self._target_sample_attempts_batch_count = 0

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        noise = sample_uniform(-1.0, 1.0, (len(env_ids), len(self._joint_ids)), device=self.device)
        joint_pos[:, self._joint_ids] = self._default_joint_pos[env_ids] + noise * self.cfg.reset_joint_pos_noise
        joint_pos[:, self._joint_ids] = torch.clamp(joint_pos[:, self._joint_ids], self._dof_lower_limits, self._dof_upper_limits)
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
        exclude_mask = torch.zeros_like(invalid_mask)

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
                next_pending = pending_mask.clone()
                next_pending[pending_mask] = invalid_r
                pending_mask = next_pending

        # ---- Hard fallback: last resort to keep episodes usable ----
        if pending_mask.any():
            pending_env_ids = env_ids[pending_mask]
            self._apply_degraded_reset(pending_env_ids)
            invalid_mask[pending_mask] = False
            target_degraded[pending_mask] = True
            obstacle_degraded[pending_mask] = True
            degraded_mask[pending_mask] = True
            exclude_mask[pending_mask] = True

        self._invalid_reset_buf[env_ids] = invalid_mask
        self._degraded_target_buf[env_ids] = target_degraded
        self._degraded_obstacle_buf[env_ids] = obstacle_degraded
        self._exclude_from_curriculum_buf[env_ids] = exclude_mask

        raw_invalid_count = initial_invalid_mask.sum().item()
        degraded_count = degraded_mask.sum().item()
        degraded_target_count = target_degraded.sum().item()
        degraded_obstacle_count = obstacle_degraded.sum().item()
        exclude_count = exclude_mask.sum().item()

        self._total_reset_count += len(env_ids)
        self._invalid_reset_count += raw_invalid_count
        self._degraded_reset_count += degraded_count
        self._degraded_target_count += degraded_target_count
        self._degraded_obstacle_count += degraded_obstacle_count
        self._exclude_from_curriculum_count += exclude_count

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
        degraded_target_window_fraction = 0.0
        degraded_obstacle_window_fraction = 0.0
        exclude_window_fraction = 0.0
        window_size = self._invalid_reset_window.numel()
        if window_size > 0:
            num = len(env_ids)
            idxs = (torch.arange(num, device=self.device) + self._invalid_reset_window_idx) % window_size
            self._invalid_reset_window[idxs] = initial_invalid_mask
            self._degraded_target_window[idxs] = target_degraded
            self._degraded_obstacle_window[idxs] = obstacle_degraded
            self._exclude_from_curriculum_window[idxs] = exclude_mask
            self._invalid_reset_window_idx = (self._invalid_reset_window_idx + num) % window_size
            self._invalid_reset_window_filled = min(window_size, self._invalid_reset_window_filled + num)
            filled = self._invalid_reset_window_filled
            if filled < window_size:
                invalid_window_fraction = self._invalid_reset_window[:filled].float().mean().item()
                degraded_target_window_fraction = self._degraded_target_window[:filled].float().mean().item()
                degraded_obstacle_window_fraction = self._degraded_obstacle_window[:filled].float().mean().item()
                exclude_window_fraction = self._exclude_from_curriculum_window[:filled].float().mean().item()
            else:
                invalid_window_fraction = self._invalid_reset_window.float().mean().item()
                degraded_target_window_fraction = self._degraded_target_window.float().mean().item()
                degraded_obstacle_window_fraction = self._degraded_obstacle_window.float().mean().item()
                exclude_window_fraction = self._exclude_from_curriculum_window.float().mean().item()

        # ---- Per-stage counters (curriculum) ----
        if self._reset_count_by_stage:
            stage_idx = self._curriculum_stage
            self._reset_count_by_stage[stage_idx] += len(env_ids)
            self._invalid_count_by_stage[stage_idx] += raw_invalid_count
            self._degraded_count_by_stage[stage_idx] += degraded_count
            self._exclude_count_by_stage[stage_idx] += exclude_count
            if has_prev_episode:
                prev_valid_mask = ~(prev_invalid_mask | prev_exclude_mask)
                self._success_count_by_stage[stage_idx] += self._success_buf[env_ids][prev_valid_mask].sum().item()
                self._timeout_count_by_stage[stage_idx] += self._time_out_buf[env_ids].sum().item()
                self._stuck_count_by_stage[stage_idx] += self._stuck_buf[env_ids].sum().item()
                self._collision_count_by_stage[stage_idx] += self._collision_buf[env_ids].sum().item()
        # ---- Counter deltas ----
        reset_count_delta = self._total_reset_count - self._last_total_reset_count
        invalid_count_delta = self._invalid_reset_count - self._last_invalid_reset_count
        degraded_count_delta = self._degraded_reset_count - self._last_degraded_reset_count
        degraded_target_delta = self._degraded_target_count - self._last_degraded_target_count
        degraded_obstacle_delta = self._degraded_obstacle_count - self._last_degraded_obstacle_count
        exclude_count_delta = self._exclude_from_curriculum_count - self._last_exclude_from_curriculum_count
        target_invalid_distance_delta = (
            self._target_sample_invalid_distance_total - self._last_target_sample_invalid_distance_total
        )
        target_unreachable_delta = self._target_sample_unreachable_total - self._last_target_sample_unreachable_total
        target_success_delta = self._target_sample_success_total - self._last_target_sample_success_total
        target_attempts_delta = self._target_sample_attempts_total - self._last_target_sample_attempts_total

        self._last_total_reset_count = self._total_reset_count
        self._last_invalid_reset_count = self._invalid_reset_count
        self._last_degraded_reset_count = self._degraded_reset_count
        self._last_degraded_target_count = self._degraded_target_count
        self._last_degraded_obstacle_count = self._degraded_obstacle_count
        self._last_exclude_from_curriculum_count = self._exclude_from_curriculum_count
        self._last_target_sample_invalid_distance_total = self._target_sample_invalid_distance_total
        self._last_target_sample_unreachable_total = self._target_sample_unreachable_total
        self._last_target_sample_success_total = self._target_sample_success_total
        self._last_target_sample_attempts_total = self._target_sample_attempts_total

        target_attempts_avg = self._target_sample_attempts_batch / max(1, self._target_sample_attempts_batch_count)

        # ---- Rolling fractions and rates ----
        self.extras["Stats/success_rate_valid"] = float(self._success_rate)
        self.extras["Stats/success_rate_all"] = float(self._success_rate_all)
        self.extras["Stats/invalid_env_fraction"] = float(self._invalid_reset_ema)
        self.extras["Stats/invalid_env_fraction_window"] = float(invalid_window_fraction)
        self.extras["Stats/degraded_target_fraction_recent"] = float(degraded_target_window_fraction)
        self.extras["Stats/degraded_obstacle_fraction_recent"] = float(degraded_obstacle_window_fraction)
        self.extras["Stats/exclude_from_curriculum_fraction_recent"] = float(exclude_window_fraction)
        self.extras["Stats/degraded_env_fraction"] = float(self._degraded_reset_count / max(1, self._total_reset_count))
        self.extras["Stats/repair_attempts_used_avg"] = self._repair_attempts_total / max(1, self._repair_attempts_count)
        self.extras["Stats/target_sample_attempts_avg"] = float(target_attempts_avg)

        # ---- Totals ----
        self.extras["StatsTotals/reset_count"] = float(self._total_reset_count)
        self.extras["StatsTotals/invalid_env_count"] = float(self._invalid_reset_count)
        self.extras["StatsTotals/degraded_env_count"] = float(self._degraded_reset_count)
        self.extras["StatsTotals/degraded_target_count"] = float(self._degraded_target_count)
        self.extras["StatsTotals/degraded_obstacle_count"] = float(self._degraded_obstacle_count)
        self.extras["StatsTotals/exclude_from_curriculum_count"] = float(self._exclude_from_curriculum_count)
        self.extras["StatsTotals/target_sample_invalid_distance"] = float(self._target_sample_invalid_distance_total)
        self.extras["StatsTotals/target_sample_unreachable"] = float(self._target_sample_unreachable_total)
        self.extras["StatsTotals/target_sample_success"] = float(self._target_sample_success_total)
        self.extras["StatsTotals/target_sample_attempts"] = float(self._target_sample_attempts_total)

        # ---- Deltas ----
        self.extras["StatsDelta/reset_count"] = float(reset_count_delta)
        self.extras["StatsDelta/invalid_env_count"] = float(invalid_count_delta)
        self.extras["StatsDelta/degraded_env_count"] = float(degraded_count_delta)
        self.extras["StatsDelta/degraded_target_count"] = float(degraded_target_delta)
        self.extras["StatsDelta/degraded_obstacle_count"] = float(degraded_obstacle_delta)
        self.extras["StatsDelta/exclude_from_curriculum_count"] = float(exclude_count_delta)
        self.extras["StatsDelta/target_sample_invalid_distance"] = float(target_invalid_distance_delta)
        self.extras["StatsDelta/target_sample_unreachable"] = float(target_unreachable_delta)
        self.extras["StatsDelta/target_sample_success"] = float(target_success_delta)
        self.extras["StatsDelta/target_sample_attempts"] = float(target_attempts_delta)

        # ---- Per-stage totals/deltas ----
        stage_reset_deltas = []
        stage_invalid_deltas = []
        stage_degraded_deltas = []
        stage_exclude_deltas = []
        stage_success_deltas = []
        stage_timeout_deltas = []
        stage_stuck_deltas = []
        stage_collision_deltas = []
        for idx in range(len(self._reset_count_by_stage)):
            reset_delta = self._reset_count_by_stage[idx] - self._last_reset_count_by_stage[idx]
            invalid_delta = self._invalid_count_by_stage[idx] - self._last_invalid_count_by_stage[idx]
            degraded_delta = self._degraded_count_by_stage[idx] - self._last_degraded_count_by_stage[idx]
            exclude_delta = self._exclude_count_by_stage[idx] - self._last_exclude_count_by_stage[idx]
            success_delta = self._success_count_by_stage[idx] - self._last_success_count_by_stage[idx]
            timeout_delta = self._timeout_count_by_stage[idx] - self._last_timeout_count_by_stage[idx]
            stuck_delta = self._stuck_count_by_stage[idx] - self._last_stuck_count_by_stage[idx]
            collision_delta = self._collision_count_by_stage[idx] - self._last_collision_count_by_stage[idx]

            stage_reset_deltas.append(reset_delta)
            stage_invalid_deltas.append(invalid_delta)
            stage_degraded_deltas.append(degraded_delta)
            stage_exclude_deltas.append(exclude_delta)
            stage_success_deltas.append(success_delta)
            stage_timeout_deltas.append(timeout_delta)
            stage_stuck_deltas.append(stuck_delta)
            stage_collision_deltas.append(collision_delta)

            self._last_reset_count_by_stage[idx] = self._reset_count_by_stage[idx]
            self._last_invalid_count_by_stage[idx] = self._invalid_count_by_stage[idx]
            self._last_degraded_count_by_stage[idx] = self._degraded_count_by_stage[idx]
            self._last_exclude_count_by_stage[idx] = self._exclude_count_by_stage[idx]
            self._last_success_count_by_stage[idx] = self._success_count_by_stage[idx]
            self._last_timeout_count_by_stage[idx] = self._timeout_count_by_stage[idx]
            self._last_stuck_count_by_stage[idx] = self._stuck_count_by_stage[idx]
            self._last_collision_count_by_stage[idx] = self._collision_count_by_stage[idx]

            self.extras[f"StatsTotals/stage_{idx}_reset_count"] = float(self._reset_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_invalid_count"] = float(self._invalid_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_degraded_count"] = float(self._degraded_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_exclude_count"] = float(self._exclude_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_success_count"] = float(self._success_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_timeout_count"] = float(self._timeout_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_stuck_count"] = float(self._stuck_count_by_stage[idx])
            self.extras[f"StatsTotals/stage_{idx}_collision_count"] = float(self._collision_count_by_stage[idx])

            self.extras[f"StatsDelta/stage_{idx}_reset_count"] = float(reset_delta)
            self.extras[f"StatsDelta/stage_{idx}_invalid_count"] = float(invalid_delta)
            self.extras[f"StatsDelta/stage_{idx}_degraded_count"] = float(degraded_delta)
            self.extras[f"StatsDelta/stage_{idx}_exclude_count"] = float(exclude_delta)
            self.extras[f"StatsDelta/stage_{idx}_success_count"] = float(success_delta)
            self.extras[f"StatsDelta/stage_{idx}_timeout_count"] = float(timeout_delta)
            self.extras[f"StatsDelta/stage_{idx}_stuck_count"] = float(stuck_delta)
            self.extras[f"StatsDelta/stage_{idx}_collision_count"] = float(collision_delta)

        log = self.extras.setdefault("log", {})
        log["curriculum_stage"] = float(self._curriculum_stage)
        log["curriculum_success_tolerance"] = float(self._curr_success_tolerance)
        log["curriculum_min_success_rate"] = float(self._curriculum_min_success_rate)
        log["curriculum_window_size"] = float(self._curriculum_window_size)
        log["curriculum_use_ik_reachability"] = float(self._curr_use_ik_reachability)
        log["success_rate_valid"] = float(self._success_rate)
        log["success_rate_all"] = float(self._success_rate_all)
        log["success_rate"] = float(self._success_rate)
        log["invalid_env_fraction"] = float(self._invalid_reset_ema)
        log["invalid_env_fraction_window"] = float(invalid_window_fraction)
        log["degraded_target_fraction_recent"] = float(degraded_target_window_fraction)
        log["degraded_obstacle_fraction_recent"] = float(degraded_obstacle_window_fraction)
        log["exclude_from_curriculum_fraction_recent"] = float(exclude_window_fraction)
        log["degraded_env_fraction"] = float(self._degraded_reset_count / max(1, self._total_reset_count))
        log["repair_attempts_used_avg"] = self._repair_attempts_total / max(1, self._repair_attempts_count)
        log["target_sample_attempts_avg"] = float(target_attempts_avg)
        log["Totals/reset_count"] = float(self._total_reset_count)
        log["Totals/invalid_env_count"] = float(self._invalid_reset_count)
        log["Totals/degraded_env_count"] = float(self._degraded_reset_count)
        log["Totals/degraded_target_count"] = float(self._degraded_target_count)
        log["Totals/degraded_obstacle_count"] = float(self._degraded_obstacle_count)
        log["Totals/exclude_from_curriculum_count"] = float(self._exclude_from_curriculum_count)
        log["Totals/target_sample_invalid_distance"] = float(self._target_sample_invalid_distance_total)
        log["Totals/target_sample_unreachable"] = float(self._target_sample_unreachable_total)
        log["Totals/target_sample_success"] = float(self._target_sample_success_total)
        log["Totals/target_sample_attempts"] = float(self._target_sample_attempts_total)
        log["Delta/reset_count"] = float(reset_count_delta)
        log["Delta/invalid_env_count"] = float(invalid_count_delta)
        log["Delta/degraded_env_count"] = float(degraded_count_delta)
        log["Delta/degraded_target_count"] = float(degraded_target_delta)
        log["Delta/degraded_obstacle_count"] = float(degraded_obstacle_delta)
        log["Delta/exclude_from_curriculum_count"] = float(exclude_count_delta)
        log["Delta/target_sample_invalid_distance"] = float(target_invalid_distance_delta)
        log["Delta/target_sample_unreachable"] = float(target_unreachable_delta)
        log["Delta/target_sample_success"] = float(target_success_delta)
        log["Delta/target_sample_attempts"] = float(target_attempts_delta)
        for idx in range(len(self._reset_count_by_stage)):
            log[f"Totals/stage_{idx}_reset_count"] = float(self._reset_count_by_stage[idx])
            log[f"Totals/stage_{idx}_invalid_count"] = float(self._invalid_count_by_stage[idx])
            log[f"Totals/stage_{idx}_degraded_count"] = float(self._degraded_count_by_stage[idx])
            log[f"Totals/stage_{idx}_exclude_count"] = float(self._exclude_count_by_stage[idx])
            log[f"Totals/stage_{idx}_success_count"] = float(self._success_count_by_stage[idx])
            log[f"Totals/stage_{idx}_timeout_count"] = float(self._timeout_count_by_stage[idx])
            log[f"Totals/stage_{idx}_stuck_count"] = float(self._stuck_count_by_stage[idx])
            log[f"Totals/stage_{idx}_collision_count"] = float(self._collision_count_by_stage[idx])
            log[f"Delta/stage_{idx}_reset_count"] = float(stage_reset_deltas[idx])
            log[f"Delta/stage_{idx}_invalid_count"] = float(stage_invalid_deltas[idx])
            log[f"Delta/stage_{idx}_degraded_count"] = float(stage_degraded_deltas[idx])
            log[f"Delta/stage_{idx}_exclude_count"] = float(stage_exclude_deltas[idx])
            log[f"Delta/stage_{idx}_success_count"] = float(stage_success_deltas[idx])
            log[f"Delta/stage_{idx}_timeout_count"] = float(stage_timeout_deltas[idx])
            log[f"Delta/stage_{idx}_stuck_count"] = float(stage_stuck_deltas[idx])
            log[f"Delta/stage_{idx}_collision_count"] = float(stage_collision_deltas[idx])

        # ---- Buffer resets for new episode ----
        self._joint_targets[env_ids] = joint_pos[:, self._joint_ids]
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_dist[env_ids] = 0.0
        self._has_prev_dist[env_ids] = False
        self._collision_once_buf[env_ids] = False
        tcp_pos, _ = self._get_tcp_pose(env_ids)
        self._best_dist[env_ids] = torch.norm(self._target_pos[env_ids] - tcp_pos, dim=-1)
        self._stuck_steps[env_ids] = 0

    def _apply_degraded_reset(self, env_ids: torch.Tensor) -> None:
        # Fallback: target near TCP and inactive obstacles to keep episode usable.
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        tcp_pos, _ = self._get_tcp_pose(env_ids)
        offset = self._sample_uniform_pos(len(env_ids), ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))) * self.cfg.invalid_target_fallback_radius
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
        target_pos = self._target_pos[env_ids].clone()
        valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        degraded_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        if self._curr_use_ik_reachability:
            ee_pos_w, _ = self._get_tcp_pose_w(env_ids)
            jacobian_pos = self._get_tcp_jacobian_pos(env_ids)
            joint_pos = self._robot.data.joint_pos[env_ids][:, self._joint_ids]
            env_origins = self.scene.env_origins[env_ids]
        for i in range(num_envs):
            attempts_used = 0
            for _ in range(self.cfg.target_resample_attempts):
                attempts_used += 1
                candidate = self._sample_uniform_pos(1, self._curr_target_pos_range)[0]
                if not self._is_target_valid(candidate):
                    self._target_sample_invalid_distance_batch += 1
                    self._target_sample_invalid_distance_total += 1
                    continue
                if self._curr_use_ik_reachability:
                    if not self._is_target_reachable(
                        candidate,
                        ee_pos_w[i : i + 1],
                        jacobian_pos[i : i + 1],
                        joint_pos[i : i + 1],
                        env_origins[i : i + 1],
                    ):
                        self._target_sample_unreachable_batch += 1
                        self._target_sample_unreachable_total += 1
                        continue
                valid_mask[i] = True
                target_pos[i] = candidate
                self._target_sample_success_batch += 1
                self._target_sample_success_total += 1
                self._target_sample_attempts_batch += attempts_used
                self._target_sample_attempts_batch_count += 1
                self._target_sample_attempts_total += attempts_used
                self._target_sample_attempts_count += 1
                break
        self._target_pos[env_ids] = target_pos
        self._update_target_markers()
        return valid_mask, degraded_mask

    def _reset_obstacles(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Obstacle sampling with distance constraints and optional LOS clearance.
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        obstacle_pos = self._obstacle_pos[env_ids].clone()
        valid_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        degraded_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        inactive_pos = self._obstacle_inactive_pos
        active_count = max(0, min(self._active_obstacle_count, self.cfg.num_obstacles))
        tcp_pos, _ = self._get_tcp_pose(env_ids)
        for env_idx in range(num_envs):
            target_pos = self._target_pos[env_ids[env_idx]]
            env_valid = True
            env_obstacles = obstacle_pos[env_idx].clone()
            for obs_idx in range(active_count):
                candidate = None
                for _ in range(self.cfg.obstacle_resample_attempts):
                    if self._curr_force_path_obstacle and obs_idx == 0:
                        candidate = self._sample_path_obstacle(tcp_pos[env_idx], target_pos)
                    else:
                        candidate = self._sample_uniform_pos(1, self._curr_obstacle_pos_range)[0]
                    if candidate is None:
                        continue
                    existing = env_obstacles[:obs_idx]
                    if self._is_obstacle_valid(candidate, target_pos, existing):
                        break
                if candidate is None or not self._is_obstacle_valid(candidate, target_pos, env_obstacles[:obs_idx]):
                    env_valid = False
                    break
                env_obstacles[obs_idx] = candidate
            if env_valid and self._curr_enable_los_check and active_count > 0:
                if not self._is_line_of_sight_clear(tcp_pos[env_idx], target_pos, env_obstacles[:active_count]):
                    env_valid = False
            if env_valid:
                valid_mask[env_idx] = True
                if active_count < self.cfg.num_obstacles:
                    env_obstacles[active_count:] = inactive_pos
                obstacle_pos[env_idx] = env_obstacles
        self._obstacle_pos[env_ids] = obstacle_pos

        self._write_obstacle_positions(env_ids, obstacle_pos)
        return valid_mask, degraded_mask

    def _sample_path_obstacle(self, tcp_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor | None:
        direction = target_pos - tcp_pos
        seg_len = torch.norm(direction)
        if seg_len.item() <= 1e-6:
            return None
        direction_unit = direction / seg_len
        t_min, t_max = self.cfg.path_obstacle_t_range
        t = sample_uniform(t_min, t_max, (1,), device=self.device)
        base_point = tcp_pos + direction * t
        rand_vec = self._sample_uniform_pos(1, ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))[0]
        perp = rand_vec - torch.dot(rand_vec, direction_unit) * direction_unit
        perp_norm = torch.norm(perp)
        if perp_norm.item() <= 1e-6:
            axis = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            if torch.abs(torch.dot(axis, direction_unit)).item() > 0.9:
                axis = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            perp = torch.cross(direction_unit, axis, dim=-1)
            perp_norm = torch.norm(perp)
            if perp_norm.item() <= 1e-6:
                return None
        perp = perp / perp_norm
        r_min, r_max = self.cfg.path_obstacle_offset_range
        offset = sample_uniform(r_min, r_max, (1,), device=self.device)
        candidate = base_point + perp * offset
        candidate = torch.max(torch.min(candidate, self._curr_obstacle_max), self._curr_obstacle_min)
        return candidate

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

    def _get_tcp_pose_w(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            link_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx]
            link_quat = self._robot.data.body_quat_w[:, self._tcp_body_idx]
            offset_pos = self._tcp_offset_pos
            offset_rot = self._tcp_offset_rot
        else:
            link_pos = self._robot.data.body_pos_w[env_ids, self._tcp_body_idx]
            link_quat = self._robot.data.body_quat_w[env_ids, self._tcp_body_idx]
            offset_pos = self._tcp_offset_pos[env_ids]
            offset_rot = self._tcp_offset_rot[env_ids]
        tcp_pos = link_pos + quat_apply(link_quat, offset_pos)
        tcp_quat = quat_mul(link_quat, offset_rot)
        return tcp_pos, tcp_quat

    def _get_tcp_pose(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_pos_w, tcp_quat = self._get_tcp_pose_w(env_ids)
        if env_ids is None:
            env_origins = self.scene.env_origins
        else:
            env_origins = self.scene.env_origins[env_ids]
        tcp_pos = tcp_pos_w - env_origins
        return tcp_pos, tcp_quat

    def _check_obstacle_collision(self) -> torch.Tensor:
        # Aggregate contact from robot contact sensor.
        if self._robot_contact_sensor is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sensor = self._robot_contact_sensor
        if sensor.cfg.filter_prim_paths_expr:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is not None:
                contact = torch.norm(force_matrix, dim=-1) > self.cfg.collision_force_threshold
                return torch.any(contact, dim=(1, 2))
        net_forces = sensor.data.net_forces_w
        if net_forces is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        contact = torch.norm(net_forces, dim=-1) > self.cfg.collision_force_threshold
        return torch.any(contact, dim=1)

    def _scale_joint_pos(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return 2.0 * (joint_pos - self._dof_lower_limits) / (self._dof_upper_limits - self._dof_lower_limits) - 1.0

    def _apply_curriculum_stage(self, stage_idx: int) -> None:
        # Update sampling ranges and tolerances when curriculum advances.
        stage = self._curriculum_cfg.stages[stage_idx]
        self._curriculum_stage = stage_idx
        self._curriculum_min_success_rate = stage.min_success_rate if stage.min_success_rate is not None else self._curriculum_cfg.min_success_rate
        self._curriculum_window_size = stage.window_size if stage.window_size is not None else self._curriculum_cfg.window_size
        self._curr_use_ik_reachability = stage.use_ik_reachability if stage.use_ik_reachability is not None else self.cfg.use_ik_reachability
        self._curr_target_pos_range = stage.target_pos_range
        self._curr_success_tolerance = stage.success_tolerance
        if stage.obstacle_pos_range is None:
            self._curr_obstacle_pos_range = self.cfg.obstacle_pos_range
        else:
            self._curr_obstacle_pos_range = stage.obstacle_pos_range
        self._active_obstacle_count = max(0, min(stage.active_obstacles, self.cfg.num_obstacles))
        self._curr_enable_los_check = (
            stage.enable_los_clearance_check if stage.enable_los_clearance_check is not None else self.cfg.enable_los_clearance_check
        )
        self._curr_los_clearance_margin = (
            stage.los_clearance_margin if stage.los_clearance_margin is not None else self.cfg.los_clearance_margin
        )
        self._obstacle_clearance_radius = self._obstacle_half_diag + self._curr_los_clearance_margin
        self._curr_force_path_obstacle = (
            stage.force_path_obstacle if stage.force_path_obstacle is not None else self.cfg.force_path_obstacle
        )
        self._update_target_bounds_tensors()
        self._update_obstacle_bounds_tensors()
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

    def _update_obstacle_bounds_tensors(self) -> None:
        self._curr_obstacle_min = torch.tensor(
            [
                self._curr_obstacle_pos_range[0][0],
                self._curr_obstacle_pos_range[1][0],
                self._curr_obstacle_pos_range[2][0],
            ],
            device=self.device,
        )
        self._curr_obstacle_max = torch.tensor(
            [
                self._curr_obstacle_pos_range[0][1],
                self._curr_obstacle_pos_range[1][1],
                self._curr_obstacle_pos_range[2][1],
            ],
            device=self.device,
        )

    def _reset_success_history(self, window_size: int | None = None) -> None:
        if window_size is not None:
            size = max(0, int(window_size))
            self._episode_success_history = torch.zeros(size, device=self.device)
            self._episode_success_history_all = torch.zeros(size, device=self.device)
        else:
            self._episode_success_history.zero_()
            self._episode_success_history_all.zero_()
        self._episode_history_idx = 0
        self._episode_history_filled = 0
        self._episode_history_idx_all = 0
        self._episode_history_filled_all = 0
        self._success_rate = 0.0
        self._success_rate_all = 0.0

    def _update_success_history(self, env_ids: torch.Tensor) -> None:
        # Track success history for curriculum gating.
        if self._episode_success_history.numel() == 0:
            return
        window_size = self._episode_success_history.numel()
        all_num = len(env_ids)
        if all_num > 0:
            success_values_all = self._success_buf[env_ids].float()
            idxs_all = (torch.arange(all_num, device=self.device) + self._episode_history_idx_all) % window_size
            self._episode_success_history_all[idxs_all] = success_values_all
            self._episode_history_idx_all = (self._episode_history_idx_all + all_num) % window_size
            self._episode_history_filled_all = min(window_size, self._episode_history_filled_all + all_num)
        valid_mask = ~(self._invalid_reset_buf[env_ids] | self._exclude_from_curriculum_buf[env_ids])
        valid_env_ids = env_ids[valid_mask]
        num = len(valid_env_ids)
        if num > 0:
            success_values = self._success_buf[valid_env_ids].float()
            idxs = (torch.arange(num, device=self.device) + self._episode_history_idx) % window_size
            self._episode_success_history[idxs] = success_values
            self._episode_history_idx = (self._episode_history_idx + num) % window_size
            self._episode_history_filled = min(window_size, self._episode_history_filled + num)
        if self._episode_history_filled > 0:
            if self._episode_history_filled < window_size:
                self._success_rate = self._episode_success_history[: self._episode_history_filled].mean().item()
            else:
                self._success_rate = self._episode_success_history.mean().item()
        else:
            self._success_rate = 0.0
        if self._episode_history_filled_all > 0:
            if self._episode_history_filled_all < window_size:
                self._success_rate_all = self._episode_success_history_all[: self._episode_history_filled_all].mean().item()
            else:
                self._success_rate_all = self._episode_success_history_all.mean().item()
        else:
            self._success_rate_all = 0.0
        self.extras["Stats/success_rate_valid"] = float(self._success_rate)
        self.extras["Stats/success_rate_all"] = float(self._success_rate_all)
        self.extras["Stats/success_rate"] = float(self._success_rate)
        if (
            self._curriculum_enabled
            and self._episode_history_filled >= window_size
            and self._success_rate >= self._curriculum_min_success_rate
            and self._curriculum_stage < len(self._curriculum_cfg.stages) - 1
        ):
            self._apply_curriculum_stage(self._curriculum_stage + 1)
            self._reset_success_history(self._curriculum_window_size)

    def _get_tcp_jacobian(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        jacobians = self._robot.root_physx_view.get_jacobians()
        jacobian = jacobians[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        if env_ids is None:
            return jacobian
        return jacobian[env_ids]

    def _get_tcp_jacobian_pos(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        jacobian = self._get_tcp_jacobian(env_ids)
        jacobian_pos = jacobian[:, :3, :]
        jacobian_rot = jacobian[:, 3:6, :]
        if env_ids is None:
            link_quat = self._robot.data.body_quat_w[:, self._tcp_body_idx]
            offset_pos = self._tcp_offset_pos
        else:
            link_quat = self._robot.data.body_quat_w[env_ids, self._tcp_body_idx]
            offset_pos = self._tcp_offset_pos[env_ids]
        offset_world = quat_apply(link_quat, offset_pos)
        cross_term = torch.cross(jacobian_rot.transpose(1, 2), offset_world.unsqueeze(1), dim=-1).transpose(1, 2)
        return jacobian_pos + cross_term

    def _compute_dls_delta(self, jacobian_pos: torch.Tensor, delta_pos: torch.Tensor) -> torch.Tensor:
        jacobian_t = torch.transpose(jacobian_pos, 1, 2)
        lambda_val = self.cfg.ik_reachability_dls_lambda
        eye = torch.eye(jacobian_pos.shape[1], device=self.device).unsqueeze(0).expand(jacobian_pos.shape[0], -1, -1)
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
        if not self._curr_use_ik_reachability:
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

    def _is_line_of_sight_clear(self, tcp_pos: torch.Tensor, target_pos: torch.Tensor, obstacle_pos: torch.Tensor) -> bool:
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

    def _is_obstacle_valid(self, obstacle_pos: torch.Tensor, target_pos: torch.Tensor, existing_obstacles: torch.Tensor) -> bool:
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
