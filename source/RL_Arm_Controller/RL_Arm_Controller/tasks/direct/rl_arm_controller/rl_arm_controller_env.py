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

        num_joints = len(self._joint_ids)
        self._actions = torch.zeros((self.num_envs, num_joints), device=self.device)
        self._prev_actions = torch.zeros_like(self._actions)
        self._joint_targets = self._default_joint_pos.clone()

        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._obstacle_pos = torch.zeros((self.num_envs, self.cfg.num_obstacles, 3), device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._has_prev_dist = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_stats_initialized = False
        self._invalid_reset_count = 0
        self._total_reset_count = 0

        self._tcp_forward_axis = torch.tensor(self.cfg.tcp_forward_axis, device=self.device).repeat(self.num_envs, 1)
        self._base_pos_local = torch.tensor(self.cfg.robot_cfg.init_state.pos, device=self.device)

        self._target_markers = VisualizationMarkers(self.cfg.target_marker_cfg)
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
        self._episode_success_history = torch.zeros(self._curriculum_cfg.window_size, device=self.device)
        self._episode_history_idx = 0
        self._episode_history_filled = 0
        self._success_rate = 0.0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # spawn obstacles
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

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)
        targets = self._default_joint_pos + self.cfg.action_scale * self._actions
        self._joint_targets = torch.clamp(targets, self._dof_lower_limits, self._dof_upper_limits)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._joint_ids] * self.cfg.dof_velocity_scale

        tcp_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx] - self.scene.env_origins
        target_rel = self._target_pos - tcp_pos

        obs_terms = [self._scale_joint_pos(joint_pos), joint_vel, target_rel]

        if self.cfg.include_obstacle_obs:
            obstacle_rel = self._obstacle_pos - tcp_pos.unsqueeze(1)
            obs_terms.append(obstacle_rel.view(self.num_envs, -1))

        if self.cfg.include_prev_actions:
            obs_terms.append(self._prev_actions)

        obs = torch.cat(obs_terms, dim=-1)
        obs = torch.clamp(obs, -self.cfg.obs_clip, self.cfg.obs_clip)

        self._prev_actions = self._actions.clone()

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        tcp_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx] - self.scene.env_origins
        to_target = self._target_pos - tcp_pos
        dist = torch.norm(to_target, dim=-1)

        reward = -self.cfg.rew_scale_dist * dist
        reward += (dist < self._curr_success_tolerance).float() * self.cfg.rew_scale_success

        progress = self._prev_dist - dist
        progress = torch.where(self._has_prev_dist, progress, torch.zeros_like(progress))
        reward += self.cfg.rew_scale_progress * progress

        action_penalty = torch.sum(self._actions**2, dim=-1)
        reward += self.cfg.rew_scale_action * action_penalty

        action_rate_penalty = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        reward += self.cfg.rew_scale_action_rate * action_rate_penalty

        joint_vel_penalty = torch.sum(self._robot.data.joint_vel[:, self._joint_ids] ** 2, dim=-1)
        reward += self.cfg.rew_scale_joint_vel * joint_vel_penalty

        if self.cfg.rew_scale_approach != 0.0:
            tcp_rot = self._robot.data.body_quat_w[:, self._tcp_body_idx]
            tcp_forward = quat_apply(tcp_rot, self._tcp_forward_axis)
            to_target_dir = normalize(to_target)
            reward += self.cfg.rew_scale_approach * torch.sum(tcp_forward * to_target_dir, dim=-1)

        collision = self._check_obstacle_collision()
        reward = torch.where(collision, reward + self.cfg.rew_scale_collision, reward)
        reward = torch.where(self._invalid_reset_buf, torch.zeros_like(reward), reward)

        self._prev_dist = dist
        self._has_prev_dist[:] = True

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        tcp_pos = self._robot.data.body_pos_w[:, self._tcp_body_idx] - self.scene.env_origins
        dist = torch.norm(self._target_pos - tcp_pos, dim=-1)
        success = dist < self._curr_success_tolerance
        self._success_buf = success

        collision = self._check_obstacle_collision()
        terminated = success
        if self.cfg.terminate_on_collision:
            terminated = terminated | collision

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = time_out | self._invalid_reset_buf
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
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

        target_valid = self._reset_targets(env_ids)
        obstacle_valid = self._reset_obstacles(env_ids)
        invalid_mask = ~(target_valid & obstacle_valid)
        self._invalid_reset_buf[env_ids] = invalid_mask
        self._total_reset_count += len(env_ids)
        self._invalid_reset_count += invalid_mask.sum().item()
        self.extras["Stats/invalid_env_fraction"] = self._invalid_reset_count / max(1, self._total_reset_count)
        self.extras["Stats/invalid_env_count"] = float(self._invalid_reset_count)
        self.extras["Stats/reset_count"] = float(self._total_reset_count)

        self._joint_targets[env_ids] = joint_pos[:, self._joint_ids]
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        self._prev_dist[env_ids] = 0.0
        self._has_prev_dist[env_ids] = False

    def _reset_targets(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        num_envs = len(env_ids)
        target_pos = torch.zeros((num_envs, 3), device=self.device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
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
                valid_mask[i] = False
                center = ee_pos_local[i : i + 1]
                offset = self._sample_uniform_pos(1, ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))[0]
                offset = offset * self.cfg.invalid_target_fallback_radius
                candidate = center.squeeze(0) + offset
                target_min = torch.tensor(
                    [self._curr_target_pos_range[0][0], self._curr_target_pos_range[1][0], self._curr_target_pos_range[2][0]],
                    device=self.device,
                )
                target_max = torch.tensor(
                    [self._curr_target_pos_range[0][1], self._curr_target_pos_range[1][1], self._curr_target_pos_range[2][1]],
                    device=self.device,
                )
                candidate = torch.max(torch.min(candidate, target_max), target_min)
            target_pos[i] = candidate
        self._target_pos[env_ids] = target_pos
        self._update_target_markers()
        return valid_mask

    def _reset_obstacles(self, env_ids: torch.Tensor) -> torch.Tensor:
        num_envs = len(env_ids)
        obstacle_pos = torch.zeros((num_envs, self.cfg.num_obstacles, 3), device=self.device)
        valid_mask = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        inactive_pos = torch.tensor(self.cfg.obstacle_inactive_pos, device=self.device)
        active_count = max(0, min(self._active_obstacle_count, self.cfg.num_obstacles))
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
            if not env_valid:
                valid_mask[env_idx] = False
                obstacle_pos[env_idx, :active_count] = inactive_pos
            if active_count < self.cfg.num_obstacles:
                obstacle_pos[env_idx, active_count:] = inactive_pos
        self._obstacle_pos[env_ids] = obstacle_pos

        for i, obstacle in enumerate(self._obstacles):
            root_state = obstacle.data.default_root_state[env_ids].clone()
            root_state[:, :3] = obstacle_pos[:, i, :] + self.scene.env_origins[env_ids]
            root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            root_state[:, 7:] = 0.0
            obstacle.write_root_pose_to_sim(root_state[:, :7], env_ids)
            obstacle.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        return valid_mask

    def _update_target_markers(self):
        target_pos_w = self._target_pos + self.scene.env_origins
        self._target_markers.visualize(translations=target_pos_w)

    def _check_obstacle_collision(self) -> torch.Tensor:
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
        stage = self._curriculum_cfg.stages[stage_idx]
        self._curriculum_stage = stage_idx
        self._curr_target_pos_range = stage.target_pos_range
        self._curr_success_tolerance = stage.success_tolerance
        if stage.obstacle_pos_range is None:
            self._curr_obstacle_pos_range = self.cfg.obstacle_pos_range
        else:
            self._curr_obstacle_pos_range = stage.obstacle_pos_range
        self._active_obstacle_count = max(0, min(stage.active_obstacles, self.cfg.num_obstacles))
        self.extras["Stats/curriculum_stage"] = float(self._curriculum_stage)
        self.extras["Stats/curriculum_success_tolerance"] = float(self._curr_success_tolerance)

    def _reset_success_history(self) -> None:
        self._episode_success_history.zero_()
        self._episode_history_idx = 0
        self._episode_history_filled = 0
        self._success_rate = 0.0

    def _update_success_history(self, env_ids: torch.Tensor) -> None:
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
