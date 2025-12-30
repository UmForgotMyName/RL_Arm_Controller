# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .fanuc_cfg import FANUC_LRMATE_SG2_CFG


@configclass
class CurriculumStageCfg:
    target_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    success_tolerance: float
    active_obstacles: int
    obstacle_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None


@configclass
class SuccessRateCurriculumCfg:
    enabled = True
    window_size = 200
    min_success_rate = 0.7
    stages = [
        CurriculumStageCfg(
            target_pos_range=((0.4, 0.6), (-0.2, 0.2), (0.25, 0.45)),
            success_tolerance=0.05,
            active_obstacles=0,
        ),
        CurriculumStageCfg(
            target_pos_range=((0.35, 0.65), (-0.3, 0.3), (0.2, 0.55)),
            success_tolerance=0.04,
            active_obstacles=1,
            obstacle_pos_range=((0.3, 0.55), (-0.25, 0.25), (0.2, 0.5)),
        ),
        CurriculumStageCfg(
            target_pos_range=((0.35, 0.65), (-0.35, 0.35), (0.2, 0.6)),
            success_tolerance=0.03,
            active_obstacles=2,
            obstacle_pos_range=((0.25, 0.6), (-0.3, 0.3), (0.15, 0.55)),
        ),
    ]


@configclass
class RlArmControllerEnvCfg(DirectRLEnvCfg):
    # ---- Env ----
    decimation = 2
    episode_length_s = 6.0
    action_space = 6
    observation_space = 0
    state_space = 0

    # ---- Simulation ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # ---- Robot ----
    robot_cfg: ArticulationCfg = FANUC_LRMATE_SG2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    tcp_body_name = "tcp"
    tcp_prim_path = "/lrmate200ic5l_with_sg2/link_6/flange/tool0/sg2/tcp"
    tcp_forward_axis = (0.0, 0.0, 1.0)

    # ---- Scene ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # ---- Targets ----
    target_pos_range = ((0.35, 0.65), (-0.35, 0.35), (0.2, 0.6))
    target_min_distance_from_base = 0.2
    target_max_distance_from_base = 0.9
    target_resample_attempts = 30
    reset_repair_attempts = 3
    use_ik_reachability = True
    ik_reachability_iters = 8
    ik_reachability_tolerance = 0.02
    ik_reachability_dls_lambda = 0.02
    ik_reachability_max_delta = 0.15
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target_marker",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
            )
        },
    )

    # ---- Obstacles ----
    num_obstacles = 2
    obstacle_pos_range = ((0.25, 0.6), (-0.3, 0.3), (0.15, 0.55))
    obstacle_size = (0.08, 0.08, 0.12)
    obstacle_min_distance_between = 0.14
    obstacle_min_distance_from_target = 0.14
    obstacle_min_distance_from_base = 0.2
    obstacle_resample_attempts = 40
    obstacle_inactive_pos = (0.0, 0.0, 10.0)
    enable_los_clearance_check = True
    los_clearance_margin = 0.02
    obstacle_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_0",
        spawn=sim_utils.CuboidCfg(
            size=obstacle_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.3), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    obstacle_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Obstacle_0",
        update_period=0.0,
        history_length=0,
        force_threshold=1.0,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/.*"],
    )

    # ---- Control ----
    action_scale = 0.5
    reset_joint_pos_noise = 0.05
    dof_velocity_scale = 0.1
    include_obstacle_obs = True
    include_prev_actions = True
    obs_clip = 5.0

    # ---- Rewards ----
    rew_scale_dist = 1.0
    rew_scale_success = 5.0
    rew_scale_action = -0.01
    rew_scale_joint_vel = -0.005
    rew_scale_collision = -5.0
    rew_scale_progress = 0.5
    rew_scale_action_rate = -0.01
    rew_scale_approach = 0.0
    success_tolerance = 0.03
    terminate_on_collision = False
    collision_force_threshold = 1.0
    invalid_target_fallback_radius = 0.05
    enable_stuck_termination = False
    stuck_min_progress = 0.005
    stuck_steps = 60
    invalid_fraction_ema_alpha = 0.05
    invalid_fraction_window = 200
    # ---- Curriculum ----
    curriculum: SuccessRateCurriculumCfg = SuccessRateCurriculumCfg()

    def __post_init__(self):
        obs_dim = 2 * len(self.joint_names) + 3
        if self.include_obstacle_obs:
            obs_dim += 3 * self.num_obstacles
        if self.include_prev_actions:
            obs_dim += len(self.joint_names)
        self.observation_space = obs_dim
        self.action_space = len(self.joint_names)
        self.obstacle_contact_sensor.force_threshold = self.collision_force_threshold
