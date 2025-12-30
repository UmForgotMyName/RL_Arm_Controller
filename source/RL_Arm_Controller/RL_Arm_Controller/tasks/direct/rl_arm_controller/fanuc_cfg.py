"""FANUC LR Mate 200iC/5L with SG2 tool articulation configuration."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_FANUC_USD_PATH = (_PROJECT_ROOT / "assets" / "Robots" / "FANUC" / "usd" / "fanuc200ic5l_sg2.usd").as_posix()


FANUC_LRMATE_SG2_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FANUC_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.8,
            "joint_3": 1.6,
            "joint_4": 0.0,
            "joint_5": 1.0,
            "joint_6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "fanuc_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit_sim=200.0,
            velocity_limit_sim=6.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
