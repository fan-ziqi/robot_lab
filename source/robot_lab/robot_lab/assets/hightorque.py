import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

HIGHTORQUE_PI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/HighTorque/PI/pi.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3453),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_thigh_joint",
                ".*_calf_joint",
            ],
            effort_limit=21,
            velocity_limit=4,
            stiffness={
                ".*_hip_pitch_joint":40,
                ".*_hip_roll_joint":20,
                ".*_thigh_joint":20,
                ".*_calf_joint":40,
            },
            damping={
                ".*_hip_pitch_joint":0.6,
                ".*_hip_roll_joint":0.4,
                ".*_thigh_joint":0.4,
                ".*_calf_joint":0.6,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint"],
            effort_limit=21,
            velocity_limit=4,
            stiffness={
                ".*_ankle_pitch_joint":40,
                ".*_ankle_roll_joint":20,
            },
            damping={
                ".*_ankle_pitch_joint":0.6,
                ".*_ankle_roll_joint":0.4,
            },
            armature=0.01,
        ),
    },
)