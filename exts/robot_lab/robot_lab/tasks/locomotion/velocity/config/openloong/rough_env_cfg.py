import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
# use local assets
from robot_lab.assets.openloong_12 import OPENLOONG_CFG # isort: skip


@configclass
class openloong_12_RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    _run_disable_zero_weight_rewards = True
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to fftai-GR1T1
        self.scene.robot = OPENLOONG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # ------------------------------Observations------------------------------
        # self.observations.policy.base_lin_vel = None
        # self.observations.policy.height_scan = None
        self.observations.AMP = None
        # ------------------------------Actions------------------------------

        # ------------------------------Events------------------------------
        self.events.reset_base_amp = None
        self.events.reset_robot_joints_amp = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_parameters = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = -200

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0
        self.rewards.body_lin_acc_l2.weight = 0

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = 0
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_other_l1", -1.0, [
                                                                                           ".*_hip_yaw", ".*_hip_roll",                                                                                    
                                                                                           ])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_knee_l1", -0.1, ['.*_knee_pitch'])
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = [".*_ankle_pitch",".*_ankle_roll"]
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.applied_torque_limits.weight = 0
        self.rewards.action_rate_l2.weight = -0.005
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.contact_forces.weight = 0

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5

        # Others
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ["Link_r_ankle_roll","Link_l_ankle_roll"]
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.foot_contact.weight = 0
        self.rewards.base_height_rough_l2.weight = 0
        self.rewards.feet_slide.weight = -0.25
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ["Link_r_ankle_roll","Link_l_ankle_roll"]
        self.rewards.feet_slide.params["asset_cfg"].body_names = ["Link_r_ankle_roll","Link_l_ankle_roll"]
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still_when_zero_command.weight = 0

        if self._run_disable_zero_weight_rewards:
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base_link",".*_waist_.*",".*_head_.*",".*_arm_r_.*",".*_arm_l_.*",".*_hip_r_.*",".*_hip_l_.*"]
        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class openloong_12_RoughEnvCfg_PLAY(openloong_12_RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0) 
