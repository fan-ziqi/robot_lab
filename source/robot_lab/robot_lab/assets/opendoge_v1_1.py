"""OpenDog V1.1 from LainLab feat/opendoge; see bundled provenance/licenses."""

from robot_lab.assets.opendoge import OPENDOGE_CFG
from robot_lab.policies.opendoge_v1_1 import ASSET_DIR, JOINT_NAMES, PolicyState

# Keep the original robot and exports independently usable.
OPENDOGE_V1_1_CFG = OPENDOGE_CFG.copy()
OPENDOGE_V1_1_CFG.spawn.asset_path = str(ASSET_DIR / "urdf/opendoge.urdf")
OPENDOGE_V1_1_CFG.init_state.pos = (0.0, 0.0, 0.158)
OPENDOGE_V1_1_CFG.init_state.joint_pos = dict(zip(JOINT_NAMES, PolicyState().default_angles.tolist(), strict=True))
# The mjlab source has torque saturation but no motor-speed limiting.
OPENDOGE_V1_1_CFG.actuators["legs"].actuator_velocity_limit = 1.0e9
OPENDOGE_V1_1_CFG.actuators["legs"].joint_velocity_limit = 1.0e9
