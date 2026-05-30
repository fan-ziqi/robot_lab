"""Public API for the dual-framework adapter layer.

Business code imports from here only. The active backend is chosen at first
import of this module via :mod:`robot_lab.framework.detect`.
"""

from __future__ import annotations

from robot_lab.framework.detect import (
    FRAMEWORK_ENV_VAR,
    get_framework,
    set_framework,
)
from robot_lab.framework.spec import (
    FrameworkRequired,
    FrameworkRequiredError,
    current_framework_supports,
)

_FW = get_framework()

if _FW == "isaaclab":
    from robot_lab.framework.isaaclab.cfg_types import (
        ActionTermCfg,
        CommandTermCfg,
        CurriculumTermCfg,
        EnvCfg,
        EventTermCfg,
        ManagerTermBase,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from robot_lab.framework.isaaclab.container import manager_container
    from robot_lab.framework.isaaclab import mdp_aliases as mdp
    from robot_lab.framework.isaaclab.sensors import (
        make_contact_sensor,
        make_height_scan_sensor,
    )
    from robot_lab.framework.isaaclab.events import (
        apply_external_force_torque,
        push_robot,
        randomize_actuator_gains,
        randomize_body_com,
        randomize_body_mass,
        randomize_geom_friction,
        randomize_reset_base,
        randomize_reset_joints,
    )
    from robot_lab.framework.isaaclab.scene import (
        make_flat_terrain,
        make_rough_terrain,
        make_sky_light,
    )
    from robot_lab.framework.isaaclab import contact
    from robot_lab.framework.isaaclab.sim_settings import apply_sim_settings
    from robot_lab.framework.isaaclab.register import register_task
    from robot_lab.framework.isaaclab.app import launch_app
    from robot_lab.framework.isaaclab import rl as rl_cfg
    from robot_lab.framework.isaaclab.obs_group import make_observation_group
elif _FW == "mjlab":
    from robot_lab.framework.mjlab.cfg_types import (
        ActionTermCfg,
        CommandTermCfg,
        CurriculumTermCfg,
        EnvCfg,
        EventTermCfg,
        ManagerTermBase,
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        SceneCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from robot_lab.framework.mjlab.container import manager_container
    from robot_lab.framework.mjlab import mdp_aliases as mdp
    from robot_lab.framework.mjlab.sensors import (
        make_contact_sensor,
        make_height_scan_sensor,
    )
    from robot_lab.framework.mjlab.events import (
        apply_external_force_torque,
        push_robot,
        randomize_actuator_gains,
        randomize_body_com,
        randomize_body_mass,
        randomize_geom_friction,
        randomize_reset_base,
        randomize_reset_joints,
    )
    from robot_lab.framework.mjlab.scene import (
        make_flat_terrain,
        make_rough_terrain,
        make_sky_light,
    )
    from robot_lab.framework.mjlab import contact
    from robot_lab.framework.mjlab.sim_settings import apply_sim_settings
    from robot_lab.framework.mjlab.register import register_task
    from robot_lab.framework.mjlab.app import launch_app
    from robot_lab.framework.mjlab import rl as rl_cfg
    from robot_lab.framework.mjlab.obs_group import make_observation_group
else:  # pragma: no cover - validated in detect
    raise RuntimeError(f"Unknown framework {_FW!r}")

__all__ = [
    "ActionTermCfg",
    "CommandTermCfg",
    "CurriculumTermCfg",
    "EnvCfg",
    "EventTermCfg",
    "FRAMEWORK_ENV_VAR",
    "FrameworkRequired",
    "FrameworkRequiredError",
    "ManagerTermBase",
    "ObservationGroupCfg",
    "ObservationTermCfg",
    "RewardTermCfg",
    "SceneCfg",
    "SceneEntityCfg",
    "TerminationTermCfg",
    "apply_external_force_torque",
    "apply_sim_settings",
    "contact",
    "current_framework_supports",
    "get_framework",
    "launch_app",
    "make_contact_sensor",
    "make_flat_terrain",
    "make_height_scan_sensor",
    "make_observation_group",
    "make_rough_terrain",
    "make_sky_light",
    "manager_container",
    "mdp",
    "push_robot",
    "randomize_actuator_gains",
    "randomize_body_com",
    "randomize_body_mass",
    "randomize_geom_friction",
    "randomize_reset_base",
    "randomize_reset_joints",
    "register_task",
    "rl_cfg",
    "set_framework",
]
