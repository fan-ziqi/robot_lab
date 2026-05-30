# Migrating from the IsaacLab subclass style

This guide is for IsaacLab users who maintain custom tasks in robot_lab and
want to keep them working after the dual-framework migration.

**TL;DR:** existing custom tasks (subclassing `LocomotionVelocityRoughEnvCfg`,
overriding fields in `__post_init__`) continue to work on IsaacLab unchanged.
Only two things break:

1. **Task IDs** dropped the `Isaac-` infix:
   `RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0` → `RobotLab-Velocity-Rough-Unitree-A1-v0`.
   Update shell aliases / CI / wandb run names.
2. **Log root** moved from `logs/{rsl_rl,skrl,cusrl}/<exp>` to
   `logs/isaaclab/<exp>`. Old wandb / tensorboard run links no longer
   resolve; new runs work normally.

## Quick start

```bash
# Old:
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 ...

# New:
python scripts/train.py --framework isaaclab --rl rsl_rl \
    --task RobotLab-Velocity-Rough-Unitree-A1-v0 ...
```

The dispatcher (`scripts/train.py`, `scripts/play.py`) auto-`exec`s under
`.venvs/isaaclab/bin/python`, so you don't need to source-activate first.

## What changed and why

The robot_lab repo now supports both **IsaacLab** and **mjlab** as simulator
backends. A new `robot_lab.framework` adapter package re-exports cfg types,
sensor/event factories, MDP function aliases, and the task-registration entry
point. Business code imports from `robot_lab.framework`; the framework module
chooses the active backend at first import via the `ROBOT_LAB_FRAMEWORK`
environment variable (set by the dispatcher).

For IsaacLab users, this changes very little:

| Aspect | Before | After |
|---|---|---|
| `import isaaclab.managers` etc. in your code | OK | Use `robot_lab.framework` re-exports |
| `gym.register(id="RobotLab-Isaac-Velocity-...")` | OK | `register_task(task_id="RobotLab-Velocity-...", framework_required="isaaclab")` |
| `LocomotionVelocityRoughEnvCfg` subclass + `__post_init__` overrides | OK | Same (kept untouched) |
| `from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg` in `agents/` | OK | Same — agent files still import directly; `--framework mjlab` skips them via `framework_required` |

If you have a custom task following the existing pattern, the only change you
must make is:

1. Strip `Isaac-` from your task ID(s).
2. Replace the `gym.register(id="...", entry_point="isaaclab.envs:ManagerBasedRLEnv", kwargs=...)`
   call with:

   ```python
   from robot_lab.framework import register_task

   register_task(
       task_id="RobotLab-Velocity-Rough-MyRobot-v0",
       env_cfg=f"{__name__}.rough_env_cfg:MyRobotRoughEnvCfg",
       agent_cfg_entries={
           "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotRoughPPORunnerCfg",
       },
       framework_required="isaaclab",
   )
   ```

That's it. Your env_cfg subclass, your `__post_init__` overrides, your
agent cfgs — all stay verbatim.

## Why `framework_required="isaaclab"`

`LocomotionVelocityRoughEnvCfg` inherits the IL-style nested
`@configclass` pattern. Under the mjlab backend, the class body's
`ObservationGroupCfg(...)` with no `terms=` argument fails because mjlab's
`ObservationGroupCfg` requires that field. So the subclass is IL-only by
design until someone authors a mjlab-native env cfg from scratch (Phase 7+
of the migration plan).

`framework_required="isaaclab"` makes `register_task` silently skip the
registration when running under `--framework mjlab`. mjlab users see your
task as "not registered" rather than as a class-evaluation crash.

## Adding a NEW task that should work on BOTH backends

This is the new path enabled by the migration. Use the `framework` factories:

```python
from robot_lab.framework import (
    EnvCfg,
    SceneEntityCfg,
    RewardTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    manager_container,
    make_observation_group,
    make_height_scan_sensor,
    make_contact_sensor,
    make_rough_terrain,
    make_sky_light,
    push_robot,
    randomize_geom_friction,
    register_task,
    mdp,  # framework-routed mdp aliases
)


def my_robot_rough_env_cfg() -> EnvCfg:
    rewards = manager_container({
        "track_lin_vel_xy_exp": RewardTermCfg(
            func=mdp.track_lin_vel_xy_exp,
            weight=3.0,
            params={"command_name": "base_velocity", "std": 0.5},
        ),
    })
    events = manager_container({
        "push_robot": push_robot(velocity_range={"x": (-0.5, 0.5)}),
    })
    # ... assemble scene, observations, actions, terminations, etc.
    cfg = EnvCfg(
        scene=...,
        rewards=rewards,
        events=events,
        ...,
    )
    return cfg


register_task(
    task_id="RobotLab-Velocity-Rough-MyRobot-v0",
    env_cfg=my_robot_rough_env_cfg,
    framework_required="any",  # registers on both backends
)
```

`framework_required="any"` (the default) registers the task on both
backends. The factory function returns `EnvCfg` which resolves to
`isaaclab.envs.ManagerBasedRLEnvCfg` under IL and
`mjlab.envs.ManagerBasedRlEnvCfg` under mjlab; same with `RewardTermCfg`,
`EventTermCfg`, etc.

## When backend-specific helpers don't exist

Some MDP terms exist only on IsaacLab (e.g. `joint_deviation_l1`,
`applied_torque_limits`, `undesired_contacts`, `feet_air_time_positive_biped`).

Two options:

1. **Mark the whole task IL-only:** `framework_required="isaaclab"`.
2. **Branch on framework at cfg-build time:**

   ```python
   from robot_lab.framework import current_framework_supports

   rewards = manager_container({
       "track_lin_vel_xy_exp": RewardTermCfg(
           func=mdp.track_lin_vel_xy_exp, weight=3.0,
           params={"command_name": "base_velocity", "std": 0.5},
       ),
       **({
           "joint_deviation_l1": RewardTermCfg(...),  # IL-only term
       } if current_framework_supports("isaaclab") else {}),
   })
   ```

## File layout reference

- `robot_lab.framework` — public adapter API. Cfg types, factories, MDP aliases.
- `robot_lab.framework.{isaaclab,mjlab}.*` — backend-specific implementations.
- `robot_lab.assets` — robot definitions. New `assets/data/<robot>.py`
  carries framework-neutral constants (paths, init pose, gains).
- `robot_lab.assets.builders.{isaaclab,mjlab}` — `build_articulation_cfg` /
  `build_entity_cfg` consume the neutral data and return the right cfg type.
- `robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg` —
  IL-style nested `@configclass` base; per-robot subclasses still inherit it.
- `scripts/{train,play}.py` — top-level dispatcher.
- `scripts/{isaaclab,mjlab}/...` — backend-specific entry points.

## Frequently surprising things

- `import robot_lab` works under both venvs even when nothing else loads
  (`tasks/__init__.py` swallows `ImportError`s during task registration).
- Under the mjlab venv, `import robot_lab.assets.unitree` succeeds but the
  module-level `UNITREE_GO2_CFG` resolves to `None` because the IL-side
  cfg classes aren't importable. Use the `build_unitree_a1_cfg()`-style
  factory instead, which dispatches on `get_framework()`.
- The mjlab CLI is **tyro**, not argparse. Pass `--env.scene.num-envs 4096`,
  not `--num_envs 4096`, when running under `--framework mjlab`.
- Old log paths (`logs/rsl_rl/`, `logs/skrl/`, `logs/cusrl/`) are no longer
  written. New logs are at `logs/isaaclab/<exp>/<run>` and
  `logs/mjlab/<exp>/<run>`.

## See also

- README "Frameworks" section.
- `robot_lab.framework` package source for the public adapter API.
