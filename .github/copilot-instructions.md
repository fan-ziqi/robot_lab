# Copilot instructions for `robot_lab`

## Project snapshot
- This repo is an Isaac Lab 2.3.0 extension that adds dozens of locomotion tasks plus MuJoCo validation for custom robots (notably `mydog`).
- Core code lives under `source/robot_lab/robot_lab/` and is installable via `python -m pip install -e source/robot_lab` (run `scripts/tools/list_envs.py` afterward to confirm registry wiring).

## Architecture at a glance
- Config stack: `velocity_env_cfg.py` (base) → robot-specific `rough_env_cfg.py` → optional `flat_env_cfg.py`; agents live beside configs (e.g., `.../agents/rsl_rl_ppo_cfg.py`).
- Robot assets (`assets/*.py`) wrap URDF/MJCF in `ArticulationCfg` objects; `data/Robots/**/` stores meshes+MJCF that must stay in sync with those configs.
- Environments register through each config package's `__init__.py` via `gym.register(...)` naming them `RobotLab-Isaac-<Task>-<Terrain>-<Robot>-v#`.
- RL scripts sit in `scripts/reinforcement_learning/{rsl_rl,cusrl,skrl}/`, and logs/checkpoints land in `logs/<agent>/<task>/<timestamp>/` with Hydra dumps in `outputs/<date>/<time>/`.

## Daily workflows
- **VS Code setup:** Run the `setup_python_env` task once to generate `.vscode/.python.env` so Pylance resolves Isaac imports.
- **Install & format:** use the project interpreter that already has Isaac Lab, then `pip install pre-commit && pre-commit run --all-files` for formatting.
- **Train:** `python scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Rough-MyRobot-v0 --headless [--num_envs 4096]` (scale via `python -m torch.distributed.run ... --distributed` for multi-GPU/multi-node).
- **Evaluate / keyboard teleop:** `play.py --task=<ENV> --keyboard` allows single-robot WASD/QE control; prefer `--num_envs 64` for batch evals.
- **Tools:** Use `scripts/tools/clean_trash.py` to purge Isaac USD caches instead of manual `rm -rf`.
- **Logging:** tensorboard consumes `logs/`; Hydra configs for a given run live under `outputs/<date>/<time>/config.yaml`.
- Keep terminal output lean—disable tqdm (`TQDM_DISABLE=1`) when sharing commands or scrape-only summaries.

## MuJoCo validation (`sim_m.py`, `run_mujoco.py`)
- Designed to replay TorchScript policies (`policy = torch.jit.load(...)`) on MJCF models stored at `source/robot_lab/data/Robots/<...>/mjcf/`.
- Joint ordering differs from Isaac: see `dof_ids`/`dof_vel` mappings and `default_joint_angles` before adding controllers.
- Keyboard listener (auto-enabled) maps WASD/QE to desired body velocities and `[]` + arrow keys to online PD tuning (`kp_step=10`, `kd_step=0.1`).
- Realtime/summary plotting: set `--realtime-plot` for live monitoring or `--plot` to persist `joint_tracking.png` + `joint_error_torque.png`.

## Conventions & gotchas
- Always update both `assets/<robot>.py` and `data/Robots/...` when altering kinematics; controllers expect matching joint names.
- Heightfield / curriculum tuning happens through `tasks/manager_based/locomotion/velocity/mdp/` helpers—import via `from .mdp import *` instead of duplicating logic.
- Reward entries with zero weight should be removed through `disable_zero_weight_rewards()` to keep configs tidy.
- Logs and configs assume UTC-style timestamp folders; don't rename them manually or Hydrated resumes break.
- When adding new commands/observations, keep the policy/critic observation groups symmetric; critic variants should remain noise-free.

## Extending robots or tasks
- Clone an existing robot folder in `config/wheeled|quadruped|humanoid/<robot>/`, adjust spawn paths, and register the new env id in that folder's `__init__.py`.
- Pair every new env with agent configs (at least `rsl_rl_ppo_cfg.py`), and add MJCF/URDF assets before running `scripts/tools/list_envs.py`.
- Validate in Isaac first (`play.py --keyboard`), then port the policy to MuJoCo via `sim_m.py --model <mjcf> --policy <policy.pt>`; PD gains in `RobotConfig` are the main knobs for matching behaviors.
