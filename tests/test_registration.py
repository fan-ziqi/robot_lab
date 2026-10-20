# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""The installed package is discoverable through the upstream CLI."""

import subprocess
import sys

import pytest


def test_cli_discovers_project_tasks_without_manual_import():
    task = "RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0"
    result = subprocess.run(
        [sys.executable, "-m", "isaaclab", "list_envs", "--keyword", task],
        capture_output=True,
        text=True,
        check=True,
    )
    assert task in result.stdout


def test_amp_exposes_the_canonical_skrl_configuration():
    from isaaclab_tasks.utils import resolve_task_config

    import robot_lab.tasks  # noqa: F401

    _, agent_cfg = resolve_task_config("RobotLab-Isaac-G1-AMP-Dance-Direct-v0", "skrl_cfg_entry_point", overrides=[])
    assert agent_cfg["agent"]["class"] == "AMP"


@pytest.mark.parametrize(
    "model,terrain", [("GR1T1", "Rough"), ("GR1T1", "Flat"), ("GR1T2", "Rough"), ("GR1T2", "Flat")]
)
def test_fftai_cusrl_registration_selects_its_experiment(model, terrain):
    pytest.importorskip("cusrl")
    from isaaclab_tasks.utils import resolve_task_config

    import robot_lab.tasks  # noqa: F401

    task = f"RobotLab-Isaac-Velocity-{terrain}-FFTAI-{model}-v0"
    _, cfg = resolve_task_config(task, "cusrl_cfg_entry_point", overrides=[])
    assert cfg.experiment_name == f"fftai_{model.lower()}_{terrain.lower()}"
    assert cfg.agent_factory is not None
