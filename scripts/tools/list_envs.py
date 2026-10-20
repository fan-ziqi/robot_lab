# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""List registered robot_lab tasks without starting a simulator."""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyword", default=None)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable task list.")
    args = parser.parse_args()

    import gymnasium as gym
    import robot_lab.tasks  # noqa: F401

    rows = [
        {"task_id": spec.id, "entry_point": spec.entry_point, "config": spec.kwargs["env_cfg_entry_point"]}
        for spec in sorted(gym.registry.values(), key=lambda spec: spec.id)
        if str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("robot_lab.")
        and (args.keyword is None or args.keyword in spec.id)
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for index, row in enumerate(rows, 1):
            print(f"{index:3}  {row['task_id']}  {row['config']}")
        print(f"Total: {len(rows)} robot_lab tasks")


if __name__ == "__main__":
    main()
