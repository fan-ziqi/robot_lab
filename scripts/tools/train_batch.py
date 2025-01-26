# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import datetime
import subprocess
import time

import colorama

colorama.init(autoreset=True)

commands = [
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-Anymal-D-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 --headless",
    "python scripts/rsl_rl/base/train.py --task RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 --headless",
    "python scripts/rsl_rl/amp/train.py --task RobotLab-Isaac-Velocity-Flat-Amp-Unitree-A1-v0 --headless",
]

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/train_batch_{timestamp}.log"

with open(log_file, "w") as log:
    for cmd in commands:
        try:
            print(colorama.Fore.GREEN + f"Executing command: {cmd}")
            log.write(f"Executing command: {cmd}\n")
            log.flush()

            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Stream output to both console and log file
            for line in process.stdout:
                print(colorama.Fore.WHITE + line.strip())
                log.write(line)
                log.flush()

            for line in process.stderr:
                print(colorama.Fore.RED + line.strip())
                log.write(line)
                log.flush()

            return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

            print(colorama.Fore.GREEN + f"Command completed: {cmd}\n")
            log.write(f"Command completed: {cmd}\n")
            log.flush()

        except subprocess.CalledProcessError as e:
            error_message = f"Command failed: {cmd}\nError: {e}\n"
            print(colorama.Fore.RED + error_message)
            log.write(error_message)
            log.flush()
            break

        time.sleep(3)

print(colorama.Fore.GREEN + "All commands have been attempted!")
