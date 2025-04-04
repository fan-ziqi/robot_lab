# Booster T1 Description

This repository contains the urdf model of Booster Robotics T1. I get this urdf from [booster_gym](https://github.com/BoosterRobotics/booster_gym).

![T1](../../../.images/booster_t1.png)

Tested environment:

* Ubuntu 24.04
    * ROS2 Jazzy

## Build

```bash
cd ~/ros2_ws
colcon build --packages-up-to t1_description --symlink-install
```

## Visualize the robot

To visualize and check the configuration of the robot in rviz, simply launch:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch t1_description visualize.launch.py
```