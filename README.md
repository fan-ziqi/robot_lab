# omni.isaaclab.real_robot

## Get Ready

You need to install `Isaac Lab`.

## Installation

Using a python interpreter that has Isaac Lab installed, install the library

```bash
cd ext/real_robot
python -m pip install -e .
```

## Try unitree-a1 example

```bash
# To train
python scripts/rsl_rl/train.py --task RealRobot-Isaac-Velocity-Rough-Unitree-A1-v0 --headless
# To play
python scripts/rsl_rl/play.py --task RealRobot-Isaac-Velocity-Rough-Unitree-A1-Play-v0
```

## Add your own robot

To convert urdf, you need to run `convert_urdf.py` of dir `IsaacLab`

```bash
./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
  ~/git/anymal_d_simple_description/urdf/anymal.urdf \
  source/extensions/omni.isaac.lab_assets/data/Robots/<YOUR_ROBOT>/<YOUR_ROBOT>.usd \
  --merge-joints \
  --make-instanceable
```

Check [import_new_asset](https://docs.robotsfan.com/isaaclab/source/how-to/import_new_asset.html) for detail
