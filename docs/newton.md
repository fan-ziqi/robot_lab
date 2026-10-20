# Optional Newton MJWarp backend

Requires Isaac Lab 3.0 with its Newton dependencies and Isaac Sim 6.x for local URDF import.
PhysX remains the default. Select Newton when invoking a script directly:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --physics=newton_mjwarp --headless
```

## Supported tasks

The configured support set comprises 24 velocity tasks plus G1/H1 flat and rough (28 total):
Go2, Go2W, A1, B2, B2W, Agibot D1, Lite3, M20, MagicDog, ZSL1, and ZSL1W flat/rough;
ANYmal-D flat and its Play task; Unitree G1 and H1 flat/rough.
Other tasks retain PhysX configuration.

## Limitations

- Support denotes configured presets; smoke and short training do not establish learned behavior or transfer quality.
- Local URDF import uses Kit even in headless mode; Newton is not a standalone no-Kit path here.
- Joint, body, and contact ordering profiles depend on the exact asset and fixed-joint merge settings.
- Near-massless bodies receive Newton-only startup mass/inertia floors; this changes the Newton plant.
- Invalid Newton physical states terminate affected environments; resets can hide instability during long training.
- G1 flat and ANYmal-D flat have known learning difficulties also observed with Isaac Lab 2.x before this port.
- Video recording requires moviepy; Docker builds have not been exercised. RSL-RL policy configuration emits a deprecation warning ahead of Isaac Lab 3.1.

## Adding a robot profile

Measure joint/body order with PhysX for the exact asset and merge settings. Add a uniquely named
entry to `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/newton_ordering_profiles.json`
with asset provenance, SHA256 (URDF), joint names, body names, and measured contact sensor order when needed.
Call `configure_newton_from_profile` from the concrete task class, preserving its PhysX default.
Validate default/explicit PhysX config equality, then finite Newton smoke and short training before claiming support.
