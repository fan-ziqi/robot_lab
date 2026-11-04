# OpenDog V1.1 / LainLab exports

Source: https://github.com/Lain-Ego0/LainLab/tree/feat/opendoge
Revision: `7f6df34c67c455a80b9d296344a53e8a969cd765`.

- `urdf/`, `xmls/`: original LainLab robot definitions. `meshes/` duplicates
  `xmls/assets/` so the original URDF's relative mesh paths resolve locally.
- `policies/{flat,getup,rough,skills}/`: original ONNX, training checkpoint and
  upstream baseline metadata. These files are copied unchanged. The ONNX graphs
  include observation normalization. Checkpoints target mjlab/RSL-RL, not a
  generic robot_lab training runner.
- `provenance.json`: SHA-256 checksums of copied files and pinned source revision.
- `SOURCE_LICENSE`: LainLab's Apache-2.0 license, copied verbatim.
- `MODEL_SOURCE_LICENSE`: MIT license of OpenDoge_train, the original model
  source cited by LainLab. V1.1 meshes match that source byte for byte.

Migration additions/changes: this README, the provenance manifest and the mesh
directory placement. Source robot parameters and policy weights are unchanged.
Isaac Lab configuration, observation adaptation and replay are provided separately
in `robot_lab.assets.opendoge_v1_1`, `robot_lab.policies.opendoge_v1_1` and
`scripts/reinforcement_learning/play_opendoge_v1_1.py`.

Upstream baseline descriptions/metrics document upstream selection; they are not
evidence of performance in Isaac Lab. See `docs/opendoge-v1.1.md` for local results.
