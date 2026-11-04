# Original OpenDoge

Copied from the sibling OpenDoge_train checkout on 2026-11-04:

Source commit: `c969f27c282df4be9e5bae61d290d027670f9d01`.
Policy SHA256: `42302a4de504c9eac30c6aa9ed9fec872bd90b5b1115f0d943725bc6443c4864`.

- `resources/robots/Opendoge/urdf/Opendoge.urdf`
- `resources/robots/Opendoge/meshes/*.STL`
- `Tool/sim2sim_web/browser/dist/onnx/flat_opendoge_9000_omni.onnx`

Source checkout's MIT license: `SOURCE_LICENSE`. Keep it with redistributed assets.
This is the original robot, not OpenDog V1.1. The public/onnx files in the source
checkout are text placeholders referring to Linux paths; the dist copy is the real export.

Policy parameters come from `legged_gym/envs/opendoge/opendoge_config.py` and
`Tool/sim2sim_web/browser/src/engine/config.ts`. The source deploy `opendoge.yaml`
currently mixes in V1.1 parameters and must not be used for this robot.

This export supports inference only. Continuing HIM training requires a separate
port of its estimator, PPO algorithm, storage and runner.
