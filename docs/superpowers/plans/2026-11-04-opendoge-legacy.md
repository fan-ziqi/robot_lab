# OpenDoge legacy migration

Goal: make the original OpenDoge URDF and exported HIM policy independently usable in robot_lab.

Approach: copy the original URDF/meshes and the real dist ONNX binary; expose an ArticulationCfg and a bounded standalone Isaac Lab replay script. Use the original 0.6/-1.5 stance, PD 12/0.5, action scale 0.30, torque limits 6/6/9, armature 0.005 and 200/100 Hz timing. Build the exact command-first 45-value observation with six newest-first frames. Resolve joints by name. No HIM retraining or V1.1 substitution.

- [x] Test observation layout, history/reset, action clipping, real ONNX inference and asset references.
- [x] Copy assets with source attribution and implement the replay adapter and launcher.
- [x] Run focused tests, existing suite, lint and bounded headless simulation; report measured stability separately from successful execution.

Verification: 14 tests passed, 4 optional cusrl tests skipped; Ruff lint/format and uv lock check passed. Independent code review found no blocking observation/joint/PD issues. Newton replay exited successfully for 20 seconds standing and 20 seconds with vx=0.3. Neither fell below 0.08 m. Forward mean body vx=0.0728 m/s with lateral drift: playback works, tracking parity is not established. Source license is MIT despite the source README claiming BSD; retain the actual LICENSE text.
