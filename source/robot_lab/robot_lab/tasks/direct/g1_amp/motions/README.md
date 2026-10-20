# Motion files

The motion files are in NumPy-file format that contains data from the skeleton DOFs and bodies that perform the motion.

The data (accessed by key) is described in the following table, where:

* `N` is the number of motion frames recorded
* `D` is the number of skeleton DOFs
* `B` is the number of skeleton bodies

| Key | Dtype | Shape | Description |
| --- | ---- | ----- | ----------- |
| `fps` | int64 | () | FPS at which motion was sampled |
| `dof_names` | unicode string | (D,) | Skeleton DOF names |
| `body_names` | unicode string | (B,) | Skeleton body names |
| `dof_positions` | float32 | (N, D) | Skeleton DOF positions |
| `dof_velocities` | float32 | (N, D) | Skeleton DOF velocities |
| `body_positions` | float32 | (N, B, 3) | Skeleton body positions |
| `body_rotations` | float32 | (N, B, 4) | Skeleton body rotations (order declared by `quaternion_order`; untagged legacy files use `wxyz`) |
| `body_linear_velocities` | float32 | (N, B, 3) | Skeleton body linear velocities |
| `body_angular_velocities` | float32 | (N, B, 3) | Skeleton body angular velocities |

## Motion visualization

The `motion_viewer.py` file allows to visualize the skeleton motion recorded in a motion file.

Open an terminal in the `motions` folder and run the following command.

```bash
python motion_viewer.py --file MOTION_FILE_NAME.npz
```

See `python motion_viewer.py --help` for available arguments.

New converters write `quaternion_order="xyzw"`. The loader accepts `xyzw` or `wxyz`, defaults untagged legacy archives to `wxyz`, and converts once to runtime XYZW. Existing archives are not rewritten.

## Convert a retargeted CSV for AMP

From the repository root:

```bash
uv run --extra motion python source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/csv2npz.py --file /path/to/motion.csv --urdf /path/to/g1.urdf --mesh-dir /path/to/meshes --output /path/to/motion.npz --fps 30
```

The CSV contains root xyz, root quaternion xyzw and the 29 joint positions, in that order.
Use `--start-frame` and `--end-frame` to select a range. The output declares XYZW rotations
and stores body angular velocities in world coordinates.

Replay an AMP archive in Isaac Sim:

```bash
uv run --extra isaacsim python source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/motion_replayer.py --motion /path/to/motion.npz --viz kit
```

Use `--viz none --max_steps 100` for a bounded replay.
