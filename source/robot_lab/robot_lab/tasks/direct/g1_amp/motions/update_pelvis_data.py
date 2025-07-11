import numpy as np
import os
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R

def plot_pelvis_data_2d(source_data, target_data, source_pelvis_idx, target_pelvis_idx):
    """
    Generates 2D plots comparing pelvis data (positions, velocities, rotations)
    from source and target files over time.
    """
    print("Generating 2D comparison plots...")

    plots_to_create = [
        ('body_positions', 'Pelvis Position Comparison', 'Position (m)'),
        ('body_linear_velocities', 'Pelvis Linear Velocity Comparison', 'Velocity (m/s)'),
        ('body_angular_velocities', 'Pelvis Angular Velocity Comparison', 'Angular Velocity (rad/s)'),
        ('body_rotations', 'Pelvis Rotation (Quaternion) Comparison', 'Value')
    ]

    fig = make_subplots(
        rows=len(plots_to_create), cols=1,
        subplot_titles=[p[1] for p in plots_to_create],
        vertical_spacing=0.08
    )

    fps = source_data.get('fps', 60)

    for i, (key, title, y_title) in enumerate(plots_to_create, 1):
        for data, idx, name in [(source_data, source_pelvis_idx, 'Source'), (target_data, target_pelvis_idx, 'Target')]:
            if key not in data:
                continue
            
            motion_data = data[key][:, idx]
            num_frames = motion_data.shape[0]
            time = np.arange(num_frames) / fps
            
            dims = motion_data.shape[1]
            labels = ['X', 'Y', 'Z', 'W'] if dims == 4 else ['X', 'Y', 'Z']
            
            for d in range(dims):
                fig.add_trace(
                    go.Scatter(
                        x=time,
                        y=motion_data[:, d],
                        name=f'{name} {labels[d]}',
                        legendgroup=f'group{i}',
                        line=dict(dash='solid' if name == 'Source' else 'dash', width=1.5),
                        showlegend=(d==0) # Show legend only for the first component to avoid clutter
                    ),
                    row=i, col=1
                )
        fig.update_yaxes(title_text=y_title, row=i, col=1)
        fig.update_xaxes(title_text="Time (s)", row=i, col=1)

    fig.update_layout(
        title_text="Pelvis Data 2D Comparison",
        height=350 * len(plots_to_create),
        legend_tracegroupgap = 250
    )
    
    output_html_file = "pelvis_data_2d_comparison.html"
    fig.write_html(output_html_file)
    print(f"2D plot visualization saved to: {os.path.abspath(output_html_file)}")


def visualize_pelvis_trajectories(source_data, target_data, source_pelvis_idx, target_pelvis_idx):
    """
    Visualizes and compares the 3D trajectories and orientations of the pelvis from two motion data sources.
    """
    print("Generating 3D trajectory visualization...")
    
    fig = go.Figure()

    for data, idx, name in [(source_data, source_pelvis_idx, 'Source'), (target_data, target_pelvis_idx, 'Target')]:
        positions = data['body_positions'][:, idx, :]
        rotations_quat = data['body_rotations'][:, idx, :]

        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            mode='lines',
            name=f'{name} Trajectory',
            line=dict(width=4)
        ))

        # Add orientation cones
        num_frames = positions.shape[0]
        step = max(1, num_frames // 20) # Show about 20 orientation markers
        
        rotations = R.from_quat(rotations_quat[:, [1, 2, 3, 0]]) # reorder to x,y,z,w for SciPy

        # Get the forward vector (e.g., x-axis) for each orientation
        forward_vectors = rotations.apply([0.1, 0, 0]) 

        fig.add_trace(go.Cone(
            x=positions[::step, 0], y=positions[::step, 1], z=positions[::step, 2],
            u=forward_vectors[::step, 0], v=forward_vectors[::step, 1], w=forward_vectors[::step, 2],
            sizemode="absolute", sizeref=0.1,
            showscale=False,
            name=f'{name} Orientation',
            anchor="tip"
        ))

    fig.update_layout(
        title='Pelvis 3D Trajectory and Orientation Comparison',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        legend_title_text='Data Source'
    )
    
    output_html_file = "pelvis_trajectory_comparison.html"
    fig.write_html(output_html_file)
    print(f"Visualization saved to: {os.path.abspath(output_html_file)}")


def update_pelvis_data(source_file, target_file, dry_run=False):
    """
    Copies the pelvis data from a source motion file to a target motion file.

    This function loads two .npz motion files, finds the 'pelvis' body index in both,
    and then overwrites the pelvis trajectory data (position, rotation, velocities)
    in the target file with the data from the source file.

    Args:
        source_file (str): Path to the .npz file with the source pelvis data.
        target_file (str): Path to the .npz file to be updated.
        dry_run (bool): If True, print data for comparison without saving.
    """
    try:
        # Load the source and target .npz files
        source_data = np.load(source_file)
        target_data = np.load(target_file)
        print(f"Loaded source file: {source_file}")
        print(f"Loaded target file: {target_file}")

        # Find the index of 'pelvis' in the body_names array for both files
        source_body_names = list(source_data['body_names'])
        target_body_names = list(target_data['body_names'])

        source_pelvis_idx = source_body_names.index('pelvis')
        target_pelvis_idx = target_body_names.index('pelvis')
        print(f"Found 'pelvis' at index {source_pelvis_idx} in source file.")
        print(f"Found 'pelvis' at index {target_pelvis_idx} in target file.")

        if dry_run:
            print("\n--- DRY RUN MODE ---")
            print("Displaying pelvis data for the first 5 frames. No files will be changed.")
            
            print("\n--- Source Pelvis Data (first 5 frames) ---")
            print("Body Positions:")
            print(source_data['body_positions'][:5, source_pelvis_idx])
            print("\nBody Rotations:")
            print(source_data['body_rotations'][:5, source_pelvis_idx])
            
            print("\n--- Target Pelvis Data (first 5 frames) ---")
            print("Body Positions:")
            print(target_data['body_positions'][:5, target_pelvis_idx])
            print("\nBody Rotations:")
            print(target_data['body_rotations'][:5, target_pelvis_idx])
            
            visualize_pelvis_trajectories(source_data, target_data, source_pelvis_idx, target_pelvis_idx)
            plot_pelvis_data_2d(source_data, target_data, source_pelvis_idx, target_pelvis_idx)

            print("\n--- END DRY RUN ---")
            return

        # Convert the loaded target data into a mutable dictionary
        target_data_dict = {key: target_data[key] for key in target_data.files}
        
        # Check if the number of frames is consistent
        num_frames_source = source_data['body_positions'].shape[0]
        num_frames_target = target_data_dict['body_positions'].shape[0]
        if num_frames_source < num_frames_target:
            print(f"Warning: Source file has fewer frames ({num_frames_source}) than target file ({num_frames_target}). Target will be truncated.")
            num_frames = num_frames_source
            # Truncate all relevant arrays in the target data
            for key in ['dof_positions', 'dof_velocities', 'body_positions', 'body_rotations', 'body_linear_velocities', 'body_angular_velocities']:
                if key in target_data_dict:
                    target_data_dict[key] = target_data_dict[key][:num_frames]
        else:
            num_frames = num_frames_target
            if num_frames_source > num_frames_target:
                 print(f"Warning: Source file has more frames ({num_frames_source}) than target file ({num_frames_target}). Source data will be truncated.")

        # Copy the pelvis data from source to target
        print("Copying pelvis data...")
        target_data_dict['body_positions'][:, target_pelvis_idx] = source_data['body_positions'][:num_frames, source_pelvis_idx]
        target_data_dict['body_rotations'][:, target_pelvis_idx] = source_data['body_rotations'][:num_frames, source_pelvis_idx]
        # target_data_dict['body_linear_velocities'][:, target_pelvis_idx] = source_data['body_linear_velocities'][:num_frames, source_pelvis_idx]
        # target_data_dict['body_angular_velocities'][:, target_pelvis_idx] = source_data['body_angular_velocities'][:num_frames, source_pelvis_idx]
        
        # Save the updated data back to the target file
        np.savez(
            target_file,
            **target_data_dict
        )
        print(f"Successfully updated pelvis data in {target_file}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the file paths are correct.")
    except ValueError:
        print("Error: 'pelvis' not found in one of the files. Cannot proceed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Set to True to print data without modifying files
    DRY_RUN = True  

    # Define the file paths relative to the script's location
    script_dir = os.path.dirname(__file__)
    source_motion_file = os.path.join(script_dir, "G1_dance_old.npz")
    target_motion_file = os.path.join(script_dir, "G1_dance.npz")
    
    update_pelvis_data(source_motion_file, target_motion_file, dry_run=DRY_RUN) 