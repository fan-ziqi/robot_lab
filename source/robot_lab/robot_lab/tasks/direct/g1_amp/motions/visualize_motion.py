"""
Motion Data Visualization Tool

This script visualizes motion data from npz files, creating interactive plots for:
- Joint positions
- Joint velocities
- Body linear velocities
- Body angular velocities

Usage:
    python visualize_motion.py --file <motion_file.npz>

Arguments:
    --file    Path to the npz file containing motion data

Example:
    python visualize_motion.py --file G1_dance.npz

Output:
    Generates an HTML file with interactive plots in the same directory as the input file.
    The output filename will be <input_filename>_visualization.html
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import os

def visualize_motion(npz_file):
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    
    # Get basic information
    fps = data['fps'].item() if 'fps' in data else 60
    dof_names = data['dof_names'] if 'dof_names' in data else []
    body_names = data['body_names'] if 'body_names' in data else []
    
    # Create subplot layout
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=('Root Link 3D Position (pelvis)',
                       'Joint Positions (DOF Positions)', 
                       'Joint Velocities (DOF Velocities)',
                       'Body Linear Velocities',
                       'Body Angular Velocities'),
        vertical_spacing=0.05,
        specs=[[{"type": "scatter"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]],
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )
    
    # Time axis
    time = np.arange(len(data['dof_positions'])) / fps if 'dof_positions' in data else np.arange(len(data['body_linear_velocities'])) / fps
    
    # 1. Plot root link 3D position
    if 'body_positions' in data and 'pelvis' in body_names:
        root_idx = body_names.tolist().index('pelvis')
        root_positions = data['body_positions'][:, root_idx, :]
        
        # X component
        fig.add_trace(
            go.Scatter(
                x=time,
                y=root_positions[:, 0],
                name="Pelvis Position (X)",
                mode='lines',
                line=dict(width=1)
            ),
            row=1, col=1
        )
        # Y component
        fig.add_trace(
            go.Scatter(
                x=time,
                y=root_positions[:, 1],
                name="Pelvis Position (Y)",
                mode='lines',
                line=dict(width=1)
            ),
            row=1, col=1
        )
        # Z component
        fig.add_trace(
            go.Scatter(
                x=time,
                y=root_positions[:, 2],
                name="Pelvis Position (Z)",
                mode='lines',
                line=dict(width=1)
            ),
            row=1, col=1
        )
    
    # 2. Plot all joint positions
    if 'dof_positions' in data:
        for i in range(len(dof_names)):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data['dof_positions'][:, i],
                    name=dof_names[i],
                    mode='lines',
                    line=dict(width=1)
                ),
                row=2, col=1
            )
    
    # 3. Plot all joint velocities
    if 'dof_velocities' in data:
        for i in range(len(dof_names)):
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=data['dof_velocities'][:, i],
                    name=dof_names[i],
                    mode='lines',
                    line=dict(width=1)
                ),
                row=3, col=1
            )
    
    # 4. Plot all body linear velocities (X, Y, Z components)
    if 'body_linear_velocities' in data:
        for i in range(len(body_names)):
            velocities = data['body_linear_velocities'][:, i, :]
            # X component
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=velocities[:, 0],
                    name=f"{body_names[i]} (X)",
                    mode='lines',
                    line=dict(width=1)
                ),
                row=4, col=1
            )
            # Y component
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=velocities[:, 1],
                    name=f"{body_names[i]} (Y)",
                    mode='lines',
                    line=dict(width=1)
                ),
                row=4, col=1
            )
            # Z component
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=velocities[:, 2],
                    name=f"{body_names[i]} (Z)",
                    mode='lines',
                    line=dict(width=1)
                ),
                row=4, col=1
            )
    
    # 5. Plot all body angular velocities (X, Y, Z components)
    if 'body_angular_velocities' in data:
        for i in range(len(body_names)):
            velocities = data['body_angular_velocities'][:, i, :]
            # X component
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=velocities[:, 0],
                    name=f"{body_names[i]} (X)",
                    mode='lines',
                    line=dict(width=1)
                ),
                row=5, col=1
            )
            # Y component
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=velocities[:, 1],
                    name=f"{body_names[i]} (Y)",
                    mode='lines',
                    line=dict(width=1)
                ),
                row=5, col=1
            )
            # Z component
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=velocities[:, 2],
                    name=f"{body_names[i]} (Z)",
                    mode='lines',
                    line=dict(width=1)
                ),
                row=5, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Motion Data Visualization (Complete Data)",
        height=2000,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            font=dict(size=8)
        )
    )
    
    # Update axis labels
    for i in range(1, 6):
        fig.update_xaxes(title_text="Time (seconds)", row=i, col=1)
    
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Position (radians)", row=2, col=1)
    fig.update_yaxes(title_text="Velocity (rad/s)", row=3, col=1)
    fig.update_yaxes(title_text="Linear Velocity (m/s)", row=4, col=1)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", row=5, col=1)
    
    # Generate output file path automatically
    input_dir = os.path.dirname(os.path.abspath(npz_file))
    input_filename = os.path.splitext(os.path.basename(npz_file))[0]
    output_html = os.path.join(input_dir, f"{input_filename}_visualization.html")
    
    # Save visualization results
    fig.write_html(output_html)
    print(f"Visualization results saved to: {output_html}")

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    
    # Calculate Euler angles
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(2 * (w*y - z*x))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    
    return np.array([roll, pitch, yaw])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize motion data from npz file")
    parser.add_argument("--file", type=str, required=True, help="Path to the npz file")
    args = parser.parse_args()
    
    visualize_motion(args.file) 