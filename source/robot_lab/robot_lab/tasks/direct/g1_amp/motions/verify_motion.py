"""
Motion Data Verification Tool

This script verifies and displays detailed information about motion data stored in npz files.
It shows information about:
- FPS and duration
- File size and name
- Data structure and content
- Sample data for each type of motion data

Usage:
    python verify_motion.py --file <motion_file.npz>

Arguments:
    --file    Path to the npz file containing motion data

Example:
    python verify_motion.py --file G1_dance.npz

Output:
    Prints detailed information about the motion data file, including:
    - Basic file information (size, name)
    - FPS and duration
    - Data structure overview
    - Sample data for each type of motion data
"""

import numpy as np
import argparse
import os
np.set_printoptions(threshold=100, precision=3, suppress=True)

def verify_motion_file(npz_file):
    # Load npz file
    data = np.load(npz_file, allow_pickle=True)
    
    # First display FPS information
    if 'fps' in data:
        fps = data['fps'].item()
        print(f"\nFPS: {fps}")
        if 'dof_positions' in data:
            duration = data['dof_positions'].shape[0] / fps
            print(f"Motion duration: {duration:.2f} seconds")
    else:
        print("\nWarning: 'fps' key not found!")
    
    # Get file size
    file_size = os.path.getsize(npz_file) / (1024 * 1024)  # Convert to MB
    print(f"\nFile Information:")
    print(f"Filename: {os.path.basename(npz_file)}")
    print(f"File size: {file_size:.2f} MB")
    
    print("\nData Content Overview:")
    print("-" * 50)
    for key in data:
        arr = data[key]
        print(f"\n{key}:")
        print(f"  Data type: {arr.dtype}")
        print(f"  Data shape: {arr.shape}")
        
        # Display content based on different data types
        if key == 'dof_names':
            print(f"  Joint names:")
            for i, name in enumerate(arr):
                print(f"    {i+1}. {name}")
        elif key == 'body_names':
            print(f"  Body part names:")
            for i, name in enumerate(arr):
                print(f"    {i+1}. {name}")
        elif key == "body_positions":
            print(f"  Body positions sample (first 2 frames, first 3 parts):")
            print(arr[:2, :3, :])  # Show first 2 frames, first 3 parts positions
        elif key == "body_rotations":
            print(f"  Body rotations sample (first 2 frames, first 3 parts, quaternions):")
            print(arr[:2, :3, :])  # Show first 2 frames, first 3 parts quaternion rotations
        elif key == "body_linear_velocities":
            print(f"  Body linear velocities sample (first 2 frames, first 3 parts):")
            print(arr[:2, :3, :])  # Show first 2 frames, first 3 parts linear velocities
        elif key == "body_angular_velocities":
            print(f"  Body angular velocities sample (first 2 frames, first 3 parts):")
            print(arr[:2, :3, :])  # Show first 2 frames, first 3 parts angular velocities
        elif key == "dof_positions":
            print(f"  Joint positions sample (first 2 frames, first 5 joints):")
            print(arr[:2, :5])  # Show first 2 frames, first 5 joints positions
        elif key == "dof_velocities":
            print(f"  Joint velocities sample (first 2 frames, first 5 joints):")
            print(arr[:2, :5])  # Show first 2 frames, first 5 joints velocities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify and display motion data file contents")
    parser.add_argument("--file", type=str, required=True, help="Path to the npz file")
    args = parser.parse_args()
    
    verify_motion_file(args.file)
