# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil


def clean_trash(folder_path):
    """
    Delete folders that meet the following conditions:
    1. Contain `events.out.*` files.
    2. Also meet:
       - No `.pt` files, or
       - Less than 3 `.pt` files.

    :param folder_path: Target folder path
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # List of folders to delete
    folders_to_delete = []

    # Traverse subfolders
    for root, dirs, files in os.walk(folder_path):
        # Regex match `events.out.*` and `.pt` files
        event_pattern = re.compile(r"events\.out.*")
        model_pattern = re.compile(r"model_\d+\.pt")

        event_files = [f for f in files if event_pattern.match(f)]
        model_files = [f for f in files if model_pattern.match(f)]

        # Check delete conditions: have `events.out.*` files, but no `.pt` files or less than 3 `.pt` files
        if event_files and (len(model_files) < 3):
            folders_to_delete.append(os.path.abspath(root))

    # If there are folders that meet the conditions, prompt and delete
    if folders_to_delete:
        print("The following folders contain `events.out.*` and meet the deletion conditions, they will be deleted:")
        for folder in folders_to_delete:
            print(f"  - {folder}")

        # Confirm deletion
        confirm = input("Confirm deletion of these folders? (y/n): ").strip().lower()
        if confirm == "y":
            for folder in folders_to_delete:
                shutil.rmtree(folder)  # Delete the entire folder
                print(f"Deleted: {folder}")
            print("All folders that meet the conditions have been deleted.")
        else:
            print("Deletion operation canceled.")
    else:
        print("No folders meet the conditions, no need to delete.")


# Example call
if __name__ == "__main__":
    # folder = input("Please enter the target folder path: ").strip()
    folder = "logs"
    clean_trash(folder)
