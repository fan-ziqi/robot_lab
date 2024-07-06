import os
import sys
from pathlib import Path

"""This script can be used to rename the template project to a new project name.
It renames all the occurrences of real_robot (in files, directories, etc.) to the new project name.
"""


def rename_file_contents(root_dir_path: str, old_name: str, new_name: str, exclude_dirs: list = []):
    """Rename all instances of the old keyword to the new keyword in all files in the root directory.

    Args:
        root_dir_path (str): The root directory path.
        old_name (str): The old keyword to replace.
        new_name (str): The new keyword to replace with.
    """
    for dirpath, _, files in os.walk(root_dir_path):
        if any(exclude_dir in dirpath for exclude_dir in exclude_dirs):
            continue
        for file_name in files:
            with open(os.path.join(dirpath, file_name)) as file:
                file_contents = file.read()
            file_contents = file_contents.replace(old_name, new_name)
            with open(os.path.join(dirpath, file_name), "w") as file:
                file.write(file_contents)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_template.py <new_name>")
        sys.exit(1)

    root_dir_path = str(Path(__file__).resolve().parent.parent)
    old_name = "real_robot"
    new_name = sys.argv[1]

    print(f"Warning, this script will rename all instances of '{old_name}' to '{new_name}' in {root_dir_path}.")
    proceed = input("Proceed? (y/n): ")

    if proceed.lower() == "y":
        # rename the real_robot folder
        os.rename(
            os.path.join(root_dir_path, "exts", "real_robot", "real_robot"),
            os.path.join(root_dir_path, "exts", "real_robot", new_name),
        )
        os.rename(os.path.join(root_dir_path, "exts", "real_robot"), os.path.join(root_dir_path, "exts", new_name))
        # rename the file contents
        rename_file_contents(root_dir_path, old_name, new_name, exclude_dirs=[".git"])
    else:
        print("Aborting.")
