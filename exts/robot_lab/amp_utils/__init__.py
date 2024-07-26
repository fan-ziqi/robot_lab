"""Package containing asset and sensor configurations."""

import os

##
# Configuration for different assets.
##

ROBOT_LAB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../robot_lab/"))
AMP_UTILS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
"""Path to the extension data directory."""
