#!/usr/bin/env bash
# Sync skrl and rsl_rl train/play scripts from Isaac Lab releases and reapply local tweaks.
set -euo pipefail

fetch() {
  local url="$1"
  local dest="$2"
  echo "Fetching ${url} to ${dest}"
  curl -fsSL "${url}" -o "${dest}"
}

# Sync for skrl
SKRL_BASE_URL="https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/scripts/reinforcement_learning/skrl"
SKRL_DEST_DIR="scripts/reinforcement_learning/skrl"
SKRL_FILES=(train.py play.py)

echo "Syncing skrl scripts..."
for name in "${SKRL_FILES[@]}"; do
  url="${SKRL_BASE_URL}/${name}"
  dest_path="${SKRL_DEST_DIR}/${name}"
  fetch "${url}" "${dest_path}"
done

# Sync for rsl_rl
RSL_RL_BASE_URL="https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/scripts/reinforcement_learning/rsl_rl"
RSL_RL_DEST_DIR="scripts/reinforcement_learning/rsl_rl"
RSL_RL_FILES=(train.py play.py cli_args.py)

echo "Syncing rsl_rl scripts..."
for name in "${RSL_RL_FILES[@]}"; do
  url="${RSL_RL_BASE_URL}/${name}"
  dest_path="${RSL_RL_DEST_DIR}/${name}"
  fetch "${url}" "${dest_path}"
done

echo "Done."
