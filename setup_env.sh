#!/usr/bin/env bash
# Provision a dual-framework dev venv at the canonical path.
#
# Usage: ./setup_env.sh {isaaclab|mjlab}
#
# Creates:
#   .venvs/isaaclab — Python 3.11, isaaclab[isaacsim,all]
#   .venvs/mjlab    — Python 3.12, mjlab
#
# Idempotent: re-running with an existing venv skips creation and re-runs the install.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FW="${1:-}"

case "$FW" in
  isaaclab) PY=3.11 ;;
  mjlab)    PY=3.12 ;;
  *)
    echo "Usage: $0 {isaaclab|mjlab}" >&2
    exit 1
    ;;
esac

VENV="$ROOT/.venvs/$FW"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "[INFO] Creating $FW venv at $VENV (Python $PY)..."
  uv venv --python "$PY" --seed "$VENV"
fi

echo "[INFO] Installing robot_lab[$FW] into $VENV..."
uv pip install --python "$VENV/bin/python" --upgrade pip

# IsaacLab needs uv-side resolver flags that match pyproject.toml's [tool.uv]:
# - --prerelease=allow: isaacsim ships with pre-release markers (5.1.0.dev0)
# - --overrides: pin pywin32 to Windows-only since isaacsim-core lacks the marker
INSTALL_FLAGS=()
if [ "$FW" = "isaaclab" ]; then
  OVERRIDES_FILE="$ROOT/.venvs/.uv-overrides.txt"
  printf 'pywin32 ; sys_platform == "win32"\n' > "$OVERRIDES_FILE"
  INSTALL_FLAGS+=(--prerelease=allow --overrides "$OVERRIDES_FILE")
fi

uv pip install --python "$VENV/bin/python" "${INSTALL_FLAGS[@]}" -e "$ROOT/source/robot_lab[$FW]"

# IsaacLab's PyPI wheel ships all 6 subpackages (isaaclab_rl, isaaclab_tasks,
# isaaclab_assets, isaaclab_mimic, isaaclab_contrib) as source under
# site-packages/isaaclab/source/<pkg>/, but only exposes `isaaclab` itself at
# the site-packages top level. Drop a .pth file so Python adds those
# subpackage source dirs to sys.path on startup, making them importable
# without a separate clone+editable-install of IsaacLab.
if [ "$FW" = "isaaclab" ]; then
  SITE_PACKAGES="$VENV/lib/python$PY/site-packages"
  PTH="$SITE_PACKAGES/_isaaclab_subpackages.pth"
  ISAACLAB_SOURCE="$SITE_PACKAGES/isaaclab/source"
  if [ -d "$ISAACLAB_SOURCE" ]; then
    echo "[INFO] Registering IsaacLab subpackages via $PTH..."
    : > "$PTH"
    for sub in isaaclab_rl isaaclab_tasks isaaclab_assets isaaclab_mimic isaaclab_contrib; do
      if [ -d "$ISAACLAB_SOURCE/$sub" ]; then
        echo "$ISAACLAB_SOURCE/$sub" >> "$PTH"
      fi
    done
  else
    echo "[WARN] Expected IsaacLab subpackage source at $ISAACLAB_SOURCE but it's missing." >&2
  fi
fi

echo "[INFO] Done. Activate with: source $VENV/bin/activate"
echo "[INFO] Or rely on scripts/{train,play}.py which auto-exec under $VENV/bin/python."
