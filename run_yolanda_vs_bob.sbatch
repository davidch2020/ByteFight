#!/bin/bash
#SBATCH -J yolanda_bob
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o Output-%j.out

set -euo pipefail

# Allow overrides at submit time, e.g.
# sbatch --export=ALL,AGENT_A=YolandaCppAccel,AGENT_B=YolandaV8,NUM_GAMES=20,NUM_WORKERS=16 run_yolanda_vs_bob.sbatch
AGENT_A="${AGENT_A:-YolandaCurrent}"
AGENT_B="${AGENT_B:-BobExperiment}"
NUM_GAMES="${NUM_GAMES:-40}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"

cd "$SLURM_SUBMIT_DIR"

# Use a real Python module on PACE so extension builds can find Python.h.
# Try a few likely module names instead of hard-coding one exact string.
load_pace_python() {
  local mod
  for mod in python/3.12.5 python/3.12 python/3.11.9 python/3.11 python/3.10.10 python/3.10 python; do
    if module load "$mod" 2>/dev/null; then
      echo "Loaded Python module: $mod"
      return 0
    fi
  done

  echo "Warning: could not load a named Python module; falling back to existing python3"
  return 0
}

load_pace_python

# Refresh the current Yolanda snapshot only if one of the requested agents
# uses the YolandaCurrent alias.
if [ "$AGENT_A" = "YolandaCurrent" ] || [ "$AGENT_B" = "YolandaCurrent" ]; then
  rm -rf "3600-agents/YolandaCurrent"
  cp -R "3600-agents/Yolanda" "3600-agents/YolandaCurrent"
fi

echo "Running $AGENT_A vs $AGENT_B"
echo "Games: $NUM_GAMES"
echo "Workers: $NUM_WORKERS"
echo "SLURM cpus-per-task: ${SLURM_CPUS_PER_TASK:-unset}"

if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="$SLURM_SUBMIT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "Python: $($PYTHON_BIN --version)"
echo "Python path: $(command -v "$PYTHON_BIN")"

build_native_ext() {
  local agent_name="$1"
  local agent_dir="3600-agents/$agent_name"
  local ext_suffix
  local include_dir
  local cxx

  # Only build agents that actually ship native extension source files.
  if [ -f "$agent_dir/setup.py" ] && [ -f "$agent_dir/yolanda_search_ext.cpp" ]; then
    echo "Building native backend for $agent_name"
    rm -f "$agent_dir"/yolanda_search_ext*.so "$agent_dir"/yolanda_search_ext*.pyd

    if "$PYTHON_BIN" -c "import setuptools" >/dev/null 2>&1; then
      (
        cd "$agent_dir"
        # Build the .so with the same Python that will run the match.
        "$PYTHON_BIN" setup.py build_ext --inplace
      )
    else
      echo "setuptools not available; using direct compiler fallback"
      ext_suffix=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
      include_dir=$("$PYTHON_BIN" -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY') or '')")
      cxx="${CXX:-g++}"

      if [ -z "$include_dir" ]; then
        echo "Could not determine Python include directory for native build"
        exit 1
      fi

      "$cxx" -O3 -Wall -shared -std=c++17 -fPIC \
        -I"$include_dir" \
        "$agent_dir/yolanda_search_ext.cpp" \
        -o "$agent_dir/yolanda_search_ext${ext_suffix}"
    fi
  fi
}

# Build native extensions for the agents we are about to test.
build_native_ext "$AGENT_A"
if [ "$AGENT_B" != "$AGENT_A" ]; then
  build_native_ext "$AGENT_B"
fi

"$PYTHON_BIN" run_many_A.py "$AGENT_A" "$AGENT_B" -n "$NUM_GAMES" -w "$NUM_WORKERS"
