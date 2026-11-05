#!/usr/bin/env bash
set -euo pipefail
QBRIDGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QBRIDGE_EXEC="${QBRIDGE_PYTHON:-$QBRIDGE_ROOT/.venv/bin/python}"
if [[ ! -x "$QBRIDGE_EXEC" ]]; then
    echo "Create this artifact's .venv or set QBRIDGE_PYTHON to an existing compatible Python." >&2
    exit 2
fi
export PYTHONPATH="$QBRIDGE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR="$QBRIDGE_ROOT/reruns/matplotlib-cache"
cd "$QBRIDGE_ROOT"
exec "$QBRIDGE_EXEC" -B "$@"
