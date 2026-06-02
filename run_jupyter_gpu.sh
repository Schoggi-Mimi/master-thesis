#!/bin/bash
set -eo pipefail

REPO_ROOT="$(pwd)"

PARTITION="gpu"
GPU_REQUEST="gpu:rtx4090:1"
CPUS="16"
MEM="88G"
TIME="12:00:00"
PORT="8888"
CONDA_ENV="thesis"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "============================================================"
    echo "Requesting interactive GPU node with Slurm"
    echo "============================================================"
    echo "Partition:    ${PARTITION}"
    echo "GPU request:  ${GPU_REQUEST}"
    echo "CPUs:         ${CPUS}"
    echo "Memory:       ${MEM}"
    echo "Time:         ${TIME}"
    echo "Repo root:    ${REPO_ROOT}"
    echo "============================================================"
    echo "Waiting for GPU allocation..."
    echo ""

    exec srun \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --gres="${GPU_REQUEST}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --time="${TIME}" \
        --pty "$0" --inside-allocation
fi

cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo "GPU allocation active"
echo "============================================================"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Repo root:    $(pwd)"
echo "Port:         ${PORT}"
echo "============================================================"
echo ""

if command -v module >/dev/null 2>&1; then
    module load Anaconda3 || true
fi

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Could not find conda."
    echo "Try: which conda"
    exit 1
fi

export MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-}
conda activate "${CONDA_ENV}"

echo "Conda environment active: ${CONDA_ENV}"
echo "Python: $(which python)"
python --version
echo ""

echo "GPU sanity check:"
nvidia-smi || true
echo ""

python -c 'import torch; print("PyTorch CUDA available:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'

echo ""
echo "============================================================"
echo "Starting Jupyter Lab"
echo "============================================================"
echo "Copy one of the Jupyter URLs into VS Code:"
echo ""
echo "  Command Palette"
echo "  -> Jupyter: Specify Jupyter Server for Connections"
echo "  -> Existing"
echo "  -> paste URL with token"
echo ""
echo "Expected URL examples:"
echo "  http://127.0.0.1:${PORT}/lab?token=..."
echo "  http://$(hostname):${PORT}/lab?token=..."
echo "============================================================"
echo ""

jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --notebook-dir="${REPO_ROOT}"
