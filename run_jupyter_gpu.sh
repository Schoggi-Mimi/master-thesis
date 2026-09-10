#!/bin/bash
#SBATCH --job-name=jlab
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/storage/homefs/cn21m021/logs/jlab_%j.out
#SBATCH --error=/storage/homefs/cn21m021/logs/jlab_%j.out

# ============ fixed variable block ============
USER_NAME="cn21m021"
LOGIN_NODE="submit03.unibe.ch"
CONDA_ENV="thesis"
REPO_ROOT="/storage/homefs/cn21m021/projects/master-thesis"
LOGDIR="/storage/homefs/cn21m021/logs"
# ==============================================

# ---------- launcher path, runs on the login node ----------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    set -eo pipefail
    mkdir -p "$LOGDIR"

    JID=$(sbatch --parsable "$0")
    LOG="${LOGDIR}/jlab_${JID}.out"
    echo "submitted job ${JID}"
    echo "log ${LOG}"

    printf "waiting for allocation "
    while true; do
        ST=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
        [ -z "$ST" ] && { echo; echo "job left the queue, check ${LOG}"; exit 1; }
        [ "$ST" = "RUNNING" ] && break
        printf "."
        sleep 10
    done
    echo

    GNODE=$(squeue -j "$JID" -h -o "%N")
    printf "waiting for jupyter "
    while ! { [ -s "$LOG" ] && grep -q "token=" "$LOG"; }; do printf "."; sleep 5; done
    echo

    PORT=$(awk '/^PORT:/{print $2; exit}' "$LOG")
    TOKEN=$(grep -m1 -oE 'token=[a-f0-9]+' "$LOG" | cut -d= -f2)

    echo
    echo "============================================================"
    echo "JOB      ${JID}   NODE ${GNODE}   ENDS in $(squeue -j "$JID" -h -o "%L")"
    echo
    echo "URL"
    echo "  http://${GNODE}:${PORT}/lab?token=${TOKEN}"
    echo
    echo "TUNNEL, run on your laptop"
    echo "  ssh -N -L ${PORT}:${GNODE}:${PORT} -J ${USER_NAME}@${LOGIN_NODE} ${USER_NAME}@${GNODE}"
    echo
    echo "LOCAL URL after the tunnel is up"
    echo "  http://127.0.0.1:${PORT}/lab?token=${TOKEN}"
    echo
    echo "STOP  scancel ${JID}"
    echo "============================================================"
    exit 0
fi

# ---------- compute path, runs on the gnode ----------
set -eo pipefail
GNODE=$(hostname -s)
PORT=$(( 8000 + SLURM_JOB_ID % 1000 ))

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

echo "NODE: ${GNODE}"
echo "PORT: ${PORT}"
nvidia-smi
python -c 'import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))'

cd "${REPO_ROOT}"
exec jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}" \
     --ServerApp.allow_remote_access=True \
     --notebook-dir="${REPO_ROOT}"

# submit: ./run_jupyter_gpu.sh