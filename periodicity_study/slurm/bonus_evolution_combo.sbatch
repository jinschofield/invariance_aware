#!/bin/bash
#SBATCH --job-name=bonus_evolution_combo
#SBATCH --output=slurm_logs/bonus_evolution_combo_%j.out
#SBATCH --error=slurm_logs/bonus_evolution_combo_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -e

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p slurm_logs

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

if ! python -c "import ti" 2>/dev/null; then
    pip install -r requirements.txt
    pip install -e .
fi

export CUDA_VISIBLE_DEVICES=0

COMBO=${COMBO:-0}
ENV_ID=${ENV_ID:-periodicity}
REP=${REP:-crtr_learned}
POLICY=${POLICY:-rep}
OUT_DIR=${OUT_DIR:-periodicity_study/outputs_sweep/comb_${COMBO}}
EVERY=${EVERY:-2}
MAX_FRAMES=${MAX_FRAMES:-50}
DIFF_THRESHOLD=${DIFF_THRESHOLD:-1e-4}
EXTRA_ARGS=${EXTRA_ARGS:-""}

echo "Combo: ${COMBO}"
echo "Env: ${ENV_ID}"
echo "Rep: ${REP}"
echo "Policy: ${POLICY}"
echo "Output: ${OUT_DIR}"

python -m periodicity_study.bonus_evolution_combo \
  --combo "${COMBO}" \
  --env "${ENV_ID}" \
  --rep "${REP}" \
  --policy "${POLICY}" \
  --output-dir "${OUT_DIR}" \
  --device cuda:0 \
  --every "${EVERY}" \
  --max-frames "${MAX_FRAMES}" \
  --diff-threshold "${DIFF_THRESHOLD}" \
  ${EXTRA_ARGS}
