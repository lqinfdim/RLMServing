#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATASET="gsm8k"
URL="127.0.0.1"
PORT="9001"
RUN_NAME="pilot_study"
BATCH_SIZE="32"
MAX_TOKEN_NUMS="256,512,1024,2048,4096,8192,16384,20480"
CONDA_ENV_NAME="${RLLM_CONDA_ENV:-}"
USE_CONDA_RUN="true"
API_KEY="${RLLM_API_KEY:-rllm-key}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_token_nums) MAX_TOKEN_NUMS="$2"; shift 2 ;;
    --conda_env) CONDA_ENV_NAME="$2"; shift 2 ;;
    --use_conda_run) USE_CONDA_RUN="$2"; shift 2 ;;
    --api_key) API_KEY="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ "$USE_CONDA_RUN" == "true" && -z "$CONDA_ENV_NAME" ]]; then
  echo "[ERROR] Missing conda env name. Set RLLM_CONDA_ENV or pass --conda_env." >&2
  exit 1
fi

IFS=',' read -r -a TOKEN_ARRAY <<< "$MAX_TOKEN_NUMS"
PYTHON_CMD=(python "${ROOT_DIR}/main.py")
if [[ "$USE_CONDA_RUN" == "true" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found but --use_conda_run=true" >&2
    exit 1
  fi
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV_NAME" python "${ROOT_DIR}/main.py")
fi

for max_token_num in "${TOKEN_ARRAY[@]}"; do
  "${PYTHON_CMD[@]}" \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET" \
    --run_name "$RUN_NAME" \
    --url "$URL" \
    --port "$PORT" \
    --max_token_num "$max_token_num" \
    --batch_size "$BATCH_SIZE" \
    --serving_framework vllm \
    --is_reasoning_llm True \
    --api_key "$API_KEY" \
    --output_path "${ROOT_DIR}/pilot-study/outputs/benchmark_output" \
    --csv_path "${ROOT_DIR}/pilot-study/logs/benchmark_results.csv"
done
