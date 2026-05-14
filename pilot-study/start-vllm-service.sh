#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PORT="9001"
GPU_DEVICES="0,1"
TP_SIZE="2"
GPU_MEMORY_UTILIZATION="0.9"
MAX_MODEL_LEN="32768"
RUN_NAME="pilot_study"
SERVER_NOTE="open-source"
API_HOST="127.0.0.1"
WAIT_TIME="480"
BATCH_SIZE="32"
MAX_TOKEN_NUMS="256,512,1024,2048,4096,8192,16384,20480"
DATASETS="gsm8k,math500,aime24,gpqa"
CONDA_ENV_NAME="${RLLM_CONDA_ENV:-}"
USE_CONDA_RUN="true"
API_KEY="${RLLM_API_KEY:-rllm-key}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id) MODEL_ID="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu_devices) GPU_DEVICES="$2"; shift 2 ;;
    --tp_size) TP_SIZE="$2"; shift 2 ;;
    --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max_model_len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --run_name) RUN_NAME="$2"; shift 2 ;;
    --server_note) SERVER_NOTE="$2"; shift 2 ;;
    --api_host) API_HOST="$2"; shift 2 ;;
    --wait_time) WAIT_TIME="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_token_nums) MAX_TOKEN_NUMS="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
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

find_server_pid() {
  local port="$1"
  local model_id="$2"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | awk '!seen[$0]++')
  fi
  if [[ -z "$pids" ]] && command -v ss >/dev/null 2>&1; then
    pids=$(ss -H -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | awk '!seen[$0]++')
  fi
  for pid in $pids; do
    if [[ -r "/proc/$pid/cmdline" ]]; then
      local cmdline
      cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
      if [[ "$cmdline" == *"vllm"* && "$cmdline" == *"$model_id"* && "$cmdline" == *"--port $port"* ]]; then
        echo "$pid"
        return 0
      fi
    fi
  done
  [[ -n "$pids" ]] && echo "$pids" | head -n 1 && return 0
  return 1
}

start_server() {
  if [[ "$USE_CONDA_RUN" == "true" ]] && ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda not found but --use_conda_run=true" >&2
    return 1
  fi
  local ts model_filename log_dir log_file
  ts=$(date +'%Y%m%d_%H%M%S')
  model_filename=${MODEL_ID//\//_}
  log_dir="${ROOT_DIR}/pilot-study/logs/server/${model_filename}"
  log_file="${log_dir}/${ts}_${model_filename}.log"
  mkdir -p "$log_dir"
  {
    echo "run_name=$RUN_NAME"
    echo "note=$SERVER_NOTE"
    echo "model_id=$MODEL_ID"
    echo "gpu_devices=$GPU_DEVICES"
    echo "port=$PORT"
    echo "tp_size=$TP_SIZE"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "log_time=$ts"
    echo
  } > "$log_file"

  local -a serve_cmd
  serve_cmd=(vllm serve "$MODEL_ID" --dtype bfloat16 --api-key "$API_KEY" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --tensor-parallel-size "$TP_SIZE" --max-model-len "$MAX_MODEL_LEN" --enable-reasoning --reasoning-parser deepseek_r1 --task generate --host "$API_HOST" --port "$PORT")

  if [[ "$USE_CONDA_RUN" == "true" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" conda run --no-capture-output -n "$CONDA_ENV_NAME" "${serve_cmd[@]}" >> "$log_file" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "${serve_cmd[@]}" >> "$log_file" 2>&1 &
  fi

  local bg_pid server_pid
  bg_pid=$!
  echo "[INFO] server log: $log_file" >&2
  sleep "$WAIT_TIME"
  server_pid=$(find_server_pid "$PORT" "$MODEL_ID" || true)
  if [[ -z "$server_pid" ]]; then
    kill "$bg_pid" 2>/dev/null || true
    wait "$bg_pid" 2>/dev/null || true
    echo "" 
    return 1
  fi
  echo "$server_pid $bg_pid"
}

stop_server() {
  local server_pid="$1"
  local bg_pid="$2"
  kill -TERM "$server_pid" 2>/dev/null || true
  sleep 2
  kill -KILL "$server_pid" 2>/dev/null || true
  kill -TERM "$bg_pid" 2>/dev/null || true
  wait "$bg_pid" 2>/dev/null || true
}

read -r server_pid bg_pid < <(start_server) || {
  echo "[ERROR] failed to start server" >&2
  exit 1
}

IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"
for dataset in "${DATASET_ARRAY[@]}"; do
  bash "${SCRIPT_DIR}/benchmark-rllm.sh" \
    --model_name "$MODEL_ID" \
    --dataset "$dataset" \
    --run_name "$RUN_NAME" \
    --url "$API_HOST" \
    --port "$PORT" \
    --batch_size "$BATCH_SIZE" \
    --max_token_nums "$MAX_TOKEN_NUMS" \
    --conda_env "$CONDA_ENV_NAME" \
    --use_conda_run "$USE_CONDA_RUN" \
    --api_key "$API_KEY"
done

stop_server "$server_pid" "$bg_pid"
