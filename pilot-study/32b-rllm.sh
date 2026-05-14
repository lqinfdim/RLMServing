#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/start-vllm-service.sh" \
  --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --port "9009" \
  --gpu_devices "0,1,2,3,4,5,6,7" \
  --tp_size "8" \
  --gpu_memory_utilization "0.9" \
  --max_model_len "32768" \
  --run_name "pilot_study" \
  --api_host "127.0.0.1" \
  --wait_time "480" \
  --batch_size "32" \
  --max_token_nums "256,512,1024,2048,4096,8192,16384,20480" \
  --datasets "gsm8k,math500,aime24,gpqa" \
  --use_conda_run "true" \
  "$@"
