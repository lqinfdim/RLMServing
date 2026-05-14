# pilot-study

Flat scripts for 7B/14B/32B reasoning-model benchmark runs with vLLM.

Files:
- `7b-rllm.sh`
- `14b-rllm.sh`
- `32b-rllm.sh`
- `start-vllm-service.sh`
- `benchmark-rllm.sh`

Run:
```bash
bash pilot-study/7b-rllm.sh
bash pilot-study/14b-rllm.sh
bash pilot-study/32b-rllm.sh
```

Environment behavior:
- Default mode: `conda run -n <your_env> ...` inside scripts.
- No manual `conda activate` required by default.
- Set environment variable `RLLM_CONDA_ENV` or pass `--conda_env <your_env>`.
- You can disable conda wrapper by passing `--use_conda_run false`.
- API key is shared by server/client via `RLLM_API_KEY` or `--api_key`.

Outputs:
- Server logs: `pilot-study/logs/server/`
- Benchmark outputs: `pilot-study/outputs/`
- Benchmark csv: `pilot-study/logs/`

Example:
```bash
export RLLM_CONDA_ENV=my_env_name
export RLLM_API_KEY=my_secret_key
bash pilot-study/7b-rllm.sh
```
