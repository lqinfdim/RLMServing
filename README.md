<h1 align="center">RLM Serving</h1>

<p align="center">
  <strong>Official code for ICLR 2026 paper "Reasoning Language Model Inference Serving Unveiled: An Empirical Study"</strong>
</p>

<p align="center">
  <a href="https://lqinfdim.github.io/project/rllm-serving/index.html">
    <img src="https://img.shields.io/badge/Project-Homepage-2ea44f?style=flat-square&logo=githubpages&logoColor=white" alt="Project Homepage" />
  </a>
  <a href="https://github.com/lqinfdim/RLMServing">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=flat-square&logo=github&logoColor=white" alt="Code" />
  </a>
  <a href="https://arxiv.org/abs/2510.18672">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv" />
  </a>
  <a href="https://openreview.net/forum?id=6CGjZYp6ft">
    <img src="https://img.shields.io/badge/OpenReview-Forum-5C2D91?style=flat-square&logo=openaccess&logoColor=white" alt="OpenReview" />
  </a>
</p>

## Project Status

- [x] Core benchmark pipeline is available.
- [x] Paper links and citation are available.
- [x] Clean and release pilot study scripts for RLLM serving.
- [ ] Add complete reproduction guide with exact commands and expected outputs.

## Paper

- **Title**: `Reasoning Language Model Inference Serving Unveiled: An Empirical Study`
- **Authors**: `Qi Li, Junpan Wu, Xiang Liu, Yuxin Wang, Zeyu Li, Zhenheng Tang, Yuhan Chen, Shaohuai Shi, Xiaowen Chu`
- **Venue**: `ICLR 2026`
- **Project Homepage**: [RLM Serving Homepage](https://lqinfdim.github.io/project/rllm-serving/index.html)
- **Code**: [GitHub Repository](https://github.com/lqinfdim/RLMServing)
- **arXiv**: [arXiv:2510.18672](https://arxiv.org/abs/2510.18672)
- **OpenReview**: [OpenReview Discussion](https://openreview.net/forum?id=6CGjZYp6ft)

## Environment

This project uses Conda for the base Python environment and `pip` for most Python packages.

### 1. Install Conda

Install Miniconda or Anaconda first.

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
```

If the environment already exists:

```bash
conda env update -f environment.yml
```

### 3. Activate the environment

```bash
conda activate rllm
```

### 4. Install Python packages with pip

```bash
pip install -r req.txt
```

`environment.yml` only contains Conda packages. `req.txt` contains the Python packages installed by `pip`.

## Pilot Study Scripts (RLLM + vLLM)

The cleaned pilot scripts are in `pilot-study/` and are intended for public release.

### Script responsibilities

- `pilot-study/start-vllm-service.sh`: starts and stops `vllm serve`, then orchestrates dataset-level benchmark runs.
- `pilot-study/benchmark-rllm.sh`: runs the actual benchmark by calling `main.py` over datasets and token budgets.
- `pilot-study/7b-rllm.sh`: preset entrypoint for DeepSeek-R1-Distill-Qwen-7B.
- `pilot-study/14b-rllm.sh`: preset entrypoint for DeepSeek-R1-Distill-Qwen-14B.
- `pilot-study/32b-rllm.sh`: preset entrypoint for DeepSeek-R1-Distill-Qwen-32B.

### How to run

From repository root:

```bash
bash pilot-study/7b-rllm.sh
bash pilot-study/14b-rllm.sh
bash pilot-study/32b-rllm.sh
```

### Environment behavior for pilot scripts

- Default mode uses `conda run -n <your_env> ...` inside scripts.
- You do not need to manually run `conda activate` in default mode.
- Set environment variable `RLLM_CONDA_ENV` or pass `--conda_env <your_env>`.
- Set environment variable `RLLM_API_KEY` or pass `--api_key <your_key>`.
- You can disable conda wrapping with `--use_conda_run false` and then run in your currently active environment.

### Outputs

- Server logs: `pilot-study/logs/server/`
- Benchmark raw outputs: `pilot-study/outputs/`
- Benchmark CSV logs: `pilot-study/logs/`

### Citation (BibTeX)

```bibtex
  @inproceedings{
              rllm-serving,
              title={Reasoning Language Model Inference Serving Unveiled: An Empirical Study},
              author={Li, Qi and Wu, Junpan and Liu, Xiang and Wang, Yuxin and Li, Zeyu and Tang, Zhenheng and Chen, Yuhan and Shi, Shaohuai and Chu, Xiaowen},
              booktitle={The Fourteenth International Conference on Learning Representations},
              year={2026},
              url={https://openreview.net/forum?id=6CGjZYp6ft}
}
```
