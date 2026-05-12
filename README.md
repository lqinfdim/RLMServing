# rllm-serving

<p align="left">
  <a href="https://lqinfdim.github.io/project/rllm-serving/index.html">
    <img src="https://img.shields.io/badge/Project-Homepage-2ea44f?style=flat-square&logo=githubpages&logoColor=white" alt="Project Homepage" />
  </a>
  <a href="https://arxiv.org/abs/2510.18672">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv" />
  </a>
  <a href="https://openreview.net/forum?id=6CGjZYp6ft">
    <img src="https://img.shields.io/badge/OpenReview-Forum-5C2D91?style=flat-square&logo=openaccess&logoColor=white" alt="OpenReview" />
  </a>
</p>

## Paper

- **Title**: `Reasoning Language Model Inference Serving Unveiled: An Empirical Study`
- **Authors**: `<AUTHOR_1>, <AUTHOR_2>, <AUTHOR_3>, ...`
- **Venue**: `ICLR 2026`
- **Project Homepage**: `https://lqinfdim.github.io/project/rllm-serving/index.html`
- **arXiv**: `https://arxiv.org/abs/2510.18672`
- **OpenReview**: `https://openreview.net/forum?id=6CGjZYp6ft`

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
