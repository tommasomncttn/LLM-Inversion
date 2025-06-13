# LLM-Inversion: SIP-It

## Overview

**SIP-It** (Sequential Inverse Prompt via Iterative Updates) is a novel algorithm for *prompt inversion*—the task of recovering original input sequences from internal hidden states of decoder-only Large Language Models (LLMs). Unlike prior heuristic or output-based methods, SIP-It achieves **exact recovery** using a provably efficient, gradient-guided approach that leverages the sequential structure of LLMs.

This method has implications for both model **interpretability** and **privacy**, demonstrating that prompts can be reconstructed even from intermediate activations. We validate our method on the [TinyStories model](https://huggingface.co/roneneldan/TinyStories-1M), showing superior efficiency compared to exhaustive and baseline optimization-based approaches.

## Features

- 🔄 **Exact inversion** of LLM hidden states to original prompts
- ⚡ **Linear-time complexity** relative to sequence length and vocabulary
- 📉 **Gradient-based optimization** with theoretical guarantees
- 📊 **Extensive experiments** on meaningful and random sequences

---

## Reproducing Results

### Step 1: Set up your environment

Create a Python 3.10 virtual environment and install dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Run the experiments

Execute all notebooks located in the `./notebooks` directory sequentially:

1. Open each notebook
2. Click **"Run All"**

[!WARNING]
Full execution may take **several days** depending on your hardware. For quicker inspection, we recommend running on reduced prompt lengths or fewer test cases.

---

## Notebooks Structure

- `0.0-dataset.ipynb` – Creation of `meaningful` and `random` datasets
- `1.1-.ipynb` – 
- `1.2-SIP-It-experiments.ipynb` – Execution of SIP-It algorithm for different parameters and data
- `2.1-SIP-It-plots.ipynb` – Creates the plots used in the report

---

## Acknowledgements

This project was developed as part of the **Opt4ML** course at EPFL.  
Based on experiments using the [TinyStories](https://arxiv.org/abs/2305.07759) dataset.
