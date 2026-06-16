# Inter-Task Representational Overlap Predicts the EWC–Replay Performance Gap

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Preprint:** [arXiv link — add after submission]  
**Status:** Under review at [Journal name — add after submission]

---

## Overview

This repository contains the complete code for a class-incremental continual
learning study examining whether inter-task representational similarity
moderates the performance difference between Elastic Weight Consolidation
(EWC) and Experience Replay (ER).

**Research question:**  
*Does inter-task representational overlap predict the EWC–Replay performance
gap, and can representational drift (measured via CKA) explain why?*

**Factorial design:**  
4 methods × 3 similarity conditions × 10 seeds = **120 experiments**

| Method | Description |
|---|---|
| Fine-tuning | Sequential training, no forgetting protection (lower bound) |
| EWC | Elastic Weight Consolidation (Kirkpatrick et al., 2017) |
| ER | Experience Replay with reservoir sampling (Rolnick et al., 2019) |
| Joint Training | All tasks simultaneously (upper bound) |

| Similarity Condition | Description |
|---|---|
| Low | Digit pairs with minimal pixel-space cosine similarity |
| Medium | Standard Split-MNIST ordering (van de Ven & Tolias, 2019) |
| High | Digit pairs with maximal pixel-space cosine similarity |

**Primary metric:** Backward Transfer (BWT) — Class-IL protocol, shared 10-class head  
**Mechanistic measure:** Linear CKA between layer-2 activations after task 0 vs. task 4

---

## Repository Structure

```
continual-learning-similarity/
├── src/                         # Reusable Python modules
│   ├── data.py                  # Split-MNIST pipeline (all 3 conditions)
│   ├── models.py                # MLP: 784→256→256→10
│   ├── metrics.py               # BWT, accuracy matrix (Class-IL)
│   ├── ewc.py                   # Online EWC with diagonal Fisher IM
│   ├── replay.py                # Reservoir buffer for experience replay
│   ├── similarity.py            # Cosine similarity + task pairing design
│   ├── cka.py                   # Linear CKA (Kornblith et al., 2019)
│   └── experiment.py            # run_experiment() + checkpoint management
├── notebooks/                   # Step-by-step experiment scripts
│   ├── 01_baselines.py          # Week 1: Fine-tuning & joint training
│   ├── 02_ewc.py                # Week 2: EWC lambda sweep
│   ├── 03_experience_replay.py  # Week 3: ER buffer size sweep
│   ├── 04_similarity_design.py  # Week 4: Task pairing + ANOVA validation
│   ├── 05_cka.py                # Week 5: CKA validation + drift measurement
│   ├── 06_full_factorial.py     # Week 6: 120-run factorial experiment
│   ├── 07_analysis.py           # Week 7: Publication-quality figures
│   └── reproduce_results.py     # Regenerate all figures from checkpoints
├── manuscript/
│   └── manuscript_template.md   # Paper template (fill in results)
├── results/
│   ├── raw/                     # Checkpoint .pkl files (120 files after Week 6)
│   └── figures/                 # All output figures (PNG + SVG)
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Option A — Google Colab (recommended for GPU access)

**Step 1:** Open [Google Colab](https://colab.research.google.com/)

**Step 2:** In the first cell of any notebook, run:
```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive')

# Clone the repository (first time only)
!git clone https://github.com/YOUR_USERNAME/continual-learning-similarity
os.chdir('continual-learning-similarity')
```

**Step 3:** Install dependencies:
```python
!pip install torch==2.1.0 torchvision==0.16.0 numpy==1.24.4 \
             matplotlib==3.7.2 seaborn==0.12.2 scipy==1.11.2 \
             pandas==2.0.3 statsmodels==0.14.0 tqdm==4.66.1
```

**Step 4:** Verify GPU:
```python
import torch
print(torch.cuda.is_available())  # Should print True on a GPU runtime
```

> **Runtime tip:** Go to Runtime → Change runtime type → GPU (T4) before starting.

### Option B — Local Installation

```bash
git clone https://github.com/YOUR_USERNAME/continual-learning-similarity
cd continual-learning-similarity
pip install -r requirements.txt
```

---

## Step-by-Step Run Instructions

Run notebooks **in order**. Each week builds on the previous one.  
Each notebook is a `.py` script — copy the entire file content into a Colab
notebook, or paste each `# %%` section into a separate cell.

---

### WEEK 1 — Baselines (`notebooks/01_baselines.py`)

**What it does:** Trains the fine-tuning lower bound and joint training upper
bound. Validates the Class-IL data pipeline against published numbers.

**Run:** Copy `01_baselines.py` into a Colab notebook. Run all cells.

**Expected output:**
- `results/figures/baseline_accuracy_matrices.png`
- Fine-tuning BWT printed to console (expect −0.45 to −0.60)
- Joint accuracy printed to console (expect ≥ 0.95)

**Validation gate:**  
✓ Fine-tuning BWT ∈ [−0.60, −0.30] — confirms forgetting  
✓ Joint accuracy ≥ 0.90 — confirms model/training correct  
If either fails, **stop** and fix the data pipeline before proceeding.

---

### WEEK 2 — EWC Lambda Sweep (`notebooks/02_ewc.py`)

**What it does:** Sweeps λ ∈ {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0} and
selects the best regularisation strength. Validates Fisher statistics.

**Run:** Copy `02_ewc.py`. Run all cells. (~45 min on T4)

**Expected output:**
- `results/figures/ewc_lambda_sweep.png` — U-shaped BWT vs. log(λ) curve
- `results/ewc_best_lambda.json` — saved automatically

**Validation gate:**  
✓ Best-λ EWC BWT exceeds fine-tuning BWT by ≥ 0.10  
✓ Fisher `frac_near_zero` < 0.20 (printed to console)

---

### WEEK 3 — ER Buffer Sweep (`notebooks/03_experience_replay.py`)

**What it does:** Sweeps buffer sizes {100, 500, 1000, 2000, 5000} and
validates reservoir sampling class balance.

**Run:** Copy `03_experience_replay.py`. Run all cells. (~60 min on T4)

**Expected output:**
- `results/figures/er_buffer_sweep.png`
- `results/er_best_config.json` — saved automatically

**Validation gate:**  
✓ All 10 digit classes present in buffer after task 5  
✓ ER with 2000 samples improves BWT vs. fine-tuning

---

### WEEK 4 — Similarity Design (`notebooks/04_similarity_design.py`)

**What it does:** Computes the 10×10 cosine similarity matrix, finds optimal
Low and High task pairings, validates conditions with one-way ANOVA.

**Run:** Copy `04_similarity_design.py`. Run all cells. (~10 min)

**Expected output:**
- `results/figures/digit_similarity_matrix.png`
- `results/figures/similarity_conditions_validation.png`
- `results/task_pairings.json` ← **required by all subsequent notebooks**

**Validation gate:**  
✓ ANOVA p < 0.05 across three conditions  
✓ Mean similarity: Low < Medium < High  
✓ Gap (High − Low) ≥ 0.05  
If ANOVA fails, see **Pivot Trigger 1** in the roadmap.

---

### WEEK 5 — CKA Validation (`notebooks/05_cka.py`)

**What it does:** Runs three unit tests on the CKA implementation, then
measures representational drift under fine-tuning across all three conditions.

**Run:** Copy `05_cka.py`. Run all cells. (~20 min)

**Expected output:**
- `results/cka_validation.txt` — all three tests must say PASS
- `results/figures/cka_finetune_conditions.png`

**Validation gate:**  
✓ CKA(identical) = 1.000 ± 0.001  
✓ CKA(random) < 0.05  
✓ CKA(rotated) = CKA(identical) ± 0.001  
**Do not proceed to Week 6 if any test fails.**

---

### WEEK 6 — Full Factorial (`notebooks/06_full_factorial.py`)

**What it does:** Runs all 120 experiments (4 methods × 3 conditions × 10
seeds). Saves each result immediately as a `.pkl` checkpoint. Computes
the two-way ANOVA on BWT.

**Session management:**
- Run **25 experiments per Colab session** (~2 hrs)
- At the start of each new session, re-run the Setup cell and the
  `check_completion()` cell — it will resume exactly where you left off
- Expect **4–5 sessions** over the week

**Run sequence per session:**
1. Cell 1: Setup + Drive mount
2. Cell 2: Load best hyperparameters
3. Cell 3: Check completion status
4. Cell 4: Run experiment batch (change `BATCH_SIZE` to 25)
5. Cell 5: Build summary table (run only when all 120 are done)
6. Cell 6: ANOVA (run only when all 120 are done)

**Expected output:**
- 120 `.pkl` files in `results/raw/`
- `results/bwt_summary_table.csv`
- `results/interaction_anova.txt`

---

### WEEK 7 — Analysis (`notebooks/07_analysis.py`)

**What it does:** Loads all 120 results and generates the four publication
figures plus the Pearson correlation between CKA and BWT.

**Run:** Copy `07_analysis.py`. Run all cells. (~5 min)

**Expected output:**
- `results/figures/fig1_bwt_comparison.png` + `.svg`
- `results/figures/fig2_cka_drift.png` + `.svg`
- `results/figures/fig3_bwt_cka_scatter.png` + `.svg`
- `results/figures/fig4_interaction.png` + `.svg`

**After this notebook:** Fill in `manuscript/manuscript_template.md` with
your actual numbers and send the results to your AI assistant to complete
the manuscript.

---

### Reproduce All Figures (`notebooks/reproduce_results.py`)

Regenerates all four figures from saved `.pkl` files in a fresh Colab
session without re-running any experiments. Runtime ≈ 5 minutes.

```python
# In a fresh Colab session:
from google.colab import drive
drive.mount('/content/drive')
import os; os.chdir('/content/drive/MyDrive/continual-learning-similarity')
# Then run reproduce_results.py top to bottom
```

---

## Key Hyperparameters

| Parameter | Value | Set in |
|---|---|---|
| Architecture | 784→256→256→10 MLP | `src/models.py` |
| Optimizer | Adam, lr=1e-3 | `src/experiment.py` |
| Epochs per task | 10 | `src/experiment.py` |
| EWC λ | Set by Week 2 sweep | `results/ewc_best_lambda.json` |
| ER buffer size | Set by Week 3 sweep | `results/er_best_config.json` |
| ER replay ratio | 1:1 (current:replay) | `src/experiment.py` |
| Fisher samples | 1000 | `src/experiment.py` |
| CKA probe size | 500 (100/task × 5) | `src/data.py` |
| Seeds | 0–9 (10 per cell) | `src/experiment.py` |
| Gradient clipping | L2 norm ≤ 1.0 | `src/experiment.py` |

---

## Citing This Work

```bibtex
@article{AUTHOR_YEAR,
  title   = {Inter-Task Representational Overlap Predicts the EWC–Replay
             Performance Gap in Class-Incremental Learning},
  author  = {YOUR NAME},
  journal = {JOURNAL NAME},
  year    = {YEAR},
  doi     = {DOI},
  url     = {https://github.com/YOUR_USERNAME/continual-learning-similarity}
}
```

---

## References

- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural
  networks. *PNAS*, 114(13), 3521–3526.
- Rolnick, D. et al. (2019). Experience replay for continual learning.
  *NeurIPS 32*.
- van de Ven, G. M. & Tolias, A. S. (2019). Three scenarios for continual
  learning. *arXiv:1904.07734*.
- Kornblith, S. et al. (2019). Similarity of neural network representations
  revisited. *ICML*. arXiv:1905.00414.
- McClelland, J. L. et al. (1995). Why there are complementary learning systems
  in the hippocampus and neocortex. *Psychological Review*, 102(3), 419–457.

---

## License

MIT License. See `LICENSE` for details.
