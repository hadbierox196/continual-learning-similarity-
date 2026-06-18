
# Inter-Task Representational Overlap Predicts the EWC–Replay Performance Gap

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![OSF DOI](https://img.shields.io/badge/OSF-pink.svg)](https://doi.org/10.17605/OSF.IO/KFHZW)
[![ZENODO DOI](https://img.shields.io/badge/Zenodo-green.svg)](https://pytorch.org/)

**Preprint:** [DOI coming soon]  


---

## Overview

This repository contains the complete code for a class‑incremental continual learning study that investigates whether inter‑task representational similarity modulates the relative performance of Elastic Weight Consolidation (EWC) and Experience Replay (ER). The experimental design crosses 4 methods × 3 similarity conditions × 10 seeds, totalling 120 runs.

**Primary metric:** Backward Transfer (BWT) under the Class‑IL scenario.  
**Mechanistic measure:** Linear Centered Kernel Alignment (CKA) of layer‑2 activations.

| Method | Description |
|--------|-------------|
| Fine‑tuning | Sequential training, no forgetting protection (lower bound) |
| EWC | Elastic Weight Consolidation with diagonal Fisher |
| ER | Experience Replay with reservoir sampling |
| Joint Training | All tasks simultaneously (upper bound) |

| Similarity Condition | Description |
|----------------------|-------------|
| Low | Digit pairs with minimal pixel‑space cosine similarity |
| Medium | Standard Split‑MNIST ordering |
| High | Digit pairs with maximal pixel‑space cosine similarity |

---

## Repository Structure

```

continual-learning-similarity/
├── src/                         # Core modules
│   ├── data.py                  # Split‑MNIST pipeline
│   ├── models.py                # MLP (784→256→256→10)
│   ├── metrics.py               # BWT, accuracy matrix
│   ├── ewc.py                   # EWC with temperature‑scaled Fisher
│   ├── replay.py                # Reservoir buffer
│   ├── similarity.py            # Task pairing design
│   ├── cka.py                   # Linear CKA
│   └── experiment.py            # run_experiment() + checkpointing
├── notebooks/                   # Sequential experiment scripts
│   ├── 01_baselines.py
│   ├── 02_ewc.py
│   ├── 03_experience_replay.py
│   ├── 04_similarity_design.py
│   ├── 05_cka.py
│   ├── 06_full_factorial.py
│   ├── 07_analysis.py
│   └── reproduce_results.py
├── manuscript/
│   └── manuscript_template.md   # Paper template
├── results/                     # Generated outputs
│   ├── raw/                     # 120 .pkl checkpoints (not committed)
│   └── figures/                 # Publication‑ready figures
├── requirements.txt
└── README.md

```

---

## Setup

### Google Colab (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/YOUR_USERNAME/continual-learning-similarity
%cd /content/drive/MyDrive/continual-learning-similarity
!pip install -r requirements.txt
```

Set runtime to GPU (T4) before running any experiments.

Local

```bash
git clone https://github.com/YOUR_USERNAME/continual-learning-similarity
cd continual-learning-similarity
pip install -r requirements.txt
```

---

Running the Experiments

Execute the notebooks in order (01 through 07). Each module depends on outputs from the previous weeks. A full run takes about 3.5 hours total (split across multiple Colab sessions). Use the checkpoint system to resume after interruptions.

Key validation gates:

· Week 1: Fine‑tuning BWT ∈ [‑0.60, ‑0.30]; Joint accuracy ≥ 0.95.
· Week 2: Optimal EWC λ > 0; Fisher gradients non‑zero.
· Week 3: Buffer contains all 10 classes; ER improves over fine‑tuning.
· Week 4: ANOVA p < 0.05 across similarity conditions; mean similarities ordered Low < Medium < High.
· Week 5: All CKA unit tests pass (identical, random, rotated).
· Week 6: All 120 runs complete; ANOVA table generated.
· Week 7: All four figures produced.

To regenerate all figures from saved checkpoints without re‑running experiments, execute notebooks/reproduce_results.py (runtime ≈ 5 minutes).

---

Key Hyperparameters

Parameter Value
Architecture MLP 784→256→256→10
Optimiser Adam, lr = 1e‑3
Epochs per task 5
EWC λ Determined by Week 2 sweep (optimal ≈ 50000)
EWC temperature 5.0
ER buffer size Determined by Week 3 sweep (optimal ≈ 5000)
ER replay ratio 1:1
Fisher samples 1000
CKA probe size 500 (100 per task)
Seeds 0–9

---

Results (Summary)

Method Low BWT Medium BWT High BWT
Fine‑tuning -0.0163 ± 0.0001 -0.0120 ± 0.0005 +0.3858 ± 0.0017
EWC -0.1326 ± 0.0414 -0.0077 ± 0.0796 +0.5266 ± 0.0516
ER -0.1427 ± 0.0521 +0.0590 ± 0.0179 +0.5599 ± 0.0235
Joint -0.0057 ± 0.0014 -0.0023 ± 0.0041 +0.0033 ± 0.0068

The Method × Similarity interaction is significant (F(6,108) = 236.42, p < 0.001).
CKA correlates with BWT across all runs (Pearson r = -0.4998, p = 6.18×10⁻⁹).

Full figures and ANOVA table are generated in Week 7.

---

Data Availability

Raw checkpoints (120 .pkl files) are archived on OSF: .
They can be used with reproduce_results.py to regenerate all figures and tables.

---


}
```

---

License

MIT License. See LICENSE for details.

