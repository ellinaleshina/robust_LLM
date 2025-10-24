# LoRA Perturbation Analysis

## Overview

Analysis of non-semantic input perturbations in LLM fine-tuning using Meta-Llama 3.1 (3B, Instruct) adapted with LoRA on Natural Instructions Task 065. The project investigates how typos, Unicode variations, and formatting changes affect hidden representations and develops effective debiasing methods.

## Key Findings

### Perturbation Structure
- **Low-rank mean shift**: Effective rank r≈1-2 (not coordinate-sparse)
- **Subspace concentration**: Energy dispersed across low-dimensional subspace
- **Partial overlap**: Perturbation directions partially overlap with class-informative features

### Optimal Debiasing Pipeline
1. **Shift-aware centering**: Global mean subtraction (λ=1)
2. **Soft low-rank PCA**: Small rank (r=2) with mild attenuation (α=0.05-0.20)
3. **Coordinate masking**: refinement (k/D≤0.05)

### Performance
- **Robustness gain**: +1.0-1.5% perturbed accuracy (70.0% → 71.0-71.5%)
- **Clean performance**: Preserved at 90.1% (no degradation)

## Project Structure

```
├── delta_lora/
│   ├── collect_delta.py          # Extract clean/perturbed hidden states
│   └── train_lora.py              # LoRA fine-tuning script
│
├── extract_data/                  # Data preparation utilities
│
├── metrics/
│   ├── eval_lora_shiftaware.py   # Evaluate debiasing methods
│   └── metrics.py                 # Accuracy calculations
│
└── pipeline/
    ├── llm_pert.py                # Perturbation generation logic
    └── make_perturbations.py      # Apply perturbations to dataset
```

## Method Details

### Default Configuration
- **Target layer**: `layers.9.mlp.up_proj`
- **Shift-aware**: Global demeaning (λ=1)
- **PCA**: Rank r=2, attenuation α=0.08
- **Masking** (optional): k/D=0.002-0.05

### Why This Works
1. Perturbations concentrate in 1-2 dominant directions
2. Partial overlap with task subspace requires **soft** (not hard) projection
3. Mean shift removal handles global bias component
4. Sequential processing: global → low-rank → localized corrections

## Results

### Accuracy Comparison

| Method | Perturbed Acc | Δ from Baseline | Clean Acc |
|--------|---------------|-----------------|-----------|
| Baseline | 70.02% | - | 90.12% |
| + PCA (α=0.08) | 71.04% | +1.02% | 90.12% |
| + UCB mask (k/D=0.05) | 71.55% | +1.53% | 90.12% |

### PCA Attenuation Sweep (rank r=2)

| α | Perturbed Acc | Δ |
|---|---------------|---|
| 0.05 | 70.87% | +0.85% |
| 0.08 | 71.04% | +1.02% |
| 0.10 | 71.04% | +1.02% |
| 0.20 | 70.70% | +0.68% |

### Coordinate Masking Results

| k/D | Mask Type | Perturbed Acc | Δ |
|-----|-----------|---------------|---|
| 0.002 | UCB | 71.38% | +1.36% |
| 0.005 | UCB | 71.38% | +1.36% |
| 0.010 | UCB | 71.38% | +1.36% |
| 0.050 | UCB | 71.55% | +1.53% |

*All masking results shown with PCA (r=2, α=0.08) baseline*

## Usage

### Run Evaluation
```bash
python metrics/eval_lora_shiftaware.py
```

### Customize Parameters
Edit the evaluation script to test different configurations:
- PCA rank (r)
- Attenuation factor (α)
- Masking sparsity (k/D)
- Target layer

## Technical Details

### Perturbation Types
- Zero-width characters
- Unicode confusables
- Layout/markup wrappers
- Keyboard typos
- LLM-generated natural typos

### Diagnostic Metrics
- **Covariance spectrum**: Eigenvalue decay analysis (Figure 1)
- **Support stability**: Jaccard similarity of top-k coordinates (Figure 3)
- **Near-zero fraction**: Distribution of coordinate magnitudes (Figure 2)

### Key Observations
1. **Common low-rank shift**: Clean→pert difference lies in nearly the same low-dimensional subspace
2. **Not coordinate-wise "spiky"**: Zeroing individual coordinates yields negligible gains
3. **Shift-aware first**: Subtracting average shift removes substantial portion of effect
4. **Soft projection (PCA)**: Hard removal (α≈1) hurts; mild attenuation (α≈0.1-0.2) improves robustness
5. **Masks as finisher**: UCB/TUCB masking gives small extra gains on top of low-rank corrector

## Installation

### Requirements
```bash
pip install torch transformers datasets peft numpy scikit-learn
```

### Model
- Meta-Llama-3.1-3B-Instruct
- LoRA adaptation on Natural Instructions Task 065

## Citation

```bibtex
@techreport{lora_perturbation_analysis_2025,
  title={Analysis of Non-Semantic Perturbations in LoRA Fine-Tuning},
  year={2025},
  month={October}
}
```

## License

[Skoltech]

## Contact

allina.aleshina@gmail.com
