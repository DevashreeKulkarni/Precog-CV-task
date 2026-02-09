**Devashree Kulkarni**

# Precog CV Task: The Lazy Artist - Neural Network Interpretability 



> "The eye sees only what the mind is prepared to comprehend." — Henri Bergson

This project explores how Convolutional Neural Networks (CNNs) can exploit spurious correlations in datasets and implements various interpretability and debiasing techniques to make models more robust and transparent.

## Table of Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Project Approach](#project-approach)
- [Task Breakdown](#task-breakdown)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)

## Overview

Modern CNNs are powerful but can be "lazy" - they exploit the easiest patterns in data, even if those patterns are spurious correlations. This project deliberately creates a biased Colored-MNIST dataset, trains models that exploit these biases, diagnoses their behavior using interpretability tools, and then implements debiasing strategies to force models to learn proper shape-based features rather than color shortcuts.

## Directory Structure

```
Precog-CV-task/
├── README.md                    # This file
├── CV_task.txt                  # Original task description
├── task0.md                     # Detailed implementation report
├── precog_cv_2.ipynb           # Main Jupyter notebook with all implementations
└── data/                        # MNIST dataset (auto-downloaded)
```

## Dependencies

The project requires the following Python libraries:

### Core Libraries
- **PyTorch** (`torch`, `torchvision`) - Deep learning framework for model building and training
- **NumPy** - Numerical computations and array operations
- **Matplotlib** - Data visualization and plotting
- **Seaborn** - Statistical data visualization
- **Pandas** - Data manipulation and analysis

### Machine Learning & Visualization
- **scikit-learn** (`sklearn`) - Metrics (confusion matrix, etc.)
- **Pillow** (PIL) - Image processing utilities

### Python Version
- Python 3.7+

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/DevashreeKulkarni/Precog-CV-task.git
cd Precog-CV-task
```

2. **Install dependencies:**
```bash
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn pillow
```

Or using a requirements file (if you create one):
```bash
pip install -r requirements.txt
```

3. **Verify CUDA availability (optional, for GPU acceleration):**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Running the Project

### Using Jupyter Notebook

1. **Launch Jupyter:**
```bash
jupyter notebook precog_cv_2.ipynb
```

2. **Run all cells sequentially:**
   - Click `Kernel` → `Restart & Run All`
   - Or run cells individually from top to bottom

### Using VS Code

1. Open the notebook in VS Code with Jupyter extension installed
2. Select a Python kernel
3. Run cells sequentially using the play button or `Shift+Enter`

### Expected Outputs

The notebook will:
- Download MNIST dataset automatically (if not present)
- Generate colored MNIST datasets with biased train and hard test sets
- Train multiple CNN models with different architectures and strategies
- Generate visualizations including:
  - Confusion matrices
  - Grad-CAM heatmaps
  - Filter visualizations
  - Attention maps
  - Feature decompositions
- Print accuracy metrics and analysis results


## Project Approach

### Phase 1: 
1. Create a synthetically biased dataset where color is spuriously correlated with digit class
2. Demonstrate that standard CNNs exploit this bias and fail on debiased test data

### Phase 2: Diagnosis 
1. Implement visualization tools to understand what the network learns
2. Use Grad-CAM (from scratch) to visualize attention patterns
3. Analyze neuron activations and polysemanticity

### Phase 3: Debiasing & Robustness
1. Implement multiple debiasing strategies:
   - Color jittering with consistency loss
   - Focal loss for hard examples
   - Activation suppression
   - Architectural improvements (ShapeCNN)
2. Evaluate robustness against adversarial attacks (FGSM, PGD)

### Phase 4: Feature Decomposition
1. Train Sparse Autoencoders (SAEs) to decompose hidden representations
2. Perform feature interventions
3. Compare lazy vs. robust model representations

## Task Breakdown

### Task 0: The Biased Canvas
**Goal:** Create a Colored-MNIST dataset with spurious correlations

**Implementation:**
- Training set: 95% of digit X has dominant color Y, 5% random colors
- Test set: Inverted correlation - digit X never has dominant color Y
- Color applied to foreground (digit stroke) with textured backgrounds
- 10 colors mapped to 10 digits (0=Red, 1=Green, 2=Blue, etc.)

**Key Code:** `BiasedMNIST` class with `_colorize()` method

### Task 1: The Cheater
**Goal:** Train a baseline CNN that exploits the bias

**Results:**
- Training accuracy: >95%
- Hard test accuracy: <20%
- Confusion matrix shows systematic misclassification based on color

**Architecture:** Standard 3-layer CNN with Conv - Pool - Conv - Pool - Conv - FC layers

### Task 2: The Prober
**Goal:** Visualize what neurons "see" through activation maximization

**Techniques:**
- Filter visualization through gradient ascent
- CSI (Color Selectivity Index) calculation
- Polysemanticity analysis (how many classes activate each neuron)

**Findings:**
- 15+ neurons with CSI > 3.0 (highly color-selective)
- Most neurons are weakly monosemantic (polysemanticity ~0.4-0.5)

### Task 3: The Interrogation
**Goal:** Implement Grad-CAM from scratch to visualize attention

**Implementation:**
- Hook into final convolutional layer
- Compute gradients w.r.t. target class
- Generate weighted activation maps
- Overlay heatmaps on original images

**Results:**
- Biased model spreads attention uniformly (color averaging)
- Debiased model concentrates attention on digit strokes

### Task 4: The Intervention
**Goal:** Retrain models to ignore color and focus on shape

**Methods Implemented:**

1. **Color Jittering + Consistency Loss** 
   - Accuracy: 91% on hard test set
   - Penalizes prediction changes when color is jittered
   
2. **Focal Loss** 
   - Accuracy: 21% (failed)
   - Fights architecture rather than fixing it
   
3. **Activation Suppression** 
   - Accuracy: 36% (failed)
   - Compensatory learning in other neurons
   
4. **ShapeCNN Architecture** 
   - Accuracy: 96% on hard test set
   - 5 conv layers with gradual pooling
   - Makes shape features easier to learn

### Task 5: The Invisible Cloak
**Goal:** Test adversarial robustness of biased vs. debiased models

**Attack Methods:**
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent, 10 steps)
- Epsilon: 0.05 (imperceptible perturbations)

**Surprising Result:**
- Debiased models are MORE vulnerable to adversarial attacks
- FGSM success: 60% (biased) - 90% (debiased)
- PGD success: 70% (biased) - 100% (debiased)
- Hypothesis: Shape features have smoother gradients

### Task 6: The Decomposition
**Goal:** Use Sparse Autoencoders to decompose hidden representations

**Implementation:**
- 4x overcomplete SAE (64 to 256 features)
- L1 sparsity penalty
- 100 epochs training

**Results:**
- 99.97% reconstruction accuracy
- 70% active features (good sparsity)
- Feature interventions show 60+ percentage point prediction shifts
- Counterintuitive finding: Robust models produce MORE specialized features (selectivity 0.90 vs 0.76)

## Results Summary

| Task | Method | Key Metric | Result |
|------|--------|------------|--------|
| Task 1 | Baseline CNN | Hard Test Accuracy | 18% |
| Task 2 | Neuron Analysis | Color-Selective Neurons | 15 with CSI > 3.0 |
| Task 3 | Grad-CAM | Attention Localization | Diffuse (biased) vs. Focused (robust) |
| Task 4 | Color Jittering | Hard Test Accuracy | 91% |
| Task 4 | ShapeCNN | Hard Test Accuracy | 96% |
| Task 5 | Adversarial (PGD) | Attack Success Rate | 70% (biased), 100% (robust) |
| Task 6 | SAE | Reconstruction + Sparsity | 99.97%, 70% active |

## Key Findings

1. **Spurious Correlations are Easy Shortcuts:** CNNs will exploit color-digit correlations when available, achieving high training accuracy but failing on distribution shifts.

2. **Grad-CAM Reveals Reasoning:** Visualization shows biased models spread attention uniformly (averaging color) while robust models focus on digit strokes.

3. **Debiasing Trade-offs:** 
   - Color jittering works well (91% accuracy)
   - Architectural changes work best (96% accuracy)
   - Loss-based methods can fail catastrophically

4. **Adversarial Vulnerability Paradox:** Robust models trained on shape features are more vulnerable to adversarial attacks, suggesting smoother gradient landscapes.

5. **Feature Specialization:** Surprisingly, robust models develop MORE specialized features despite being trained to ignore spurious correlations.





Repository: [Precog-CV-task](https://github.com/DevashreeKulkarni/Precog-CV-task)

---
