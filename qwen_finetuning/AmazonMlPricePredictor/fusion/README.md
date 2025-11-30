# Fusion Models

Multi-modal fusion models combining different embedding sources for price prediction.

## Package Structure

```
fusion/
├── __init__.py           # Package exports (shared components)
├── loss_functions.py     # Custom loss functions (SMAPE variants, Huber, etc.)
├── utils.py              # Shared utilities (checkpointing, metrics, etc.)
│
├── qwen_clip/            # Two-branch baseline model
│   ├── config.py         # Configuration
│   ├── model.py          # CLIP image + Qwen text fusion
│   ├── train.py          # Training script
│   ├── evaluate_by_groups.py  # Evaluation
│   ├── predict_test.py   # Test predictions
│   └── README.md         # Detailed documentation
│
└── all_embeddings/       # Four-branch best model
    ├── config.py         # Configuration
    ├── model.py          # CLIP + SigLIP + Qwen fusion
    ├── train.py          # Training script
    ├── evaluate_by_groups.py  # Evaluation
    ├── predict_test.py   # Test predictions
    └── README.md         # Detailed documentation
```

## Models

### 1. `qwen_clip/` - Two-Branch Baseline
**Embeddings:** CLIP image (768-dim) + Qwen text (1027-dim)  
**Performance:** ~35% SMAPE  
**Parameters:** 7.7M  

Simple two-branch fusion with concatenation and MLP regression head.

### 2. `all_embeddings/` - Four-Branch Best Model
**Embeddings:** CLIP + SigLIP images + SigLIP + Qwen text  
**Performance:** ~30-33% SMAPE  
**Parameters:** 13.5M  

Hierarchical fusion with cross-attention and mixture-of-experts:
- Stage 1: Image-to-image fusion (CLIP ↔ SigLIP)
- Stage 2: Text-to-text fusion (Qwen ↔ SigLIP)
- Stage 3: Cross-modal fusion (images ↔ texts)
- Stage 4: MoE regression (3 experts)

## Shared Components

### Loss Functions (`loss_functions.py`)

- **WeightedSMAPELoss** - Tiered weights by price range (10x for $0-5)
- **FocalSMAPELoss** - Quadratic penalty for high-error samples
- **HybridFocalSMAPELoss** - Combined Huber + SMAPE
- **WeightedHuberLoss** - Sample-specific weighting
- And more...

### Utilities (`utils.py`)

- `set_seed()` - Reproducible random seeds
- `save_checkpoint() / load_checkpoint()` - Model persistence
- `compute_smape()` - SMAPE metric calculation
- `compute_metrics()` - Multiple evaluation metrics
- `EarlyStopping` - Early stopping helper

## Usage

### Training

```bash
# Two-branch baseline
cd fusion/qwen_clip
python -m fusion.qwen_clip.train

# Best model (four-branch)
cd fusion/all_embeddings
python -m fusion.all_embeddings.train
```

### Evaluation

```bash
# Evaluate with detailed breakdown
python -m fusion.qwen_clip.evaluate_by_groups
python -m fusion.all_embeddings.evaluate_by_groups
```

### Predictions

```bash
# Generate test predictions
python -m fusion.qwen_clip.predict_test
python -m fusion.all_embeddings.predict_test
```

## Importing Shared Components

```python
# From within subpackages (relative imports)
from ..loss_functions import get_loss_function
from ..utils import compute_smape, save_checkpoint

# From outside the package
from fusion.loss_functions import WeightedSMAPELoss
from fusion.utils import EarlyStopping
```

## Benefits of This Structure

1. **Proper Python package** - Clean imports, no sys.path hacks
2. **No code duplication** - Shared code at package root
3. **Clear organization** - Related models grouped together
4. **Easy to extend** - Add new fusion variants as subpackages
5. **Standard practice** - Follows Python packaging conventions

## Next Steps

1. Train both models to compare performance
2. Analyze which embeddings contribute most (ablation studies)
3. Ensemble predictions from both models
4. Experiment with different fusion strategies

See individual model READMEs for detailed documentation.

