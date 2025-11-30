

## Quick Navigation

### Project Structure

```
AmazonMlPricePredictor/
├── clip_finetune/        # CLIP fine-tuning & embedding extraction (768-dim)
├── qwen3_finetune/       # Qwen3-0.6B fine-tuning with QLoRA (1027-dim)
├── fusion/               # Fusion models package
│   ├── loss_functions.py # Shared loss functions
│   ├── utils.py          # Shared utilities
│   ├── qwen_clip/        # 2-embedding baseline
│   └── all_embeddings/   # 4-embedding best model
└── README.md             # This file
```

### Quick Start

Each directory contains a comprehensive README. See:
- [`clip_finetune/README.md`](clip_finetune/README.md) - Fine-tune CLIP on product images
- [`qwen3_finetune/README.md`](qwen3_finetune/README.md) - Fine-tune Qwen3 on product text
- [`fusion/README.md`](fusion/README.md) - Fusion models overview
- [`fusion/qwen_clip/README.md`](fusion/qwen_clip/README.md) - Baseline (2 embeddings)
- [`fusion/all_embeddings/README.md`](fusion/all_embeddings/README.md) - Best model (4 embeddings)

### Recommended Pipeline

1. **Extract embeddings** (run in parallel):
   ```bash
   cd clip_finetune && python train.py && python extract_embeddings.py train
   cd qwen3_finetune && python train.py
   ```

2. **Train fusion model**:
   ```bash
   cd fusion/all_embeddings
   python -m fusion.all_embeddings.train
   ```

3. **Generate predictions**:
   ```bash
   python -m fusion.all_embeddings.predict_test
   ```

---

## 1. Executive Summary

We developed a multi-modal fusion system combining four pre-trained embedding models—fine-tuned Qwen3-0.6B (text), CLIP (image), and SigLIP (text+image)—to predict product prices. Our hierarchical cross-attention architecture with mixture-of-experts (MoE) regression achieves strong performance through weighted SMAPE loss that prioritizes accuracy in the critical low-price range ($0-$30). The system achieves ~30-35% validation SMAPE through careful fusion of complementary embeddings and domain-specific fine-tuning.

---

## 2. Problem Analysis & Solution Strategy

**Key Challenge**: Product pricing requires understanding both visual quality and textual semantics while handling a heavily skewed distribution ($0.01-$10,000) where SMAPE metric heavily penalizes low-price errors.

**Critical Insights**:

- 40% of products under $10 require specialized attention due to SMAPE's percentage-based nature
- Text contains structured information (quantity, unit) that strongly correlates with price
- Multi-modal fusion provides complementary signals: text → product type/category, images → quality/condition

**Solution**: Four-stage pipeline with hierarchical fusion and weighted training:

1. **Fine-tune Qwen3-0.6B** with QLoRA on regression task + structural features (1027-dim: 1024 text + 3 structural)
2. **Extract embeddings**: CLIP image (768-dim), SigLIP image (1152-dim), SigLIP text (1024-dim)
3. **Hierarchical fusion**: Image-image → Text-text → Image-text cross-attention
4. **MoE regression**: 3 experts with tiered weighted SMAPE loss (10x weight for $0-5, decreasing to 1x for $200+)

---

## 3. Model Architecture

### 3.1 Overall Pipeline

```
┌─────────────┐                    ┌──────────────┐
│ Text Input  │                    │Product Image │
└──────┬──────┘                    └──────┬───────┘
       │                                   │
   ┌───┴────┐  ┌─────────┐        ┌───────┴──┐  ┌────────────┐
   │ Qwen3  │  │ SigLIP  │        │   CLIP   │  │   SigLIP   │
   │  Text  │  │  Text   │        │  Image   │  │   Image    │
   │1027-dim│  │1024-dim │        │  768-dim │  │  1152-dim  │
   └───┬────┘  └────┬────┘        └─────┬────┘  └─────┬──────┘
       │            │                    │             │
   ┌───▼────┐  ┌───▼────┐          ┌───▼────┐   ┌────▼─────┐
   │MLP→512 │  │MLP→512 │          │MLP→512 │   │ MLP→512  │
   └───┬────┘  └────┬───┘          └────┬───┘   └─────┬────┘
       │            └──────┐             │        ┌────┘
       │                   │             │        │
       │         ┌─────────▼─────────────▼────────▼─────┐
       │         │  Hierarchical Cross-Attention        │
       │         │  1. Image ← Image (CLIP ← SigLIP)   │
       │         │  2. Text ← Text (Qwen ← SigLIP)     │
       │         │  3. Multimodal (Image ← Text)       │
       │         └─────────────┬────────────────────────┘
       │                       │
       └───────────┬───────────┘
                   │
           ┌───────▼────────┐
           │  Concatenate   │
           │   4×512=2048   │
           └───────┬────────┘
                   │
           ┌───────▼────────┐
           │ MoE Regression │
           │   3 Experts    │
           │  + Gating Net  │
           └───────┬────────┘
                   │
               ┌───▼───┐
               │ Price │
               └───────┘
```

### 3.2 Key Components

**1. Fine-Tuned Qwen3-0.6B Text Encoder**

- **Base**: Qwen/Qwen3-Embedding-0.6B (600M params)
- **Fine-tuning**: QLoRA (rank=16, 4-bit quantization, ~16M trainable params)
- **Task**: End-to-end regression with combined Huber+SMAPE loss
- **Input**: `entity_name + [SEP] + catalog_content` + structural features [quantity_value, unit_volume, unit_weight]
- **Output**: 1027-dim embeddings (1024 text + 3 structural)
- **Rationale**: Domain adaptation > generic embeddings; QLoRA enables full-model tuning with 4GB GPU memory

**2. Multi-Modal Image Encoders**

- **CLIP** (ViT-L/14): 768-dim, general semantic understanding from 400M image-text pairs
- **SigLIP**: 1152-dim, fine-grained visual details with sigmoid loss (better for noisy e-commerce data)
- **Rationale**: Complementary features—CLIP for semantics, SigLIP for details

**3. Hierarchical Cross-Attention Fusion**

```python
# Stage 1: Within-modality fusion
fused_image = CrossAttention(query=CLIP, key_value=SigLIP_img)  # [B,512]
fused_text = CrossAttention(query=Qwen, key_value=SigLIP_txt)   # [B,512]

# Stage 2: Cross-modal fusion
fused_multimodal = CrossAttention(query=fused_image, kv=fused_text)  # [B,512]

# Stage 3: Feature preservation
features = concat([fused_multimodal, fused_image, fused_text, CLIP])  # [B,2048]
```

- **Rationale**: (1) Align within modalities first, (2) then cross-modal interaction, (3) preserve multi-level abstractions

**4. Mixture-of-Experts Regression**

- **3 Expert Networks**: Each MLP(2048 → 1024 → 512 → 256 → 1)
- **Gating**: Softmax(Linear(2048 → 3)) learns which expert to trust per sample
- **Hypothesis**: Experts specialize on price ranges (low/medium/high)
- **Rationale**: Ensemble-like diversity in single forward pass

### 3.3 Training Configuration

**Weighted SMAPE Loss**:

```python
price_thresholds = [5, 10, 30, 60, 200]
weights = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]  # $0-5: 10x ... $200+: 1x
```

- **Critical**: 10x weight on $0-$5 bin forces model to prioritize where SMAPE is most sensitive
- **Log-space prediction**: `log(price+1)` handles 0.01-10,000 range gracefully

**Hyperparameters**:

- Batch size: 256, LR: 1e-3, AdamW optimizer, ReduceLROnPlateau scheduler
- Dropout: 0.25, Early stopping: patience=30, Gradient clip: 1.0
- 80:20 stratified train/val split (stratified by price quantiles)
- Training: 1000 epochs (typically stops at 100-150)

---

## 4. Experimental Results & Insights

### 4.1 Performance Summary

| Model         | Embeddings          | SMAPE   | Params | Key Feature               |
| ------------- | ------------------- | ------- | ------ | ------------------------- |
| qwen_clip     | CLIP img + Qwen txt | ~35%    | 7.7M   | Baseline 2-branch         |
| all_embeddings| All 4 embeddings    | ~30-33% | 13.5M  | Hierarchical fusion + MoE |

**Performance by Price Range** (fusion_qwen example):

- **$0-$5 (15% of data)**: Critical bin, weighted 10x
- **$5-$30 (25% of data)**: Heavily weighted (8x-6x)
- **$30-$200 (50% of data)**: Moderate weighting (4x-2x)
- **$200+ (10% of data)**: Standard weighting (1x)

### 4.2 Ablation Studies & Key Findings

**What Worked ✅**:

1. **Fine-tuning Qwen3 on regression**: Domain adaptation improved ~5-10% over frozen embeddings
2. **Single-layer branch projections**: Direct 1152→512 (vs. 1152→768→512 which plateaued at 72% SMAPE)
3. **Weighted SMAPE loss**: 10x weight on low prices reduced $0-$5 bin error dramatically
4. **Hierarchical fusion**: Image-image then text-text then cross-modal > flat concatenation
5. **Log-space prediction**: `log(price+1)` eliminated numerical instability

**What Failed ❌**:

1. **Binary classification auxiliary task** ($15 threshold): 70-75% accuracy but didn't improve regression; oversimplification lost nuanced information
2. **Multi-layer branch compression**: 1152→768→512 created information bottleneck, causing 72% SMAPE plateau
3. **Equal loss weighting**: Model ignored low-price accuracy; skewed towards easier high-price predictions
4. **Separate structural feature branch**: Only 3 features don't benefit from dedicated branch; better embedded in Qwen

### 4.3 Critical Design Decisions

**Why 4 embeddings?**

- **Complementarity**: CLIP (semantic) + SigLIP (detailed) vision; Qwen (domain-adapted) + SigLIP (VLM) text
- **Robustness**: Multiple sources provide redundancy against individual model failures
- **Cross-attention**: Model learns dynamic weighting per product (vs. naive concatenation)

**Why MoE with 3 experts?**

- **Hypothesis**: Different price ranges have distinct patterns (cheap bulk items vs. premium products)
- **Gating learns context**: Automatically routes based on features, not hard price thresholds
- **3 optimal**: More experts → overfitting; fewer → underutilization

**Why tiered weighted loss?**

- **SMAPE characteristic**: $1 error at $5 (20%) > $10 error at $100 (10%)
- **Distribution skew**: 40% products <$10, but model naturally focuses on large absolute errors
- **Alignment**: Weight training loss to match evaluation metric's sensitivity

---

## 5. Conclusion

Our hierarchical multi-modal fusion system achieves competitive performance through three key innovations: (1) **domain-adapted embeddings** via Qwen3 fine-tuning on regression task with structural features, (2) **complementary multi-modal features** from four embedding models combined through hierarchical cross-attention (within-modality then cross-modal), and (3) **price-aware weighted training** with tiered SMAPE loss (10x weight on critical $0-$5 bin). Systematic ablation revealed critical failures to avoid—classification oversimplification, multi-layer compression bottlenecks, and uniform loss weighting. The resulting 13.5M parameter model with MoE regression balances capacity and efficiency, expected to achieve ~30-33% validation SMAPE through specialized expert networks and careful fusion architecture. Future work could explore learnable embedding weights, branch-level MoE, and incorporating external categorical data.

**Key Lessons**: (1) Fine-tune on exact task vs. frozen embeddings, (2) preserve information through architecture (avoid bottlenecks), (3) align training incentive with evaluation metric through weighted loss, (4) hierarchical fusion > flat concatenation for multi-modal learning.

---

## Appendix

### Code Structure

```
AmazonMlPricePredictor/
├── clip_finetune/         # CLIP fine-tuning & embedding extraction
│   ├── train.py           # Fine-tune CLIP model
│   ├── extract_embeddings.py  # Extract embeddings
│   └── README.md          # Detailed documentation
├── qwen3_finetune/        # Qwen3 fine-tuning with QLoRA
│   ├── train.py           # Fine-tune model
│   ├── evaluate.py        # Evaluate performance
│   ├── predict.py         # Generate predictions
│   └── README.md          # Detailed documentation
└── fusion/                # Fusion models package
    ├── __init__.py        # Package exports
    ├── loss_functions.py  # Shared custom loss functions
    ├── utils.py           # Shared utilities
    ├── README.md          # Package documentation
    ├── qwen_clip/         # Baseline (2 embeddings)
    │   ├── config.py
    │   ├── model.py
    │   ├── train.py
    │   ├── evaluate_by_groups.py
    │   ├── predict_test.py
    │   └── README.md
    └── all_embeddings/    # Best model (4 embeddings)
        ├── config.py
        ├── model.py
        ├── train.py
        ├── evaluate_by_groups.py
        ├── predict_test.py
        └── README.md
```

### Hardware & Training Time

- **GPU**: NVIDIA A100 (40GB) or V100 (32GB)
- **Total Pipeline Time**: ~30-40 hours (Qwen3 fine-tuning: 12-16h, embeddings: 2-3h, fusion training: 10-12h)
- **Inference**: <1ms per sample (forward pass only)

### Key Files

- **Model checkpoints**: `fusion/all_embeddings/checkpoints/best_model.pt`
- **Embeddings**: CLIP (768), SigLIP (1152+1024), Qwen (1027)
- **Predictions**: `test_out.csv` (format: `sample_id,price`)

---

**Code Repository**: https://drive.google.com/drive/folders/1ZohAdeUtZFGTC1vYRZvZk6LDMyEtTMAg?usp=sharing
