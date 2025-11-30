"""
Configuration for SigLIP Fusion Model
Combines SigLIP image embeddings (1152-dim) with SigLIP text embeddings (1024-dim)
"""
from pathlib import Path


class Config:
    """Configuration parameters for SigLIP fusion model training"""
    
    # Paths
    CLIP_EMBEDDINGS_DIR = Path("/data/utk/amazon_ml_2025/embeddings_clip/train")
    SIGLIP_EMBEDDINGS_DIR = Path("/data/utk/amazon_ml_2025/embedding_cache")
    QWEN_EMBEDDINGS_PATH = Path("/data/utk/amazon_ml_2025/embeddings_qwen/qwen3_embeddings_train.pkl")
    CSV_PATH = Path("/data/utk/amazon_ml_2025/dataset/train.csv")
    OUTPUT_DIR = Path("/data/utk/amazon_ml_2025/outputs/fusion_siglip_v2")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    
    # Data splits
    TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation
    RANDOM_SEED = 41
    STRATIFY_BINS = 10  # Number of bins for stratified split by price
    
    # Model architecture - ALL 4 EMBEDDINGS (SIMPLE SINGLE-LAYER BRANCHES)
    CLIP_IMAGE_DIM = 768      # CLIP image embeddings
    SIGLIP_IMAGE_DIM = 1152   # SigLIP image embeddings
    SIGLIP_TEXT_DIM = 1024    # SigLIP text embeddings (actual dim from data)
    QWEN_TEXT_DIM = 1027      # Qwen text embeddings (with structural)
    
    # Branch dimensions - KEEP SIMPLE (single layer each)
    CLIP_IMAGE_BRANCH_DIMS = [768, 512]    # 768 → 512
    SIGLIP_IMAGE_BRANCH_DIMS = [1152, 512] # 1152 → 512
    SIGLIP_TEXT_BRANCH_DIMS = [1024, 512]  # 1024 → 512
    QWEN_TEXT_BRANCH_DIMS = [1027, 512]    # 1027 → 512
    
    # Fusion and regression head
    FUSION_DIM = 512 * 4  # 2048 (4 branches × 512)
    REGRESSION_HEAD_DIMS = [2048, 1024, 512, 256, 128, 1]
    
    # Regularization
    DROPOUT_RATE = 0.25
    USE_LAYER_NORM = True
    USE_RESIDUAL = True
    
    # Advanced model options
    USE_ADVANCED_MODEL = True  # Use MoE + Cross-Attention
    NUM_EXPERTS = 3
    EXPERT_HIDDEN_DIMS = [1024, 512, 256]  # Larger for 2048-dim input
    NUM_ATTENTION_HEADS = 8  # For image-image and text-text fusions (1024/8=128)
    
    # Training
    BATCH_SIZE = 256
    NUM_EPOCHS = 1000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = 1.0
    
    # Loss function
    LOSS_TYPE = "weighted_smape"
    HUBER_DELTA = 1.0
    PREDICT_LOG = True  # Predict log1p(price)
    
    # SMAPE loss parameters
    SMAPE_GAMMA = 2.0
    SMAPE_EPS = 1e-8
    SMAPE_NORMALIZE = True
    
    # Hybrid loss parameters
    HYBRID_BASE_LOSS = "huber"
    SMAPE_WEIGHT = 0.3
    
    # Optimizer
    OPTIMIZER = "adamw"
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "reduce_on_plateau"
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_METRIC = "val_smape"
    
    # Hardware
    DEVICE = "cuda:0"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Logging
    LOG_INTERVAL = 20
    SAVE_BEST_ONLY = True
    VERBOSE = True
    
    def __init__(self):
        """Create output directories if they don't exist"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

