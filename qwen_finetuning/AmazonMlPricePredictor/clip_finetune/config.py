"""
Configuration for CLIP fine-tuning
"""
from pathlib import Path


class Config:
    """Configuration parameters for CLIP fine-tuning"""
    
    # Paths (dataset paths are now in dataset.py)
    OUTPUT_DIR = Path("/data/utk/amazon_ml_2025/embeddings_clip")
    CHECKPOINT_DIR = Path("/data/utk/amazon_ml_2025/checkpoints_clip")
    
    # Model
    MODEL_NAME = "openai/clip-vit-large-patch14"
    NUM_UNFROZEN_BLOCKS = 3  # Unfreeze last N transformer blocks in vision encoder
    
    # Training
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    GRADIENT_CLIP = 1.0
    VAL_SPLIT = 0.1  # 10% validation
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N epochs
    
    # Learning Rate Scheduler
    LR_SCHEDULER_PATIENCE = 2  # Reduce LR if no improvement for N epochs
    LR_SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor
    LR_SCHEDULER_MIN_LR = 1e-7  # Minimum learning rate
    
    # Hardware
    DEVICE = "cuda:0"  # Using GPU 1 (more free memory)
    NUM_WORKERS = 4
    
    # Logging
    LOG_INTERVAL = 50  # Log every N batches
    SAVE_CHECKPOINTS = True
    
    def __init__(self):
        """Create output directories if they don't exist"""
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

