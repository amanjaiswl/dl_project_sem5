"""
CLIP model wrapper with selective layer unfreezing for fine-tuning
"""
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


class CLIPFineTuner:
    """Fine-tune CLIP with selective layer unfreezing"""
    
    def __init__(self, config):
        """
        Args:
            config: Config object with model and training parameters
        """
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Load model and processor
        print(f"Loading CLIP model: {config.MODEL_NAME}")
        
        # Load default CLIPConfig and modify projection_dim to 512 to match the checkpoint
        # clip_config = CLIPConfig.from_pretrained(config.MODEL_NAME)
        # clip_config.projection_dim = 512  # Set to 512 to match the checkpoint

        # Initialize model with custom config (randomly initialized projection layer)
        self.model = CLIPModel.from_pretrained(config.MODEL_NAME)
        
        # Load original pretrained weights, ignoring the mismatched projection layers
        # base_model = CLIPModel.from_pretrained(config.MODEL_NAME)
        # self.model.load_state_dict(base_model.state_dict(), strict=False)

        self.processor = CLIPProcessor.from_pretrained(config.MODEL_NAME)
        
        # Unfreeze last N transformer blocks in vision encoder
        self._selective_unfreeze()
        
        self.model.to(self.device)
        
        # Setup optimizer (only for unfrozen parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_SCHEDULER_MIN_LR
        )
        
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def _selective_unfreeze(self):
        """Freeze all parameters except last N transformer blocks in vision encoder and last 2 blocks in text encoder"""
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N transformer blocks in vision encoder
        # CLIP ViT-Large has 24 transformer layers
        vision_num_layers = len(self.model.vision_model.encoder.layers)
        vision_unfreeze_from = vision_num_layers - self.config.NUM_UNFROZEN_BLOCKS
        
        print(f"Unfreezing vision transformer blocks {vision_unfreeze_from} to {vision_num_layers-1}")
        for i in range(vision_unfreeze_from, vision_num_layers):
            for param in self.model.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze the final layer norm and projection for vision
        for param in self.model.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        if hasattr(self.model.vision_model, 'visual_projection'):
            for param in self.model.vision_model.visual_projection.parameters():
                param.requires_grad = True
        
        # Unfreeze last 2 transformer blocks in text encoder
        # CLIP text encoder has 12 transformer layers
        text_num_layers = len(self.model.text_model.encoder.layers)
        text_unfreeze_from = text_num_layers - 2  # Unfreeze last 2 layers
        
        print(f"Unfreezing text transformer blocks {text_unfreeze_from} to {text_num_layers-1}")
        for i in range(text_unfreeze_from, text_num_layers):
            for param in self.model.text_model.encoder.layers[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze the final layer norm and projection for text
        for param in self.model.text_model.final_layer_norm.parameters():
            param.requires_grad = True
    
    def train_epoch(self, dataloader, epoch):
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRADIENT_CLIP
            )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if (batch_idx + 1) % self.config.LOG_INTERVAL == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: avg_loss={avg_loss:.4f}")
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader):
        """
        Validate on validation set
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        if is_best:
            path = self.config.CHECKPOINT_DIR / 'best_model.pt'
        else:
            path = self.config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch+1}.pt'
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path, strict=True):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state_dict keys match
            
        Returns:
            Tuple of (epoch, val_loss)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint.get('val_loss', 0.0)

