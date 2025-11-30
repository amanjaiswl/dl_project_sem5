import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from pathlib import Path
import re
from autocorrect import Speller

# Allow loading of truncated/corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProductDataset(Dataset):
    def __init__(self, split="train", processor=None, use_augmentation=False):
        """
        Args:
            split (str): Either "train" or "test"
            processor: CLIP processor for text and image preprocessing (required for CLIP mode)
            use_augmentation (bool): Whether to use data augmentation (default: False)
        """
        self.split = split
        self.use_augmentation = use_augmentation
        self.processor = processor
        
        # Setup paths
        base_dir = Path("/data/utk/amazon_ml_2025")
        csv_path = base_dir / "dataset" / f"{split}.csv"
        self.images_dir = base_dir / "data" / split
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        
        # Determine mode
        self.use_clip_mode = processor is not None
        
        # Initialize autocorrect speller (only if using augmentation)
        if self.use_augmentation:
            self.speller = Speller(lang='en')
        else:
            self.speller = None
        
        # Define image transformations
        # Augmentation 1: Scale down to 112x112, then up to 224x224
        self.transform_down_up = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Augmentation 2: Direct resize to 224x224
        self.transform_direct = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Augmentation 3: Horizontal flip
        self.transform_flip = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.ToTensor(),
        ])
        
        self.image_transforms = [
            self.transform_down_up,
            self.transform_direct,
            self.transform_flip
        ]
    
    def add_misspellings(self, text, error_rate=0.03):
        """Add character-level errors to text with given error rate"""
        if not text or pd.isna(text):
            return text
        
        chars = list(text)
        num_errors = max(1, int(len(chars) * error_rate))
        
        # Get random positions to introduce errors
        error_positions = random.sample(range(len(chars)), min(num_errors, len(chars)))
        
        for pos in error_positions:
            if chars[pos].isalpha():
                error_type = random.choice(['swap', 'delete', 'duplicate'])
                
                if error_type == 'swap' and pos < len(chars) - 1:
                    # Swap with next character
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                elif error_type == 'delete':
                    chars[pos] = ''
                elif error_type == 'duplicate':
                    chars[pos] = chars[pos] * 2
        
        return ''.join(chars)
    
    def clean_catalog_text(self, text):
        """
        Clean catalog content by removing Value and Unit lines
        """
        if pd.isna(text):
            return ""
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines that start with "Value:" or "Unit:"
            if line.startswith('Value:') or line.startswith('Unit:'):
                continue
            # Skip empty lines
            if not line:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_text(self, text, augmentation_type):
        """
        Process text with different augmentations:
        0: Ground truth (original, cleaned)
        1: With misspellings
        2: Autocorrected
        """
        if pd.isna(text):
            text = ""
        
        # Always clean the text first (remove Value and Unit)
        text = self.clean_catalog_text(text)
        
        if augmentation_type == 0:
            # Ground truth (cleaned)
            return text
        elif augmentation_type == 1:
            # Add misspellings
            return self.add_misspellings(text)
        elif augmentation_type == 2:
            # Autocorrect
            return self.speller(text)
        
        return text
    
    def get_image_path(self, row):
        """Extract image filename from URL and construct full path. If original image does not exist, use sample_id.jpg."""
        image_url = row['image_link']
        sample_id = row['sample_id']

        if pd.isna(image_url):
            return None
        
        # Extract filename from URL (e.g., 71hoAn78AWL.jpg)
        filename = image_url.split('/')[-1]
        image_path = self.images_dir / filename
        
        if not image_path.exists():
            # Fallback to sample_id.jpg if the original image path does not exist
            print(f"Warning: Image {image_path} not found. Trying {sample_id}.jpg")
            image_path = self.images_dir / f"{sample_id}.jpg"

        return image_path
    
    def __len__(self):
        if self.use_augmentation:
            # Each row has 9 augmentations (3 image x 3 text)
            return len(self.df) * 9
        else:
            # No augmentation: one sample per row
            return len(self.df)
    
    def __getitem__(self, idx):
        if self.use_augmentation:
            # Augmentation mode: each row has 9 variations
            original_idx = idx // 9
            aug_idx = idx % 9
            
            # Determine image and text augmentation indices
            img_aug_idx = aug_idx // 3  # 0, 1, or 2
            text_aug_idx = aug_idx % 3   # 0, 1, or 2
            
            # Get row data
            row = self.df.iloc[original_idx]
            
            # Load and transform image
            image_path = self.get_image_path(row)
            
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.image_transforms[img_aug_idx](image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image = torch.zeros(3, 224, 224)
            
            # Process text
            catalog_content = self.process_text(row['catalog_content'], text_aug_idx)
            
            # Create sample dictionary
            sample = {
                'sample_id': row['sample_id'],
                'catalog_content': catalog_content,
                'image': image,
                'image_aug_type': img_aug_idx,
                'text_aug_type': text_aug_idx,
                'original_idx': original_idx,
                'aug_idx': aug_idx
            }
            
            return sample
        
        elif self.use_clip_mode:
            # CLIP fine-tuning mode: use processor, no augmentation
            row = self.df.iloc[idx]
            sample_id = row['sample_id']
            text = self.clean_catalog_text(row['catalog_content'])  # Clean the text
            
            # Load image
            image_path = self.get_image_path(row)
            
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load {image_path}: {e}")
                image = Image.new('RGB', (224, 224), color='white')
            
            # Process inputs through CLIP processor (no augmentation)
            # Process as single text to avoid batch dimension issues
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding="max_length",  # Pad to max_length
                truncation=True
            )
            
            # Ensure all tensors are contiguous and remove batch dimension
            processed_inputs = {}
            for k, v in inputs.items():
                if v.dim() > 1:  # Remove batch dimension if present
                    processed_inputs[k] = v.squeeze(0).contiguous()
                else:
                    processed_inputs[k] = v.contiguous()
            
            processed_inputs['sample_id'] = sample_id
            
            return processed_inputs
        
        else:
            # Original mode without augmentation
            row = self.df.iloc[idx]
            
            # Load image
            image_path = self.get_image_path(row)
            
            try:
                image = Image.open(image_path).convert('RGB')
                image = self.transform_direct(image)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                image = torch.zeros(3, 224, 224)
            
            # Get text (no augmentation, cleaned)
            catalog_content = self.clean_catalog_text(row['catalog_content'])
            
            sample = {
                'sample_id': row['sample_id'],
                'catalog_content': catalog_content,
                'image': image,
            }
            
            return sample


# Example usage
if __name__ == "__main__":
    from transformers import CLIPProcessor
    
    # Example 1: CLIP mode (for fine-tuning)
    print("=" * 50)
    print("CLIP Mode Example")
    print("=" * 50)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    train_dataset = ProductDataset(split="train", processor=processor)
    print(f"Train dataset size: {len(train_dataset)}")
    
    sample = train_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Sample ID: {sample['sample_id']}")
    
    # Example 2: Test set (no augmentation)
    print("\n" + "=" * 50)
    print("Test Set Example")
    print("=" * 50)
    test_dataset = ProductDataset(split="test", processor=processor)
    print(f"Test dataset size: {len(test_dataset)}")
