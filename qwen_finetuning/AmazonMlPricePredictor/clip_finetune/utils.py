"""
Utility functions for CLIP fine-tuning and embedding extraction
"""
import numpy as np
import torch
from tqdm import tqdm
import json


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all samples in dataloader
    
    Args:
        model: CLIP model
        dataloader: DataLoader with samples
        device: torch device
        
    Returns:
        Dictionary with sample_ids, image_embeddings, text_embeddings, and image_text_embeddings
    """
    model.eval()
    
    all_sample_ids = []
    all_image_embeddings = []
    all_text_embeddings = []
    
    print("Extracting embeddings...")
    for batch in tqdm(dataloader, desc="Extracting"):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sample_ids = batch['sample_id']
        
        # Get embeddings
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Extract normalized embeddings
        image_embeds = outputs.image_embeds.cpu().numpy()  # (batch, 768)
        text_embeds = outputs.text_embeds.cpu().numpy()    # (batch, 768)
        
        all_sample_ids.extend(sample_ids.tolist())
        all_image_embeddings.append(image_embeds)
        all_text_embeddings.append(text_embeds)
    
    # Concatenate all batches
    all_image_embeddings = np.vstack(all_image_embeddings)
    all_text_embeddings = np.vstack(all_text_embeddings)
    all_sample_ids = np.array(all_sample_ids)
    
    # Compute image+text embeddings (concatenate [image_emb || text_emb])
    all_image_text_embeddings = np.concatenate([all_image_embeddings, all_text_embeddings], axis=1)
    
    print(f"\nExtracted embeddings:")
    print(f"  Sample IDs: {all_sample_ids.shape}")
    print(f"  Image embeddings: {all_image_embeddings.shape}")
    print(f"  Text embeddings: {all_text_embeddings.shape}")
    print(f"  Image+Text embeddings: {all_image_text_embeddings.shape}")
    
    return {
        'sample_ids': all_sample_ids,
        'image_embeddings': all_image_embeddings,
        'text_embeddings': all_text_embeddings,
        'image_text_embeddings': all_image_text_embeddings
    }


def save_embeddings(embeddings, output_dir):
    """
    Save embeddings to disk
    
    Args:
        embeddings: Dictionary with embeddings arrays
        output_dir: Path to output directory
    """
    # Save as individual numpy arrays
    np.save(output_dir / 'sample_ids.npy', embeddings['sample_ids'])
    np.save(output_dir / 'image_embeddings.npy', embeddings['image_embeddings'])
    np.save(output_dir / 'text_embeddings.npy', embeddings['text_embeddings'])
    np.save(output_dir / 'image_text_embeddings.npy', embeddings['image_text_embeddings'])
    
    # Also save as a single npz file for convenience
    np.savez(
        output_dir / 'all_embeddings.npz',
        sample_ids=embeddings['sample_ids'],
        image_embeddings=embeddings['image_embeddings'],
        text_embeddings=embeddings['text_embeddings'],
        image_text_embeddings=embeddings['image_text_embeddings']
    )
    
    print(f"\nSaved embeddings to {output_dir}/")
    print(f"  - sample_ids.npy")
    print(f"  - image_embeddings.npy ({embeddings['image_embeddings'].shape})")
    print(f"  - text_embeddings.npy ({embeddings['text_embeddings'].shape})")
    print(f"  - image_text_embeddings.npy ({embeddings['image_text_embeddings'].shape})")
    print(f"  - all_embeddings.npz (combined)")


def save_training_history(history, output_path):
    """
    Save training history to JSON
    
    Args:
        history: Dictionary with train_losses and val_losses
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {output_path}")


def load_embeddings(embeddings_dir):
    """
    Load saved embeddings
    
    Args:
        embeddings_dir: Path to directory containing embeddings
        
    Returns:
        Dictionary with embeddings arrays
    """
    data = np.load(embeddings_dir / 'all_embeddings.npz')
    return {
        'sample_ids': data['sample_ids'],
        'image_embeddings': data['image_embeddings'],
        'text_embeddings': data['text_embeddings'],
        'image_text_embeddings': data['image_text_embeddings']
    }

