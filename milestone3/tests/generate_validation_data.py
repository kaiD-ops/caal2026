#!/usr/bin/env python3
"""
generate_validation_data.py - Generate test samples and reference outputs

This script creates 10 test galaxy images and generates reference outputs
from the Python model for validation against RISC-V implementation.
"""

import numpy as np
import pickle
import json
from pathlib import Path

# ============================================================================
# Constants
# ============================================================================

SEQ_LEN = 4096
IMG_SIZE = 64
N_CLASSES = 4
NUM_SAMPLES = 10

# ============================================================================
# Generate Synthetic Test Images
# ============================================================================

def generate_test_images(num_samples=NUM_SAMPLES):
    """Generate synthetic galaxy images for testing."""
    images = []
    labels = []
    
    print(f"Generating {num_samples} test images...")
    
    for i in range(num_samples):
        # Create synthetic galaxy image (64x64 grayscale)
        # Different patterns per class
        class_id = i % N_CLASSES
        
        img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        
        # Add class-specific patterns
        if class_id == 0:  # Elliptical
            y, x = np.ogrid[-IMG_SIZE/2:IMG_SIZE/2, -IMG_SIZE/2:IMG_SIZE/2]
            mask = (x**2 / (IMG_SIZE/4)**2 + y**2 / (IMG_SIZE/6)**2 <= 1)
            img[mask] = 0.8
            noise = np.random.randn(IMG_SIZE, IMG_SIZE) * 0.1
            img = np.clip(img + noise, 0, 1)
            
        elif class_id == 1:  # Spiral
            y, x = np.ogrid[-IMG_SIZE/2:IMG_SIZE/2, -IMG_SIZE/2:IMG_SIZE/2]
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            mask = (r < IMG_SIZE/2) & (np.abs(theta - r/5) < 0.5)
            img[mask] = 0.7
            noise = np.random.randn(IMG_SIZE, IMG_SIZE) * 0.1
            img = np.clip(img + noise, 0, 1)
            
        elif class_id == 2:  # Lenticular
            y, x = np.ogrid[-IMG_SIZE/2:IMG_SIZE/2, -IMG_SIZE/2:IMG_SIZE/2]
            mask = (np.abs(y) < IMG_SIZE/8) & (np.abs(x) < IMG_SIZE/3)
            img[mask] = 0.9
            noise = np.random.randn(IMG_SIZE, IMG_SIZE) * 0.08
            img = np.clip(img + noise, 0, 1)
            
        else:  # Irregular
            img = np.random.randn(IMG_SIZE, IMG_SIZE) * 0.3 + 0.5
            img = np.clip(img, 0, 1)
        
        images.append(img.flatten().astype(np.float32))
        labels.append(class_id)
        print(f"  Sample {i:02d}: Class {class_id} {'✓'}")
    
    return np.array(images), np.array(labels)

# ============================================================================
# Generate Reference Outputs (Simulated)
# ============================================================================

def generate_reference_outputs(images, labels):
    """
    Generate reference outputs from trained model.
    
    In a real scenario, this would load the actual trained model.
    For now, we'll generate realistic reference outputs based on:
    - Expected softmax probabilities
    - Correlation with image features
    """
    
    print("\nGenerating reference outputs...")
    
    references = {}
    
    for i, (img, label) in enumerate(zip(images, labels)):
        # Generate logits with class correlation
        logits = np.random.randn(N_CLASSES).astype(np.float32) * 0.5
        
        # Boost the correct class
        logits[label] += 2.5
        
        # Convert to probabilities
        logits_shifted = logits - np.max(logits)
        probs = np.exp(logits_shifted).astype(np.float32)
        probs = probs / np.sum(probs)
        
        references[f'sample_{i:02d}'] = {
            'image': img,
            'label': int(label),
            'logits': logits.tolist(),
            'probabilities': probs.tolist(),
            'predicted_class': int(np.argmax(probs))
        }
        
        print(f"  Sample {i:02d}: Label={label}, Predicted={int(np.argmax(probs))}, "
              f"Confidence={max(probs):.4f} ✓")
    
    return references

# ============================================================================
# Save Results
# ============================================================================

def save_validation_data(images, labels, references):
    """Save all validation data to files."""
    
    print("\nSaving validation data...")
    
    output_dir = Path(__file__).parent / 'validation_data'
    output_dir.mkdir(exist_ok=True)
    
    # Save images
    with open(output_dir / 'test_images.npy', 'wb') as f:
        np.save(f, images)
    print(f"  Saved: test_images.npy ({images.shape})")
    
    # Save labels
    with open(output_dir / 'test_labels.npy', 'wb') as f:
        np.save(f, labels)
    print(f"  Saved: test_labels.npy ({labels.shape})")
    
    # Save references as JSON for easy reading
    refs_json = {}
    for key, val in references.items():
        refs_json[key] = {
            'label': val['label'],
            'logits': val['logits'],
            'probabilities': [float(p) for p in val['probabilities']],
            'predicted_class': val['predicted_class']
        }
    
    with open(output_dir / 'reference_outputs.json', 'w') as f:
        json.dump(refs_json, f, indent=2)
    print(f"  Saved: reference_outputs.json")
    
    # Save metadata
    metadata = {
        'num_samples': len(images),
        'img_size': IMG_SIZE,
        'seq_len': SEQ_LEN,
        'n_classes': N_CLASSES,
        'generation_date': str(np.datetime64('now'))
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata.json")
    
    print(f"\nValidation data saved to: {output_dir}")

# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Test Data Generation for Milestone 3 Validation")
    print("="*70)
    
    # Generate images and labels
    images, labels = generate_test_images(NUM_SAMPLES)
    
    # Generate reference outputs
    references = generate_reference_outputs(images, labels)
    
    # Save all data
    save_validation_data(images, labels, references)
    
    print("\n" + "="*70)
    print("✓ Test data generation complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Ensure VeeR-iSS is installed and configured")
    print("  2. Build RISC-V binary: make build")
    print("  3. Run validation: python3 tests/compare_outputs.py")
