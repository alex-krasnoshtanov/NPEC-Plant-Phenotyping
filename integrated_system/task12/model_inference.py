"""
Model inference with patch-based processing.

Handles large images by:
1. Padding to make dimensions compatible with patching
2. Dividing into overlapping patches
3. Running model on each patch
4. Reconstructing with overlap averaging
5. Removing padding to restore original size
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from pathlib import Path


class ModelInference:
    """
    Handles model loading and patch-based inference.
    
    Design choice: Patch-based approach allows processing arbitrarily large images
    with fixed GPU memory. Overlap reduces edge artifacts.
    
    Alternative: Could resize entire image (loses resolution) or use
    fully convolutional approach with sliding window.
    """
    
    def __init__(self, model_path: str, patch_size: int = 256, overlap: float = 0.5):
        """
        Initialize model inference.
        
        Args:
            model_path: Path to model checkpoint (.pth file)
            patch_size: Size of square patches (should match training)
            overlap: Overlap ratio between patches (0.0 to 1.0)
                    0.5 means 50% overlap (step = patch_size * 0.5)
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.step = int(patch_size * (1 - overlap))
        
        print(f"Initializing model inference:")
        print(f"  Patch size: {patch_size}x{patch_size}")
        print(f"  Overlap: {overlap*100:.0f}%")
        print(f"  Step: {self.step}")
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {self.device}")
        
        # Detect input channels from checkpoint and load model
        self.in_channels = self._detect_input_channels(model_path)
        print(f"  Model input channels: {self.in_channels}")
        
        # Load model
        self._load_model(model_path)
    
    def _detect_input_channels(self, model_path: str) -> int:
        """
        Detect number of input channels from checkpoint.
        
        Looks at the shape of the first convolution layer to determine
        if the model was trained with grayscale (1 channel) or RGB (3 channels).
        
        Args:
            model_path: Path to checkpoint
            
        Returns:
            Number of input channels (1 or 3)
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get the state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check the first conv layer shape
        # In ResNet34 encoder, first layer is 'encoder.conv1.weight'
        if 'encoder.conv1.weight' in state_dict:
            weight_shape = state_dict['encoder.conv1.weight'].shape
            in_channels = weight_shape[1]  # Shape is [out_channels, in_channels, height, width]
            print(f"  Detected from encoder.conv1.weight: {in_channels} input channels")
            return in_channels
        else:
            # Fallback: assume 1 channel (most common for root segmentation)
            print(f"  Warning: Could not detect input channels, assuming 1 (grayscale)")
            return 1
    
    def _load_model(self, model_path: str):
        """Load ResNet-UNet model from checkpoint."""
        import segmentation_models_pytorch as smp
        
        print(f"Loading model from: {model_path}")
        
        # Initialize architecture (must match training)
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=None,  # We'll load trained weights
            in_channels=self.in_channels,  # Use detected channels (1 or 3)
            classes=1,      # Binary segmentation
            activation=None # Apply sigmoid during inference
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  Loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            self.model.load_state_dict(checkpoint)
            print(f"  Loaded state dict")
        
        self.model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model parameters: {total_params:,}")
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Run prediction on image with patching.
        
        Args:
            image: RGB image (H, W, 3) as uint8
            
        Returns:
            Prediction mask (H, W) as float32 in range [0, 1]
        """
        h, w = image.shape[:2]
        print(f"Running inference on image: {h}x{w}")
        
        # Convert to grayscale if model expects 1 channel
        if self.in_channels == 1:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                print(f"  Converted RGB -> Grayscale: {image.shape}")
            # Add channel dimension for processing
            image = np.expand_dims(image, axis=-1)  # (H, W) -> (H, W, 1)
            print(f"  Added channel dimension: {image.shape}")
        elif self.in_channels == 3:
            if len(image.shape) == 2:
                # Convert grayscale to RGB by replicating
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                print(f"  Converted Grayscale -> RGB for model")
        
        # Step 1: Calculate padding
        padding = self._calculate_padding(h, w)
        print(f"  Padding: top={padding[0]}, bottom={padding[1]}, left={padding[2]}, right={padding[3]}")
        
        # Step 2: Pad image
        padded = self._pad_image(image, padding)
        print(f"  Padded shape: {padded.shape}")
        
        # Step 3: Create patches
        patches = self._create_patches(padded)
        print(f"  Created {len(patches)} patches of shape {patches.shape[1:]}")
        
        # Step 4: Run model on patches
        predictions = self._predict_patches(patches)
        print(f"  Predictions shape: {predictions.shape}")
        
        # Step 5: Reconstruct from patches
        padded_h, padded_w = padded.shape[:2]
        reconstructed = self._reconstruct(predictions, padded_h, padded_w)
        print(f"  Reconstructed shape: {reconstructed.shape}")
        
        # Step 6: Remove padding
        final = self._remove_padding(reconstructed, padding)
        print(f"  Final shape after unpadding: {final.shape}")
        
        # Verify shape
        if final.shape != (h, w):
            raise ValueError(f"Shape mismatch! Expected {(h, w)}, got {final.shape}")
        
        return final
    
    def _calculate_padding(self, h: int, w: int) -> Tuple[int, int, int, int]:
        """
        Calculate padding needed for patching.
        
        The image must be divisible by step size (with remainder for last patch).
        Formula: padded_size >= (patch_size + n * step) where n >= 0
        
        Args:
            h: Image height
            w: Image width
            
        Returns:
            (top, bottom, left, right) padding
        """
        def pad_needed(size):
            if size < self.patch_size:
                return self.patch_size - size
            
            remainder = (size - self.patch_size) % self.step
            return (self.step - remainder) % self.step if remainder != 0 else 0
        
        h_pad = pad_needed(h)
        w_pad = pad_needed(w)
        
        # Split padding evenly on both sides
        top = h_pad // 2
        bottom = h_pad - top
        left = w_pad // 2
        right = w_pad - left
        
        return top, bottom, left, right
    
    def _pad_image(self, image: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Pad image with reflection.
        
        Design choice: BORDER_REFLECT_101 avoids edge discontinuities better than
        constant padding. The reflected pixels maintain local structure.
        """
        top, bottom, left, right = padding
        
        # cv2.copyMakeBorder works with both grayscale and color images
        padded = cv2.copyMakeBorder(
            image, 
            top, bottom, left, right, 
            cv2.BORDER_REFLECT_101
        )
        
        # If original had channel dimension but padded lost it, restore it
        # This happens with (H, W, 1) -> (H, W)
        if len(image.shape) == 3 and len(padded.shape) == 2:
            padded = np.expand_dims(padded, axis=-1)
        
        return padded
    
    def _create_patches(self, image: np.ndarray) -> np.ndarray:
        """
        Create overlapping patches from image.
        
        Args:
            image: Padded image (H, W, C) where C is 1 or 3
            
        Returns:
            Patches array (N, C, patch_size, patch_size) in PyTorch CHW format
            
        Note: Converts from numpy HWC to PyTorch CHW format.
        """
        from patchify import patchify
        
        # Ensure image has channel dimension
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        h, w, c = image.shape
        
        # Create patches (returns shape: n_h, n_w, 1, patch_size, patch_size, C)
        patches_array = patchify(
            image, 
            (self.patch_size, self.patch_size, c), 
            step=self.step
        )
        
        n_h, n_w = patches_array.shape[0], patches_array.shape[1]
        print(f"    Patch grid: {n_h}x{n_w}")
        
        # Extract and stack patches
        patch_list = []
        for i in range(n_h):
            for j in range(n_w):
                patch = patches_array[i, j, 0, :, :, :]  # Shape: (patch_size, patch_size, C)
                patch_list.append(patch)
        
        # Stack into batch (N, H, W, C)
        patches_batch = np.stack(patch_list, axis=0)
        
        # Convert to PyTorch format: (N, C, H, W)
        patches_batch = patches_batch.transpose(0, 3, 1, 2)
        
        return patches_batch
    
    def _predict_patches(self, patches: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """
        Run model prediction on all patches.
        
        Args:
            patches: Patches array (N, C, patch_size, patch_size) as uint8 where C is 1 or 3
            batch_size: Batch size for inference
            
        Returns:
            Predictions (N, patch_size, patch_size) as float32 in [0, 1]
        """
        # Convert to torch tensor and normalize [0, 255] -> [0, 1]
        patches_tensor = torch.from_numpy(patches).float() / 255.0
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size].to(self.device)
                
                # Forward pass
                output = self.model(batch)
                
                # Apply sigmoid to get probabilities
                output = torch.sigmoid(output)
                
                # Move to CPU and remove channel dimension
                output = output.cpu().numpy()[:, 0, :, :]  # (N, 1, H, W) -> (N, H, W)
                
                predictions.append(output)
        
        # Concatenate all batches
        all_predictions = np.concatenate(predictions, axis=0)
        
        return all_predictions
    
    def _reconstruct(self, predictions: np.ndarray, padded_h: int, padded_w: int) -> np.ndarray:
        """
        Reconstruct full image from overlapping patches.
        
        Uses weighted averaging in overlap regions to reduce edge artifacts.
        
        Args:
            predictions: Patch predictions (N, patch_size, patch_size)
            padded_h: Height of padded image
            padded_w: Width of padded image
            
        Returns:
            Reconstructed prediction (padded_h, padded_w)
        """
        reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)
        counts = np.zeros((padded_h, padded_w), dtype=np.float32)
        
        # Calculate number of patches in each dimension
        n_h = (padded_h - self.patch_size) // self.step + 1
        n_w = (padded_w - self.patch_size) // self.step + 1
        
        idx = 0
        for i in range(n_h):
            for j in range(n_w):
                y_start = i * self.step
                x_start = j * self.step
                y_end = y_start + self.patch_size
                x_end = x_start + self.patch_size
                
                # Add prediction to accumulated result
                reconstructed[y_start:y_end, x_start:x_end] += predictions[idx]
                counts[y_start:y_end, x_start:x_end] += 1
                
                idx += 1
        
        # Average overlapping regions
        counts = np.maximum(counts, 1)  # Avoid division by zero
        reconstructed = reconstructed / counts
        
        return reconstructed
    
    def _remove_padding(self, array: np.ndarray, padding: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove padding to restore original size."""
        top, bottom, left, right = padding
        h, w = array.shape[:2]
        
        y_start = top
        y_end = h - bottom if bottom > 0 else h
        x_start = left
        x_end = w - right if right > 0 else w
        
        return array[y_start:y_end, x_start:x_end]


# Import cv2 here (used by _pad_image)
import cv2


if __name__ == "__main__":
    """Test model inference on a sample image."""
    print("This is a module. Import and use ModelInference class.")
    print("Example usage:")
    print("  from model_inference import ModelInference")
    print("  model = ModelInference('path/to/model.pth')")
    print("  prediction = model.predict(image_rgb)")