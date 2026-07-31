"""
Custom U-Net training for root segmentation - Training from scratch.

Architecture: ResUNet with Attention Gates
- Residual blocks: Better gradient flow for deep networks
- Attention gates: Focus on sparse root regions, suppress background
- Batch normalization: Training stability
- Kaiming initialization: Proper weight initialization for ReLU networks

Implemented features:
- Early stopping based on validation F1 score
- Patch filtering to remove empty/low-information patches  
- Gradient clipping for training stability
- Combined Dice-BCE loss with numerical stability
- F1 score calculation matching Keras implementation
- Training history saved to JSON
- Model checkpointing (best and latest)

Why this approach:
- ResUNet is proven for biomedical/root segmentation tasks
- Attention gates help with sparse targets (roots vs background)
- Training from scratch avoids pretrained weight mismatch
- Early stopping prevents overfitting
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast as cuda_autocast
from torch.amp import GradScaler
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# ============================================================================
# AUGMENTATION
# ============================================================================

class ShootAugmentation:
    """
    Augmentation for shoot segmentation.
    
    Why different from roots:
    - Shoots grow upward with horizontal branching and leaves
    - More varied orientations than roots
    - Leaves can point in multiple directions
    - More tolerance for rotation since structure is less directional
    
    Augmentations:
    - Horizontal flip (50%): Shoots can branch left or right
    - Moderate rotation (±15°): Shoots have more angular variation
    - Brightness/contrast: Leaf color can vary with lighting
    """
    def __init__(self, 
                 horizontal_flip_prob=0.5,
                 max_rotation_degrees=15,  # Increased from 5°
                 brightness_range=0.15,     # Increased from 0.1
                 contrast_range=0.1):       # New parameter
        self.horizontal_flip_prob = horizontal_flip_prob
        self.max_rotation_degrees = max_rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, image, mask):
        # Horizontal flip
        if np.random.random() < self.horizontal_flip_prob:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
    
        
        # Rotation - larger range for shoots
        if self.max_rotation_degrees > 0:
            angle = np.random.uniform(-self.max_rotation_degrees, self.max_rotation_degrees)
            image = self._rotate(image, angle)
            mask = self._rotate(mask, angle)
        
        # Brightness adjustment
        if self.brightness_range > 0:
            factor = np.random.uniform(1 - self.brightness_range, 1 + self.brightness_range)
            image = np.clip(image * factor, 0, 1)
        
        # Contrast adjustment (new for shoots)
        if self.contrast_range > 0:
            factor = np.random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 1)
        
        return image, mask
    
    def _rotate(self, img, angle):
        if img.ndim == 3:
            rotated = np.zeros_like(img)
            for c in range(img.shape[0]):
                rotated[c] = self._rotate_2d(img[c], angle)
            return rotated
        else:
            return self._rotate_2d(img, angle)
    
    def _rotate_2d(self, img, angle):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), 
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        return rotated

# ============================================================================
# ARCHITECTURE COMPONENTS
# ============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with dropout: Two convolutions with skip connection.
    Structure: Conv -> BN -> ReLU -> Dropout -> Conv -> BN -> Dropout -> Add input -> ReLU
    
    Dropout helps prevent overfitting by randomly dropping activations during training.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
        # Skip connection - match channels if needed
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        out = out + identity
        out = self.relu(out)
        
        return out


class AttentionGate(nn.Module):
    """
    Attention gate: Highlights relevant features in skip connections.
    Helps focus on sparse root regions and suppress background.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Channels in gating signal (from decoder)
            F_l: Channels in encoder features (skip connection)
            F_int: Intermediate channels
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (coarser resolution)
            x: Encoder features (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class DownBlock(nn.Module):
    """
    Encoder block: Residual block + max pooling.
    Returns both pooled output and pre-pool features for skip connection.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.residual(x)
        return self.pool(x), x  # (pooled, skip)


class UpBlock(nn.Module):
    """
    Decoder block: Upsample + attention gate + concatenate + residual block.
    
    CRITICAL: Channel flow fixed
    - Input x: in_channels
    - After upsampling: out_channels
    - Skip connection: in_channels (from corresponding encoder level)
    - After attention + concat: out_channels + in_channels
    - After residual: out_channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        
        # FIX: Skip connection has in_channels, not out_channels
        self.attention = AttentionGate(F_g=out_channels, F_l=in_channels, F_int=out_channels//2)
        
        # FIX: After concat we have out_channels + in_channels
        self.residual = ResidualBlock(in_channels + out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.attention(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        x = self.residual(x)
        return x


# ============================================================================
# MAIN ARCHITECTURE
# ============================================================================

class ResAttentionUNet(nn.Module):
    """
    ResUNet with Attention Gates for root segmentation.
    5-level architecture with residual connections and attention gates.
    """
    def __init__(self, in_channels=1, out_channels=1, init_filters=64):
        super().__init__()
        
        self.init_conv = ResidualBlock(in_channels, init_filters)
        
        # Encoder
        self.down1 = DownBlock(init_filters, init_filters * 2)
        self.down2 = DownBlock(init_filters * 2, init_filters * 4)
        self.down3 = DownBlock(init_filters * 4, init_filters * 8)
        self.down4 = DownBlock(init_filters * 8, init_filters * 16)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(init_filters * 16, init_filters * 16)
        
        # Decoder
        self.up1 = UpBlock(init_filters * 16, init_filters * 8)
        self.up2 = UpBlock(init_filters * 8, init_filters * 4)
        self.up3 = UpBlock(init_filters * 4, init_filters * 2)
        self.up4 = UpBlock(init_filters * 2, init_filters)
        
        # Output
        self.out_conv = nn.Conv2d(init_filters, out_channels, 1)
        
        # Initialize weights for training from scratch
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights using Kaiming initialization.
        
        Why Kaiming: Designed for ReLU activations, helps prevent vanishing/exploding
        gradients when training deep networks from scratch.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial
        x = self.init_conv(x)
        
        # Encoder with skip connections
        x1, skip1 = self.down1(x)     # 128 channels
        x2, skip2 = self.down2(x1)    # 256 channels
        x3, skip3 = self.down3(x2)    # 512 channels
        x4, skip4 = self.down4(x3)    # 1024 channels
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder with attention
        x = self.up1(x, skip4)  # 512 channels
        x = self.up2(x, skip3)  # 256 channels
        x = self.up3(x, skip2)  # 128 channels
        x = self.up4(x, skip1)  # 64 channels
        
        # Output
        x = self.out_conv(x)
        
        return x


# ============================================================================
# DATASET
# ============================================================================

class RootSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filter_empty=True, min_positive_pixels=10, 
                 cache_file=None, augmentation=None, empty_patch_ratio=0.45):
        """
        Args:
            empty_patch_ratio: Multiplier for empty patches relative to patches with roots.
                              0.5 = include 0.5x as many empty patches as patches with roots
                              1.0 = equal number of empty and root patches (1:1 ratio)
                              2.0 = include 2x as many empty patches as patches with roots
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augmentation = augmentation
        
        all_image_files = sorted(list(self.image_dir.glob('*.png')))
        print(f"Found {len(all_image_files)} total image files")
        
        if cache_file and Path(cache_file).exists():
            print(f"Loading cached file list from {cache_file}")
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            self.image_files = [self.image_dir / name for name in cache_data['files']]
            print(f"Loaded {len(self.image_files)} cached patches")
        else:
            # Separate patches into two categories
            patches_with_roots = []
            empty_patches = []
            missing_count = 0
            
            print("Categorizing patches...")
            for img_file in tqdm(all_image_files, desc="Checking patches"):
                mask_file = self.mask_dir / img_file.name
                
                if not mask_file.exists():
                    missing_count += 1
                    continue
                
                if self._has_sufficient_labels(mask_file, min_positive_pixels):
                    patches_with_roots.append(img_file)
                else:
                    empty_patches.append(img_file)
            
            # Calculate how many empty patches to include based on multiplier
            if filter_empty and empty_patch_ratio > 0:
                # Simple multiplier: empty_patch_ratio * num_patches_with_roots
                n_empty_to_include = int(len(patches_with_roots) * empty_patch_ratio)
                n_empty_to_include = min(n_empty_to_include, len(empty_patches))
                
                # Randomly sample empty patches
                np.random.seed(42)  # Reproducibility
                selected_empty = np.random.choice(empty_patches, n_empty_to_include, replace=False).tolist()
                
                self.image_files = patches_with_roots + selected_empty
                
                # Calculate actual ratio for reporting
                actual_ratio = len(selected_empty) / len(patches_with_roots) if len(patches_with_roots) > 0 else 0
                empty_percentage = len(selected_empty) / len(self.image_files) * 100 if len(self.image_files) > 0 else 0
                
            elif not filter_empty:
                self.image_files = patches_with_roots + empty_patches
                actual_ratio = len(empty_patches) / len(patches_with_roots) if len(patches_with_roots) > 0 else 0
                empty_percentage = len(empty_patches) / len(self.image_files) * 100 if len(self.image_files) > 0 else 0
            else:
                self.image_files = patches_with_roots
                actual_ratio = 0
                empty_percentage = 0
            
            print(f"\nDataset composition:")
            print(f"  Patches with roots: {len(patches_with_roots)}")
            print(f"  Empty patches available: {len(empty_patches)}")
            if filter_empty and empty_patch_ratio > 0:
                print(f"  Empty patches included: {len(selected_empty)} (ratio: {actual_ratio:.2f}x, {empty_percentage:.1f}% of dataset)")
            elif not filter_empty:
                print(f"  Empty patches included: all {len(empty_patches)} (ratio: {actual_ratio:.2f}x, {empty_percentage:.1f}% of dataset)")
            print(f"  Total in dataset: {len(self.image_files)}")
            if missing_count > 0:
                print(f"  Missing masks: {missing_count}")
            
            # Save cache
            if cache_file:
                cache_path = Path(cache_file)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    'files': [f.name for f in self.image_files],
                    'n_with_roots': len(patches_with_roots),
                    'n_empty': len(selected_empty) if filter_empty and empty_patch_ratio > 0 else (len(empty_patches) if not filter_empty else 0)
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                print(f"Saved cache to {cache_file}")
    
    def _has_sufficient_labels(self, mask_path, min_pixels):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False
        positive_pixels = np.count_nonzero(mask)
        return positive_pixels >= min_pixels
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
    
        # Load grayscale image
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
    
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(np.float32)
        
        # Apply augmentation if provided (training only)
        if self.augmentation is not None:
            image, mask = self.augmentation(image, mask)
        
        # Add channel dimension and convert to torch
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
    
        return image, mask


# ============================================================================
# LOSS AND METRICS
# ============================================================================

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and BCE Loss for binary segmentation.
    
    Numerically stable implementation:
    - Smoothing factor prevents division by zero
    - Clamping prevents log(0) in BCE
    - NaN detection with fallback
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred, target):
        """
        Dice loss: 1 - Dice coefficient
        Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)
        
        Uses higher smoothing factor for stability.
        """
        # Apply sigmoid and clamp to avoid extreme values
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient with smoothing
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss (1 - Dice coefficient)
        return 1.0 - dice
    
    def forward(self, pred, target):
        """
        Combined loss with numerical stability checks.
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        # Check for NaN values
        if torch.isnan(dice) or torch.isnan(bce):
            print(f"Warning: NaN detected in loss! Dice: {dice.item()}, BCE: {bce.item()}")
            # Return only the valid component or a default value
            if torch.isnan(dice) and not torch.isnan(bce):
                return bce
            elif torch.isnan(bce) and not torch.isnan(dice):
                return dice
            else:
                # Both are NaN, return a small default value
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        total_loss = self.dice_weight * dice + self.bce_weight * bce
        return total_loss


def calculate_f1(pred, target, threshold=0.5):
    """
    Calculate F1 score with handling for empty ground truth.
    
    Returns:
        tuple: (f1_score, has_positives)
        - has_positives=False indicates ground truth has no roots (empty patch)
    """
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Check if ground truth has any positives
    positives = torch.sum(torch.round(torch.clamp(target_flat, 0, 1)))
    has_positives = positives > 0
    
    # If no positives in ground truth, F1 is undefined
    # Return 1.0 if model correctly predicts nothing, 0.0 otherwise
    if not has_positives:
        pred_positives = torch.sum(torch.round(torch.clamp(pred_flat, 0, 1)))
        # Model correctly predicts empty patch
        if pred_positives < 10:  # Allow small false positives
            return 1.0, False
        else:
            return 0.0, False
    
    # Standard F1 calculation for patches with roots
    tp = torch.sum(torch.round(torch.clamp(target_flat * pred_flat, 0, 1)))
    pred_positives = torch.sum(torch.round(torch.clamp(pred_flat, 0, 1)))
    
    epsilon = 1e-7
    precision = tp / (pred_positives + epsilon)
    recall = tp / (positives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return f1.item(), True


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Stop training when validation metric stops improving.
    
    Benefits:
    - Prevents overfitting
    - Saves training time
    - Automatically selects best model
    """
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize (F1), 'min' for metrics to minimize (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """Train for one epoch, tracking only loss and F1 score."""
    model.train()
    total_loss = 0
    total_f1 = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with cuda_autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected, skipping batch")
            continue
        
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate F1 score - unpack tuple
        f1, _ = calculate_f1(outputs, masks)
        
        total_loss += loss.item()
        total_f1 += f1
        
        # Format loss for display
        loss_val = loss.item()
        loss_str = f'{loss_val:.2e}' if loss_val < 0.01 else f'{loss_val:.4f}'
        
        pbar.set_postfix({
            'loss': loss_str,
            'f1': f'{f1:.4f}',
        })
    
    n = len(train_loader)
    return {
        'loss': total_loss / n,
        'f1': total_f1 / n,
    }


@torch.no_grad()
@torch.no_grad()
def validate_epoch(model, val_loader, criterion, device):
    """Validate with separate tracking for patches with/without roots."""
    model.eval()
    total_loss = 0
    
    # Separate F1 tracking
    f1_with_roots = 0
    f1_empty_patches = 0
    n_with_roots = 0
    n_empty = 0
    
    pbar = tqdm(val_loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Calculate F1 score with empty patch detection
        f1, has_positives = calculate_f1(outputs, masks)
        
        if has_positives:
            f1_with_roots += f1
            n_with_roots += 1
        else:
            f1_empty_patches += f1
            n_empty += 1
        
        total_loss += loss.item()
        
        # Display current batch metrics
        loss_str = f'{loss.item():.2e}' if loss.item() < 0.01 else f'{loss.item():.4f}'
        pbar.set_postfix({
            'loss': loss_str,
            'f1': f'{f1:.4f}',
            'has_roots': has_positives
        })
    
    n_total = len(val_loader)
    
    # Calculate average F1 scores
    avg_f1_with_roots = f1_with_roots / n_with_roots if n_with_roots > 0 else 0
    avg_f1_empty = f1_empty_patches / n_empty if n_empty > 0 else 0
    overall_f1 = (f1_with_roots + f1_empty_patches) / n_total
    
    print(f"\n  Batches with roots: {n_with_roots} (F1: {avg_f1_with_roots:.4f})")
    print(f"  Batches without roots: {n_empty} (F1: {avg_f1_empty:.4f})")
    
    return {
        'loss': total_loss / n_total,
        'f1': overall_f1,
        'f1_with_roots': avg_f1_with_roots,
        'f1_empty': avg_f1_empty,
        'n_with_roots': n_with_roots,
        'n_empty': n_empty
    }


def train_model(
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    output_dir='./checkpoints_custom',
    # Architecture
    init_filters=64,
    # Training
    batch_size=16,
    num_epochs=100,
    learning_rate=1e-3,
    num_workers=8,
    # Loss
    dice_weight=0.5,
    bce_weight=0.5,
    # Filtering
    filter_empty_patches=True,
    min_positive_pixels=10,
    use_cache=True,
    train_empty_patch_ratio=0.45,  # Separate for training
    val_empty_patch_ratio=0.0,     # Separate for validation
    # Early stopping
    early_stopping_patience=15,
    early_stopping_min_delta=0.001,
):
    """
    Train ResAttentionUNet from scratch.
    
    Why separate empty_patch_ratio for train/val:
    - Training: Include some empty patches (e.g., 0.45) to teach the model what "no roots" looks like
    - Validation: Use fewer or no empty patches (e.g., 0.0) to focus evaluation on actual detection performance
    
    empty_patch_ratio interpretation:
    - 0.5 = 50% as many empty patches as patches with roots
    - 1.0 = equal number (1:1 ratio)
    - 2.0 = twice as many empty patches
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup cache
    cache_dir = output_dir / 'cache'
    train_cache = cache_dir / 'train_valid_patches.json' if use_cache else None
    val_cache = cache_dir / 'val_valid_patches.json' if use_cache else None
    
    # Create augmentation for training only
    train_augmentation = ShootAugmentation(
        horizontal_flip_prob=0.5,
        max_rotation_degrees=15,
        brightness_range=0.15,
        contrast_range=0.1
    )
    
    # Create datasets with separate empty patch ratios
    print("\n--- Training Dataset ---")
    train_dataset = RootSegmentationDataset(
        train_image_dir, train_mask_dir,
        filter_empty=filter_empty_patches,
        min_positive_pixels=min_positive_pixels,
        empty_patch_ratio=train_empty_patch_ratio,  
        cache_file=train_cache,
        augmentation=None  
    )
    
    print("\n--- Validation Dataset ---")
    val_dataset = RootSegmentationDataset(
        val_image_dir, val_mask_dir,
        filter_empty=filter_empty_patches,
        min_positive_pixels=min_positive_pixels,
        empty_patch_ratio=val_empty_patch_ratio, 
        cache_file=val_cache,
        augmentation=None 
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # Create model
    print(f"\nCreating ResAttentionUNet with {init_filters} initial filters...")
    model = ResAttentionUNet(in_channels=1, out_channels=1, init_filters=init_filters)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    scaler = GradScaler('cuda')
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode='max'  # Maximize F1 score
    )
    
    # Training loop
    print(f"\nStarting training with early stopping (patience={early_stopping_patience})...")
    best_f1 = 0.0
    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        history['train_loss'].append(train_metrics['loss'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Format loss for printing (scientific notation if < 0.01)
        train_loss_str = f"{train_metrics['loss']:.2e}" if train_metrics['loss'] < 0.01 else f"{train_metrics['loss']:.4f}"
        val_loss_str = f"{val_metrics['loss']:.2e}" if val_metrics['loss'] < 0.01 else f"{val_metrics['loss']:.4f}"
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss_str}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_loss_str}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'metrics': val_metrics,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"Saved best model (F1: {best_f1:.4f})")
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, output_dir / 'latest_model.pth')
        
        # Save history
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Check early stopping
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"No improvement in validation F1 for {early_stopping_patience} epochs")
            break
    
    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*50)
    
    return history


if __name__ == "__main__":
    config = {
        'train_image_dir': '../data/processed/dataset_unified_patches/train_images',
        'train_mask_dir': '../data/processed/dataset_unified_patches/train_masks',
        'val_image_dir': '../data/processed/dataset_unified_patches/val_images',
        'val_mask_dir': '../data/processed/dataset_unified_patches/val_masks',
        'output_dir': './checkpoints_custom',
        
        # Architecture
        'init_filters': 64,
        
        # Training
        'batch_size': 16,  
        'num_epochs': 100,
        'learning_rate': 3e-4,
        'num_workers': 8,
        
        # Loss
        'dice_weight': 0.5,
        'bce_weight': 0.5,
        
        # Filtering - separate ratios for train and val
        'filter_empty_patches': True,
        'min_positive_pixels': 15,
        'use_cache': True,
        
        # Training: Moderate empty patch ratio to avoid overwhelming the model during learning
        'train_empty_patch_ratio': 0.75,  
        
        # Validation: Match real-world distribution of 75% empty patches
        'val_empty_patch_ratio': 3.0,
        
        # Early stopping
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.005,
    }
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    print("\n")
    
    history = train_model(**config)