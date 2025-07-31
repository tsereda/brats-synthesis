#!/usr/bin/env python3
"""
Enhanced medical image synthesis script with SSIM evaluation and Wandb integration
Adds comprehensive metrics including SSIM for Fast-CWDM evaluation
Now supports both real synthesis and evaluation modes
FIXED: Proper checkpoint parsing for sampled_X.pt format
ENHANCED: Brain masking for accurate clinical evaluation
NEW: Wandb integration for experiment tracking and metrics logging
"""

import argparse
import nibabel as nib
import numpy as np
import os
import torch as th
import glob
import sys
import random
import re
import time
import wandb

sys.path.append(".")

from fast_cwdm.guided_diffusion import dist_util
from fast_cwdm.guided_diffusion.script_util import (
    model_and_diffusion_defaults, 
    create_model_and_diffusion,
    args_to_dict
)
from fast_cwdm.guided_diffusion.bratsloader import clip_and_normalize
from fast_cwdm.DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from monai.metrics import SSIMMetric, PSNRMetric
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not available. Visual sampling will be disabled.")
    Image = None

# Constants
MODALITIES = ['t1n', 't1c', 't2w', 't2f']

def create_brain_mask_from_target(target, threshold=0.01):
    """Create brain mask from target image"""
    if target.dim() > 3:
        # Remove batch/channel dimensions for mask creation
        target_for_mask = target.squeeze()
    else:
        target_for_mask = target
    
    brain_mask = (target_for_mask > threshold).float()
    
    # Ensure mask has same dimensions as target
    while brain_mask.dim() < target.dim():
        brain_mask = brain_mask.unsqueeze(0)
    
    return brain_mask

class ComprehensiveMetrics:
    """Calculate comprehensive metrics for synthesis evaluation with brain masking"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.ssim_metric = SSIMMetric(
            spatial_dims=3,
            data_range=1.0,
            win_size=7,  # Smaller window for medical images
            k1=0.01,
            k2=0.03
        )
        self.psnr_metric = PSNRMetric(max_val=1.0)
        
    def calculate_metrics(self, predicted, target, case_name=""):
        """Calculate L1, MSE, PSNR, and SSIM metrics with brain masking"""
        metrics = {}
        
        with th.no_grad():
            # Ensure tensors are on the same device
            predicted = predicted.to(self.device)
            target = target.to(self.device)
            
            # Add channel dimension if needed
            if predicted.dim() == 3:  # [H, W, D]
                predicted = predicted.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
            elif predicted.dim() == 4:  # [B, H, W, D] or [1, H, W, D]
                predicted = predicted.unsqueeze(1)  # [B, 1, H, W, D]
                
            if target.dim() == 3:
                target = target.unsqueeze(0).unsqueeze(0)
            elif target.dim() == 4:
                target = target.unsqueeze(1)
            
            # CREATE BRAIN MASK FROM TARGET (GROUND TRUTH)
            brain_mask = create_brain_mask_from_target(target, threshold=0.01)
            
            # APPLY MASK TO BOTH PREDICTED AND TARGET
            predicted_masked = predicted * brain_mask
            target_masked = target * brain_mask
            
            # Calculate metrics on MASKED images only
            l1_loss = F.l1_loss(predicted_masked, target_masked).item()
            mse_loss = F.mse_loss(predicted_masked, target_masked).item()
            
            # PSNR on masked images
            try:
                psnr_score = self.psnr_metric(y_pred=predicted_masked, y=target_masked).mean().item()
            except Exception as e:
                print(f"  Warning: PSNR calculation failed for {case_name}: {e}")
                psnr_score = 0.0
            
            # SSIM on masked images - KEY IMPROVEMENT!
            try:
                ssim_score = self.ssim_metric(y_pred=predicted_masked, y=target_masked).mean().item()
            except Exception as e:
                print(f"  Warning: SSIM calculation failed for {case_name}: {e}")
                ssim_score = 0.0
            
            # Calculate brain volume for debugging/reporting
            brain_volume = brain_mask.sum().item()
            total_volume = brain_mask.numel()
            brain_ratio = brain_volume / total_volume
            
            metrics = {
                'l1': l1_loss,
                'mse': mse_loss,
                'psnr': psnr_score,
                'ssim': ssim_score,
                'brain_volume_ratio': brain_ratio
            }
            
            if case_name:
                print(f"  {case_name}: SSIM={ssim_score:.4f} (brain region = {brain_ratio:.1%})")
            
        return metrics


def load_image(file_path):
    """Load and preprocess image EXACTLY like training dataloader."""
    print(f"Loading: {file_path}")
    
    # Load image
    img = nib.load(file_path).get_fdata()
    print(f"  Original shape: {img.shape}")
    
    # Normalize using EXACT training function
    img_normalized = clip_and_normalize(img)
    
    # Preprocess EXACTLY like training (from bratsloader.py __getitem__)
    img_tensor = th.zeros(1, 240, 240, 160)
    img_tensor[:, :, :, :155] = th.tensor(img_normalized)
    img_tensor = img_tensor[:, 8:-8, 8:-8, :]  # ✅ MATCHES training exactly
    
    print(f"  Preprocessed shape: {img_tensor.shape}")
    return img_tensor.float()


def find_missing_modality(case_dir, evaluation_mode=False, target_modality=None):
    """Find which modality is missing (real) or select one to exclude (evaluation)."""
    case_name = os.path.basename(case_dir)
    
    if evaluation_mode:
        # Evaluation mode: artificially select a modality to exclude
        if target_modality:
            # Use specified target modality
            return target_modality
        else:
            # Randomly select a modality to exclude
            return random.choice(MODALITIES)
    else:
        # Real synthesis mode: find actually missing modality
        for modality in MODALITIES:
            file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
            if not os.path.exists(file_path):
                return modality
        return None


def check_complete_case(case_dir):
    """Check if case has all 4 modalities (for evaluation mode)."""
    case_name = os.path.basename(case_dir)
    
    for modality in MODALITIES:
        file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        if not os.path.exists(file_path):
            return False
    return True


def load_available_modalities(case_dir, missing_modality, evaluation_mode=False):
    """Load all available modalities (excluding the missing/target one)."""
    case_name = os.path.basename(case_dir)
    available = [m for m in MODALITIES if m != missing_modality]
    
    modalities = {}
    for modality in available:
        file_path = os.path.join(case_dir, f"{case_name}-{modality}.nii.gz")
        if os.path.exists(file_path):  # Extra safety check
            modalities[modality] = load_image(file_path)
        elif not evaluation_mode:
            print(f"  Warning: Expected file missing: {file_path}")
    
    return modalities


def find_checkpoint(missing_modality, checkpoint_dir):
    """Find the best checkpoint for the missing modality."""
    # Look for BEST checkpoints first
    pattern = f"brats_{missing_modality}_*.pt"
    best_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if best_files:
        checkpoint = best_files[0]
        print(f"Found checkpoint: {checkpoint}")
        return checkpoint
    
    # Fallback to regular checkpoints
    pattern = f"brats_{missing_modality}_*.pt"
    regular_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not regular_files:
        raise FileNotFoundError(f"No checkpoint found for {missing_modality}")
    
    # Sort by iteration number
    def get_iteration(filename):
        parts = os.path.basename(filename).split('_')
        try:
            return int(parts[2])
        except (IndexError, ValueError):
            return 0
    
    regular_files.sort(key=get_iteration, reverse=True)
    checkpoint = regular_files[0]
    print(f"Found checkpoint: {checkpoint}")
    return checkpoint


def parse_checkpoint_info(checkpoint_path):
    """Parse checkpoint filename to get training parameters - FIXED VERSION."""
    basename = os.path.basename(checkpoint_path)
    
    # Default values
    diffusion_steps = 1000
    sample_schedule = "direct"
    
    print(f"Parsing checkpoint: {basename}")
    
    # Handle different checkpoint naming patterns
    
    # Pattern 1: brats_t1n_BEST100epoch_sampled_10.pt
    if "_BEST100epoch_" in basename:
        parts = basename.split('_')
        if len(parts) >= 4:
            sample_schedule = parts[3]
        if len(parts) >= 5:
            try:
                diffusion_steps = int(parts[4].split('.')[0])
            except ValueError:
                pass
    
    # Pattern 2: brats_t1c_074500_sampled_100.pt (YOUR FORMAT)
    elif "_sampled_" in basename:
        parts = basename.split('_')
        for i, part in enumerate(parts):
            if part == "sampled" and i + 1 < len(parts):
                try:
                    diffusion_steps = int(parts[i + 1].split('.')[0])
                    sample_schedule = "direct"  # Assume direct sampling
                    break
                except ValueError:
                    pass
    
    # Pattern 3: Look for any number after "sampled" using regex
    else:
        match = re.search(r'sampled[_-](\d+)', basename)
        if match:
            diffusion_steps = int(match.group(1))
            sample_schedule = "direct"
    
    print(f"✅ Checkpoint config: schedule={sample_schedule}, steps={diffusion_steps}")
    return sample_schedule, diffusion_steps


def create_model_args(sample_schedule="direct", diffusion_steps=1000):
    """Create model arguments."""
    class Args:
        pass
    
    args = Args()
    
    # Model architecture
    args.image_size = 224
    args.num_channels = 64
    args.num_res_blocks = 2
    args.channel_mult = "1,2,2,4,4"
    args.learn_sigma = False
    args.class_cond = False
    args.use_checkpoint = False
    args.attention_resolutions = ""
    args.num_heads = 1
    args.num_head_channels = -1
    args.num_heads_upsample = -1
    args.use_scale_shift_norm = False
    args.dropout = 0.0
    args.resblock_updown = True
    args.use_fp16 = False
    args.use_new_attention_order = False
    args.dims = 3
    args.num_groups = 32
    args.bottleneck_attention = False
    args.resample_2d = False
    args.additive_skips = False
    args.use_freq = False
    
    # Diffusion parameters
    args.predict_xstart = True
    args.noise_schedule = "linear"
    args.timestep_respacing = ""
    args.use_kl = False
    args.rescale_timesteps = False
    args.rescale_learned_sigmas = False
    
    # Model channels: 8 (target) + 24 (3 modalities * 8 DWT components each)
    args.in_channels = 32
    args.out_channels = 8
    
    # From checkpoint
    args.diffusion_steps = diffusion_steps
    args.sample_schedule = sample_schedule
    args.mode = 'i2i'
    args.dataset = "brats"
    
    return args


def create_visual_comparison(available_modalities, synthesized, target_data, missing_modality, case_name, slice_indices=None):
    """Create visual comparison of available modalities, synthesized, and ground truth."""
    
    # Check if PIL is available
    if Image is None:
        print("  Warning: PIL not available, skipping visual comparison")
        return []
    
    # Convert tensors to numpy for visualization
    if isinstance(synthesized, th.Tensor):
        synthesized_np = synthesized.detach().cpu().numpy()
    else:
        synthesized_np = synthesized
        
    if isinstance(target_data, th.Tensor):
        target_np = target_data.detach().cpu().numpy()
    else:
        target_np = target_data
    
    # Get available modality data
    available_np = {}
    for mod_name, mod_data in available_modalities.items():
        if isinstance(mod_data, th.Tensor):
            # Remove batch dimension if present
            mod_np = mod_data.detach().cpu().numpy()
            if mod_np.ndim == 4:  # [B, H, W, D]
                mod_np = mod_np[0]  # [H, W, D]
        else:
            mod_np = mod_data
        available_np[mod_name] = mod_np
    
    # Determine slice indices to visualize
    if slice_indices is None:
        depth = synthesized_np.shape[2]
        # Use middle slice and a few others for good coverage
        slice_indices = [
            depth // 4,      # Quarter
            depth // 2,      # Middle  
            3 * depth // 4   # Three quarters
        ]
    
    visualizations = []
    
    for slice_idx in slice_indices:
        # Create figure with subplots
        n_modalities = len(available_modalities)
        n_cols = n_modalities + 2  # available + synthesized + ground truth
        fig, axes = plt.subplots(1, n_cols, figsize=(3*n_cols, 3))
        
        if n_cols == 1:
            axes = [axes]
        
        col_idx = 0
        
        # Plot available modalities
        modality_order = [m for m in MODALITIES if m != missing_modality]
        for mod_name in modality_order:
            if mod_name in available_np:
                slice_data = available_np[mod_name][:, :, slice_idx]
                axes[col_idx].imshow(slice_data, cmap='gray', vmin=0, vmax=1)
                axes[col_idx].set_title(f'{mod_name.upper()}\n(Available)', fontsize=10)
                axes[col_idx].axis('off')
                col_idx += 1
        
        # Plot synthesized
        synth_slice = synthesized_np[:, :, slice_idx]
        axes[col_idx].imshow(synth_slice, cmap='gray', vmin=0, vmax=1)
        axes[col_idx].set_title(f'{missing_modality.upper()}\n(Synthesized)', fontsize=10, color='blue')
        axes[col_idx].axis('off')
        
        # Add blue border to synthesized
        rect = Rectangle((0, 0), synth_slice.shape[1]-1, synth_slice.shape[0]-1, 
                        linewidth=3, edgecolor='blue', facecolor='none')
        axes[col_idx].add_patch(rect)
        col_idx += 1
        
        # Plot ground truth
        if target_np is not None:
            target_slice = target_np[:, :, slice_idx]
            axes[col_idx].imshow(target_slice, cmap='gray', vmin=0, vmax=1)
            axes[col_idx].set_title(f'{missing_modality.upper()}\n(Ground Truth)', fontsize=10, color='green')
            axes[col_idx].axis('off')
            
            # Add green border to ground truth
            rect = Rectangle((0, 0), target_slice.shape[1]-1, target_slice.shape[0]-1, 
                            linewidth=3, edgecolor='green', facecolor='none')
            axes[col_idx].add_patch(rect)
        
        # Add case info and slice number
        fig.suptitle(f'{case_name} - Slice {slice_idx}/{synthesized_np.shape[2]-1}', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to wandb Image using PIL
        import io
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to PIL Image, then to wandb Image
        pil_image = Image.open(buf)
        wandb_image = wandb.Image(pil_image, caption=f"{case_name}_slice_{slice_idx}")
        visualizations.append(wandb_image)
        
        plt.close(fig)  # Free memory
    
    return visualizations


def create_difference_maps(synthesized, target_data, missing_modality, case_name, slice_indices=None):
    """Create difference maps between synthesized and ground truth."""
    
    # Check if PIL is available
    if Image is None:
        print("  Warning: PIL not available, skipping difference maps")
        return []
    
    # Convert to numpy
    if isinstance(synthesized, th.Tensor):
        synthesized_np = synthesized.detach().cpu().numpy()
    else:
        synthesized_np = synthesized
        
    if isinstance(target_data, th.Tensor):
        target_np = target_data.detach().cpu().numpy()
    else:
        target_np = target_data
    
    if target_np is None:
        return []
    
    # Calculate absolute difference
    diff_np = np.abs(synthesized_np - target_np)
    
    # Determine slice indices
    if slice_indices is None:
        depth = synthesized_np.shape[2]
        slice_indices = [depth // 2]  # Just middle slice for difference
    
    visualizations = []
    
    for slice_idx in slice_indices:
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        
        # Synthesized
        synth_slice = synthesized_np[:, :, slice_idx]
        im1 = axes[0].imshow(synth_slice, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Synthesized', fontsize=10)
        axes[0].axis('off')
        
        # Ground truth
        target_slice = target_np[:, :, slice_idx]
        im2 = axes[1].imshow(target_slice, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Ground Truth', fontsize=10)
        axes[1].axis('off')
        
        # Absolute difference
        diff_slice = diff_np[:, :, slice_idx]
        im3 = axes[2].imshow(diff_slice, cmap='hot', vmin=0, vmax=0.3)  # Scale for visibility
        axes[2].set_title('|Difference|', fontsize=10)
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Overlay difference on synthesized
        overlay = axes[3].imshow(synth_slice, cmap='gray', vmin=0, vmax=1, alpha=0.7)
        overlay2 = axes[3].imshow(diff_slice, cmap='Reds', vmin=0, vmax=0.3, alpha=0.5)
        axes[3].set_title('Overlay', fontsize=10)
        axes[3].axis('off')
        
        # Add case info
        fig.suptitle(f'{case_name} - {missing_modality.upper()} Difference Analysis - Slice {slice_idx}', 
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to wandb Image using PIL
        import io
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to PIL Image, then to wandb Image
        pil_image = Image.open(buf)
        wandb_image = wandb.Image(pil_image, caption=f"{case_name}_{missing_modality}_diff_slice_{slice_idx}")
        visualizations.append(wandb_image)
        
        plt.close(fig)
    
    return visualizations


def log_visual_samples(available_modalities, synthesized, target_data, missing_modality, case_name, 
                      log_to_wandb=True, save_locally=False, output_dir=None):
    """Log visual samples to wandb and optionally save locally."""
    
    print(f"  📸 Creating visual samples for {case_name}...")
    
    try:
        # Create comparison visualizations
        comparison_images = create_visual_comparison(
            available_modalities, synthesized, target_data, missing_modality, case_name
        )
        
        # Create difference maps if target is available
        difference_images = []
        if target_data is not None:
            difference_images = create_difference_maps(
                synthesized, target_data, missing_modality, case_name
            )
        
        # Log to wandb
        if log_to_wandb and (comparison_images or difference_images):
            wandb_log_data = {}
            
            if comparison_images:
                wandb_log_data[f"samples/{missing_modality}/comparison"] = comparison_images
                
            if difference_images:
                wandb_log_data[f"samples/{missing_modality}/difference"] = difference_images
            
            # Add case metadata
            wandb_log_data[f"samples/case_name"] = case_name
            wandb_log_data[f"samples/modality"] = missing_modality
            
            wandb.log(wandb_log_data)
        
        # Save locally if requested
        if save_locally and output_dir:
            samples_dir = os.path.join(output_dir, "visual_samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save comparison images
            for i, img in enumerate(comparison_images):
                # Note: wandb.Image doesn't easily convert back to file
                # Would need to recreate the plots for local saving
                pass  # Implement if needed
        
        print(f"  ✅ Visual samples created and logged")
        
    except Exception as e:
        print(f"  ⚠️ Error creating visual samples: {e}")
        import traceback
        traceback.print_exc()


def prepare_conditioning(available_modalities, missing_modality, device):
    """Prepare conditioning tensor from available modalities."""
    dwt = DWT_3D("haar")
    
    # Get modalities in consistent order
    available_order = [m for m in MODALITIES if m != missing_modality]
    print(f"Available modalities: {available_order}")
    
    cond_list = []
    
    for modality in available_order:
        # Get tensor and add channel dimension
        tensor = available_modalities[modality].to(device)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)  # [B, 1, D, H, W]
        
        print(f"  {modality} input shape: {tensor.shape}")
        
        # Apply DWT
        dwt_components = dwt(tensor)
        shapes = [c.shape for c in dwt_components]
        print(f"  {modality} DWT shapes: {shapes}")
        
        # Find minimum z-dimension to fix mismatches
        min_z = min(c.shape[-1] for c in dwt_components)
        print(f"  {modality} min z-dimension: {min_z}")
        
        # Crop all components to same size
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt_components
        cropped_components = [
            LLL[:, :, :, :, :min_z] / 3.,  # Divide LLL by 3 as per training
            LLH[:, :, :, :, :min_z],
            LHL[:, :, :, :, :min_z],
            LHH[:, :, :, :, :min_z],
            HLL[:, :, :, :, :min_z],
            HLH[:, :, :, :, :min_z],
            HHL[:, :, :, :, :min_z],
            HHH[:, :, :, :, :min_z]
        ]
        
        # Concatenate DWT components
        modality_cond = th.cat(cropped_components, dim=1)
        print(f"  {modality} final conditioning: {modality_cond.shape}")
        cond_list.append(modality_cond)
    
    # Concatenate all modalities
    cond = th.cat(cond_list, dim=1)
    print(f"Final conditioning shape: {cond.shape}")
    
    return cond


def synthesize_modality(available_modalities, missing_modality, checkpoint_path, device, metrics_calculator=None, target_data=None, override_steps=None):
    """Synthesize the missing modality with comprehensive metrics."""
    print(f"\n=== Synthesizing {missing_modality} ===")
    
    # Parse checkpoint info
    sample_schedule, diffusion_steps = parse_checkpoint_info(checkpoint_path)
    
    # Override steps if specified
    if override_steps:
        diffusion_steps = override_steps
        print(f"🔧 Overriding diffusion steps to: {diffusion_steps}")
    
    # Create model
    args = create_model_args(sample_schedule, diffusion_steps)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    
    # Load weights
    print(f"Loading model from: {checkpoint_path}")
    state_dict = dist_util.load_state_dict(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Prepare conditioning
    cond = prepare_conditioning(available_modalities, missing_modality, device)
    
    # Create noise tensor with matching dimensions
    _, _, cond_d, cond_h, cond_w = cond.shape
    noise_shape = (1, 8, cond_d, cond_h, cond_w)
    noise = th.randn(*noise_shape, device=device)
    
    print(f"Noise shape: {noise.shape}")
    print(f"Conditioning shape: {cond.shape}")
    
    # Sample using p_sample_loop_progressive (correct method for Fast-DDPM)
    print(f"Running {diffusion.num_timesteps}-step sampling...")
    
    # Start timing
    sample_start_time = time.time()
    
    with th.no_grad():
        final_sample = None
        for sample_dict in diffusion.p_sample_loop_progressive(
            model=model,
            shape=noise.shape,
            time=diffusion.num_timesteps,  # ✅ Correct parameter for fast sampling
            noise=noise,
            cond=cond,
            clip_denoised=True,
            model_kwargs={}
        ):
            final_sample = sample_dict
        
        sample = final_sample["sample"]
    
    # End timing
    sample_end_time = time.time()
    sample_duration = sample_end_time - sample_start_time
    
    print(f"⏱Sampling completed in {sample_duration:.2f} seconds ({sample_duration/60:.2f} minutes)")
    
    # Return timing info along with other data
    timing_info = {
        'sample_time': sample_duration,
        'steps': diffusion.num_timesteps
    }
    
    print(f"Sample shape: {sample.shape}")
    
    # Convert back to spatial domain using IDWT
    idwt = IDWT_3D("haar")
    B, _, D, H, W = sample.shape
    
    spatial_sample = idwt(
        sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,  # Multiply LLL by 3
        sample[:, 1, :, :, :].view(B, 1, D, H, W),
        sample[:, 2, :, :, :].view(B, 1, D, H, W),
        sample[:, 3, :, :, :].view(B, 1, D, H, W),
        sample[:, 4, :, :, :].view(B, 1, D, H, W),
        sample[:, 5, :, :, :].view(B, 1, D, H, W),
        sample[:, 6, :, :, :].view(B, 1, D, H, W),
        sample[:, 7, :, :, :].view(B, 1, D, H, W)
    )
    
    print(f"Spatial sample shape: {spatial_sample.shape}")
    
    # Post-process
    spatial_sample = th.clamp(spatial_sample, 0, 1)
    
    # Apply brain mask from first available modality
    first_modality = list(available_modalities.values())[0].to(device)
    if first_modality.dim() == 4:
        first_modality = first_modality.unsqueeze(1)
    
    spatial_sample[first_modality == 0] = 0
    
    # Remove batch and channel dimensions
    if spatial_sample.dim() == 5:
        spatial_sample = spatial_sample.squeeze(1)  # Remove channel
    spatial_sample = spatial_sample[0]  # Remove batch
    
    print(f"Final output shape: {spatial_sample.shape}")
    
    # Calculate comprehensive metrics if target is provided
    metrics = {}
    if metrics_calculator is not None and target_data is not None:
        print(f"Calculating brain-masked metrics...")
        metrics = metrics_calculator.calculate_metrics(
            spatial_sample, target_data, f"{missing_modality}_synthesis"
        )
        print(f"  L1: {metrics['l1']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f} (brain-masked)")
        print(f"  Brain volume: {metrics['brain_volume_ratio']:.1%} of total")
    
    # Add timing info to metrics
    if metrics:
        metrics.update(timing_info)
    else:
        metrics = timing_info
    
    return spatial_sample, metrics


def save_result(synthesized, case_dir, missing_modality, output_dir):
    """Save the synthesized modality."""
    case_name = os.path.basename(case_dir)
    
    # Create output directory
    output_case_dir = os.path.join(output_dir, case_name)
    os.makedirs(output_case_dir, exist_ok=True)
    
    # Copy existing files
    for filename in os.listdir(case_dir):
        if filename.endswith('.nii.gz'):
            src = os.path.join(case_dir, filename)
            dst = os.path.join(output_case_dir, filename)
            nib.save(nib.load(src), dst)
    
    # Save synthesized modality
    output_path = os.path.join(output_case_dir, f"{case_name}-{missing_modality}.nii.gz")
    
    # Get reference for header/affine
    reference_files = [f for f in os.listdir(case_dir) 
                      if f.endswith('.nii.gz') and any(m in f for m in MODALITIES)]
    
    if reference_files:
        reference_img = nib.load(os.path.join(case_dir, reference_files[0]))
        
        # Convert to numpy
        synthesized_np = synthesized.detach().cpu().numpy()
        
        # Handle z-dimension: model outputs 160 slices, original data has 155
        if synthesized_np.shape[2] == 160:
            print(f"  Converting from 160 to 155 slices")
            synthesized_np = synthesized_np[:, :, :155]
        
        # Pad back to 240x240x155 (reverse the 8-pixel crop)
        padded = np.zeros((240, 240, 155))
        padded[8:232, 8:232, :] = synthesized_np
        
        # Create NIfTI image
        synthesized_img = nib.Nifti1Image(padded, reference_img.affine, reference_img.header)
    else:
        synthesized_np = synthesized.detach().cpu().numpy()
        if synthesized_np.shape[2] == 160:
            synthesized_np = synthesized_np[:, :, :155]
        synthesized_img = nib.Nifti1Image(synthesized_np, np.eye(4))
    
    nib.save(synthesized_img, output_path)
    print(f"✅ Saved: {output_path}")


def process_case(case_dir, output_dir, checkpoint_dir, device, metrics_calculator=None, 
                evaluation_mode=False, target_modality=None, override_steps=None, 
                case_index=0, visual_args=None):
    """Process a single case with optional metrics evaluation."""
    case_name = os.path.basename(case_dir)
    print(f"\n=== Processing {case_name} ===")
    
    # Check if case is complete (for evaluation mode)
    if evaluation_mode and not check_complete_case(case_dir):
        print(f"Skipping incomplete case in evaluation mode: {case_name}")
        return False, {}
    
    # Find missing modality (real or artificial)
    missing_modality = find_missing_modality(case_dir, evaluation_mode, target_modality)
    if not missing_modality:
        print(f"No missing modality in {case_name}")
        return False, {}
    
    print(f"{'Target' if evaluation_mode else 'Missing'} modality: {missing_modality}")
    
    try:
        # Load available modalities (excluding the target one)
        available_modalities = load_available_modalities(case_dir, missing_modality, evaluation_mode)
        
        if len(available_modalities) != 3:
            print(f"❌ Expected 3 available modalities, got {len(available_modalities)}")
            return False, {}
        
        # Load target for evaluation if requested
        target_data = None
        if evaluation_mode and metrics_calculator:
            # In evaluation mode, load the "missing" modality as ground truth
            target_file = os.path.join(case_dir, f"{case_name}-{missing_modality}.nii.gz")
            if os.path.exists(target_file):
                print(f"Loading ground truth: {target_file}")
                target_data = load_image(target_file)
                target_data = target_data[0]  # Remove batch dimension
            else:
                print(f"❌ Ground truth file not found: {target_file}")
        
        # Find checkpoint
        checkpoint_path = find_checkpoint(missing_modality, checkpoint_dir)
        
        # Synthesize
        synthesized, metrics = synthesize_modality(
            available_modalities, missing_modality, checkpoint_path, device,
            metrics_calculator, target_data, override_steps
        )
        
        # Log case-level metrics to wandb
        if metrics and 'l1' in metrics:
            case_metrics = {
                f"case/{missing_modality}/l1": metrics['l1'],
                f"case/{missing_modality}/mse": metrics['mse'],
                f"case/{missing_modality}/psnr": metrics['psnr'],
                f"case/{missing_modality}/ssim": metrics['ssim'],
                f"case/{missing_modality}/brain_volume_ratio": metrics['brain_volume_ratio'],
                f"case/{missing_modality}/sample_time": metrics['sample_time'],
                f"case/step": metrics.get('steps', 0),
                "case/name": case_name,
                "case/modality": missing_modality
            }
            wandb.log(case_metrics)
        
        # Create and log visual samples (if enabled and appropriate)
        should_log_visuals = (
            evaluation_mode and  # Only in evaluation mode (we have ground truth)
            visual_args and 
            visual_args.get('log_visual_samples', False) and
            (case_index % visual_args.get('visual_sample_frequency', 1) == 0)
        )
        
        if should_log_visuals:
            log_visual_samples(
                available_modalities=available_modalities,
                synthesized=synthesized,
                target_data=target_data,
                missing_modality=missing_modality,
                case_name=case_name,
                log_to_wandb=True,
                save_locally=visual_args.get('save_visual_samples', False),
                output_dir=output_dir
            )
        
        # Save result (skip in evaluation mode to avoid overwriting originals)
        # if not evaluation_mode:
        #     save_result(synthesized, case_dir, missing_modality, output_dir)
        # else:
        #     print(f"Evaluation mode: skipping file save for {case_name}")
        save_result(synthesized, case_dir, missing_modality, output_dir)
        print(f"overwriting file save for {case_name}")
        
        print(f"✅ Successfully processed {case_name}")
        return True, metrics
        
    except Exception as e:
        print(f"❌ Error processing {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    parser = argparse.ArgumentParser(description="Enhanced medical image synthesis with brain-masked comprehensive metrics and Wandb integration")
    parser.add_argument("--input_dir", default="./datasets/BRATS2023/pseudo_validation")
    parser.add_argument("--output_dir", default="./datasets/BRATS2023/pseudo_validation_completed")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--evaluate_metrics", action="store_true",
                        help="Calculate comprehensive metrics (requires ground truth)")
    parser.add_argument("--evaluation_mode", action="store_true",
                        help="Evaluation mode: use complete dataset and artificially exclude modalities")
    parser.add_argument("--target_modality", choices=MODALITIES, default=None,
                        help="Specific modality to synthesize in evaluation mode (random if not specified)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible evaluation")
    parser.add_argument("--diffusion_steps", type=int, default=None,
                        help="Override diffusion steps (default: parse from checkpoint)")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", default="fast-cwmd-brats-inference",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", default="timgsereda",
                        help="Wandb entity/username")
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"],
                        help="Wandb logging mode")
    parser.add_argument("--wandb_run_name", default=None,
                        help="Custom run name for wandb")
    parser.add_argument("--wandb_tags", nargs="*", default=[],
                        help="Tags for wandb run")
    
    # Visual sampling arguments
    parser.add_argument("--log_visual_samples", action="store_true", default=True,
                        help="Create and log visual comparison samples to wandb")
    parser.add_argument("--save_visual_samples", action="store_true",
                        help="Save visual samples locally as PNG files")
    parser.add_argument("--visual_sample_frequency", type=int, default=1,
                        help="Log visual samples every N cases (1=every case)")
    
    args = parser.parse_args()
    
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Wandb initialization ---
    wandb_config = vars(args).copy()
    wandb_config['device'] = str(device)
    
    # Create run name if not provided
    if not args.wandb_run_name:
        mode_suffix = "eval" if args.evaluation_mode else "synth"
        target_suffix = f"_{args.target_modality}" if args.target_modality else ""
        steps_suffix = f"_s{args.diffusion_steps}" if args.diffusion_steps else ""
        args.wandb_run_name = f"inference_{mode_suffix}{target_suffix}{steps_suffix}"
    
    # Add automatic tags
    auto_tags = []
    if args.evaluation_mode:
        auto_tags.append("evaluation")
    else:
        auto_tags.append("synthesis")
    
    if args.target_modality:
        auto_tags.append(f"target_{args.target_modality}")
    
    if args.diffusion_steps:
        auto_tags.append(f"steps_{args.diffusion_steps}")
    
    all_tags = args.wandb_tags + auto_tags
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=wandb_config,
        mode=args.wandb_mode,
        name=args.wandb_run_name,
        tags=all_tags
    )
    
    print(f"🔮 Wandb initialized: {wandb.run.name}")
    # --- End wandb initialization ---
    
    if args.evaluation_mode:
        print(f"EVALUATION MODE: Using complete dataset with artificial exclusion")
        print(f"   Target modality: {args.target_modality or 'random'}")
        print(f"   Random seed: {args.seed}")
        random.seed(args.seed)  # For reproducible evaluation
        # Force metrics calculation in evaluation mode
        args.evaluate_metrics = True
    else:
        print(f"SYNTHESIS MODE: Using incomplete dataset")
    
    if args.diffusion_steps:
        print(f"Overriding diffusion steps: {args.diffusion_steps}")
    
    print(f"Synthesis  metrics")
    if args.log_visual_samples and args.evaluation_mode:
        print(f"📸 Visual sampling enabled (frequency: every {args.visual_sample_frequency} case{'s' if args.visual_sample_frequency > 1 else ''})")
    elif args.evaluation_mode:
        print(f"📸 Visual sampling disabled")
    
    # Initialize metrics calculator
    metrics_calculator = ComprehensiveMetrics(device) if args.evaluate_metrics else None
    
    # Find cases
    case_dirs = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d))]
    case_dirs.sort()
    
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]
    
    print(f"Found {len(case_dirs)} cases")
    
    # Log run configuration to wandb
    wandb.log({
        "config/total_cases": len(case_dirs),
        "config/max_cases": args.max_cases or len(case_dirs),
        "config/evaluation_mode": args.evaluation_mode,
        "config/target_modality": args.target_modality or "random",
        "config/seed": args.seed,
        "config/visual_sampling": args.log_visual_samples and args.evaluation_mode,
        "config/visual_frequency": args.visual_sample_frequency if args.log_visual_samples else 0
    })
    
    # Process cases
    if not args.evaluation_mode:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare visual arguments
    visual_args = {
        'log_visual_samples': args.log_visual_samples,
        'save_visual_samples': args.save_visual_samples,
        'visual_sample_frequency': args.visual_sample_frequency
    }
    
    successful = 0
    all_metrics = {modality: [] for modality in MODALITIES}
    sample_times = []
    total_start_time = time.time()
    
    for i, case_dir_name in enumerate(case_dirs):
        case_dir = os.path.join(args.input_dir, case_dir_name)
        
        # Show progress and ETA
        if i > 0 and sample_times:
            avg_time = np.mean(sample_times)
            remaining_cases = len(case_dirs) - i
            eta_seconds = avg_time * remaining_cases
            eta_minutes = eta_seconds / 60
            print(f"\nProgress: {i}/{len(case_dirs)} | Avg: {avg_time:.1f}s/case | ETA: {eta_minutes:.1f} min")
            
            # Log progress to wandb
            wandb.log({
                "progress/processed_cases": i,
                "progress/successful_cases": successful,
                "progress/success_rate": successful / i if i > 0 else 0,
                "progress/avg_sample_time": avg_time,
                "progress/eta_minutes": eta_minutes
            })
        
        success, metrics = process_case(
            case_dir, args.output_dir, args.checkpoint_dir, device,
            metrics_calculator, args.evaluation_mode, args.target_modality, 
            args.diffusion_steps, case_index=i, visual_args=visual_args
        )
        
        if success:
            successful += 1
            # Track sample times
            if 'sample_time' in metrics:
                sample_times.append(metrics['sample_time'])
                print(f"⏱Case sample time: {metrics['sample_time']:.2f}s")
            
            # Collect metrics if available
            if metrics:
                if args.evaluation_mode:
                    # In evaluation mode, we know which modality was synthesized
                    target_mod = find_missing_modality(case_dir, args.evaluation_mode, args.target_modality)
                    all_metrics[target_mod].append(metrics)
                else:
                    # In synthesis mode, find the actual missing modality
                    missing_modality = find_missing_modality(case_dir, False)
                    if missing_modality:
                        all_metrics[missing_modality].append(metrics)
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}/{len(case_dirs)}")
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    
    # Log final summary to wandb
    final_summary = {
        "summary/successful_cases": successful,
        "summary/total_cases": len(case_dirs),
        "summary/success_rate": successful / len(case_dirs) if len(case_dirs) > 0 else 0,
        "summary/total_time_seconds": total_duration,
        "summary/total_time_minutes": total_duration / 60
    }
    
    # Print timing statistics
    if sample_times:
        avg_sample_time = np.mean(sample_times)
        min_sample_time = np.min(sample_times)
        max_sample_time = np.max(sample_times)
        std_sample_time = np.std(sample_times)
        
        print(f"\n=== TIMING STATISTICS ===")
        print(f"Average sample time: {avg_sample_time:.2f} ± {std_sample_time:.2f} seconds")
        print(f"Fastest sample: {min_sample_time:.2f} seconds")
        print(f"Slowest sample: {max_sample_time:.2f} seconds")
        print(f"Total synthesis time: {sum(sample_times):.2f} seconds")
        print(f"Overhead time: {total_duration - sum(sample_times):.2f} seconds")
        
        if successful > 0:
            throughput = successful / (total_duration / 3600)  # cases per hour
            print(f"Throughput: {throughput:.1f} cases/hour")
            
        # Add timing stats to summary
        timing_summary = {
            "summary/avg_sample_time": avg_sample_time,
            "summary/std_sample_time": std_sample_time,
            "summary/min_sample_time": min_sample_time,
            "summary/max_sample_time": max_sample_time,
            "summary/total_synthesis_time": sum(sample_times),
            "summary/overhead_time": total_duration - sum(sample_times),
            "summary/throughput_cases_per_hour": throughput if successful > 0 else 0
        }
        final_summary.update(timing_summary)
    
    # Print comprehensive metrics summary
    if args.evaluate_metrics and any(all_metrics.values()):
        print(f"\n=== METRICS SUMMARY ===")
        for modality, metrics_list in all_metrics.items():
            if metrics_list:
                print(f"\n{modality.upper()} Synthesis:")
                
                # Performance metrics
                if any('l1' in m for m in metrics_list):
                    avg_metrics = {
                        'l1': np.mean([m['l1'] for m in metrics_list if 'l1' in m]),
                        'mse': np.mean([m['mse'] for m in metrics_list if 'mse' in m]),
                        'psnr': np.mean([m['psnr'] for m in metrics_list if 'psnr' in m]),
                        'ssim': np.mean([m['ssim'] for m in metrics_list if 'ssim' in m])
                    }
                    std_metrics = {
                        'l1': np.std([m['l1'] for m in metrics_list if 'l1' in m]),
                        'mse': np.std([m['mse'] for m in metrics_list if 'mse' in m]),
                        'psnr': np.std([m['psnr'] for m in metrics_list if 'psnr' in m]),
                        'ssim': np.std([m['ssim'] for m in metrics_list if 'ssim' in m])
                    }
                    print(f"  L1:   {avg_metrics['l1']:.6f} ± {std_metrics['l1']:.6f}")
                    print(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse']:.6f}")
                    print(f"  PSNR: {avg_metrics['psnr']:.2f} ± {std_metrics['psnr']:.2f} dB")
                    print(f"  SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim']:.4f}")
                    
                    # Log modality-specific metrics to wandb
                    modality_summary = {
                        f"metrics/{modality}/l1_mean": avg_metrics['l1'],
                        f"metrics/{modality}/l1_std": std_metrics['l1'],
                        f"metrics/{modality}/mse_mean": avg_metrics['mse'],
                        f"metrics/{modality}/mse_std": std_metrics['mse'],
                        f"metrics/{modality}/psnr_mean": avg_metrics['psnr'],
                        f"metrics/{modality}/psnr_std": std_metrics['psnr'],
                        f"metrics/{modality}/ssim_mean": avg_metrics['ssim'],
                        f"metrics/{modality}/ssim_std": std_metrics['ssim'],
                    }
                    final_summary.update(modality_summary)
                
                # Brain volume statistics
                brain_ratios = [m['brain_volume_ratio'] for m in metrics_list if 'brain_volume_ratio' in m]
                if brain_ratios:
                    avg_brain_ratio = np.mean(brain_ratios)
                    std_brain_ratio = np.std(brain_ratios)
                    print(f"  Brain volume: {avg_brain_ratio:.1%} ± {std_brain_ratio:.1%} of total")
                    
                    final_summary.update({
                        f"metrics/{modality}/brain_volume_mean": avg_brain_ratio,
                        f"metrics/{modality}/brain_volume_std": std_brain_ratio
                    })
                
                # Timing metrics
                modality_sample_times = [m['sample_time'] for m in metrics_list if 'sample_time' in m]
                if modality_sample_times:
                    avg_time = np.mean(modality_sample_times)
                    std_time = np.std(modality_sample_times)
                    print(f"  Avg sample time: {avg_time:.2f} ± {std_time:.2f} seconds")
                    
                    final_summary.update({
                        f"metrics/{modality}/sample_time_mean": avg_time,
                        f"metrics/{modality}/sample_time_std": std_time
                    })
                
                print(f"  Cases: {len(metrics_list)}")
                final_summary[f"metrics/{modality}/cases"] = len(metrics_list)
    
    # Log final summary to wandb
    wandb.log(final_summary)
    
    # Final visual sampling summary
    if args.log_visual_samples and args.evaluation_mode and successful > 0:
        visual_cases = successful // args.visual_sample_frequency + (1 if successful % args.visual_sample_frequency > 0 else 0)
        print(f"\n📸 Visual samples logged for ~{visual_cases} cases (every {args.visual_sample_frequency} case{'s' if args.visual_sample_frequency > 1 else ''})")
        print(f"   Check wandb 'samples' section for visual comparisons and difference maps")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()