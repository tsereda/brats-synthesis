#!/usr/bin/env python3
"""
Enhanced medical image synthesis script with SSIM evaluation
Adds comprehensive metrics including SSIM for Fast-CWDM evaluation
Now supports both real synthesis and evaluation modes
"""

import argparse
import nibabel as nib
import numpy as np
import os
import torch as th
import glob
import sys
import random

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

# Constants
MODALITIES = ['t1n', 't1c', 't2w', 't2f']

class ComprehensiveMetrics:
    """Calculate comprehensive metrics for synthesis evaluation"""
    
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
        """Calculate L1, MSE, PSNR, and SSIM metrics"""
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
            
            # Basic metrics
            l1_loss = F.l1_loss(predicted, target).item()
            mse_loss = F.mse_loss(predicted, target).item()
            
            # PSNR
            try:
                psnr_score = self.psnr_metric(y_pred=predicted, y=target).mean().item()
            except Exception as e:
                print(f"  Warning: PSNR calculation failed for {case_name}: {e}")
                psnr_score = 0.0
            
            # SSIM
            try:
                ssim_score = self.ssim_metric(y_pred=predicted, y=target).mean().item()
            except Exception as e:
                print(f"  Warning: SSIM calculation failed for {case_name}: {e}")
                ssim_score = 0.0
            
            metrics = {
                'l1': l1_loss,
                'mse': mse_loss,
                'psnr': psnr_score,
                'ssim': ssim_score
            }
            
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
    pattern = f"brats_{missing_modality}_BEST_*.pt"
    best_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if best_files:
        checkpoint = best_files[0]
        print(f"Found BEST checkpoint: {checkpoint}")
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
    """Parse checkpoint filename to get training parameters."""
    basename = os.path.basename(checkpoint_path)
    
    # Default values
    diffusion_steps = 1000
    sample_schedule = "direct"
    
    # Parse filename: brats_t1n_BEST_sampled_10.pt
    if "_BEST_" in basename:
        parts = basename.split('_')
        if len(parts) >= 4:
            sample_schedule = parts[3]
        if len(parts) >= 5:
            try:
                diffusion_steps = int(parts[4].split('.')[0])
            except ValueError:
                pass
    
    print(f"Checkpoint config: schedule={sample_schedule}, steps={diffusion_steps}")
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


def synthesize_modality(available_modalities, missing_modality, checkpoint_path, device, metrics_calculator=None, target_data=None):
    """Synthesize the missing modality with comprehensive metrics."""
    print(f"\n=== Synthesizing {missing_modality} ===")
    
    # Parse checkpoint info
    sample_schedule, diffusion_steps = parse_checkpoint_info(checkpoint_path)
    
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
        print(f"Calculating comprehensive metrics...")
        metrics = metrics_calculator.calculate_metrics(
            spatial_sample, target_data, f"{missing_modality}_synthesis"
        )
        print(f"  L1: {metrics['l1']:.6f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
    
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
                evaluation_mode=False, target_modality=None):
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
            metrics_calculator, target_data
        )
        
        # Save result (skip in evaluation mode to avoid overwriting originals)
        if not evaluation_mode:
            save_result(synthesized, case_dir, missing_modality, output_dir)
        else:
            print(f"📊 Evaluation mode: skipping file save for {case_name}")
        
        print(f"✅ Successfully processed {case_name}")
        return True, metrics
        
    except Exception as e:
        print(f"❌ Error processing {case_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    parser = argparse.ArgumentParser(description="Enhanced medical image synthesis with comprehensive metrics")
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
    
    args = parser.parse_args()
    
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.evaluation_mode:
        print(f"🧪 EVALUATION MODE: Using complete dataset with artificial exclusion")
        print(f"   Target modality: {args.target_modality or 'random'}")
        print(f"   Random seed: {args.seed}")
        random.seed(args.seed)  # For reproducible evaluation
        # Force metrics calculation in evaluation mode
        args.evaluate_metrics = True
    else:
        print(f"🔧 SYNTHESIS MODE: Using incomplete dataset")
    
    print(f"🔧 Enhanced synthesis with comprehensive metrics")
    
    # Initialize metrics calculator
    metrics_calculator = ComprehensiveMetrics(device) if args.evaluate_metrics else None
    
    # Find cases
    case_dirs = [d for d in os.listdir(args.input_dir) 
                if os.path.isdir(os.path.join(args.input_dir, d))]
    case_dirs.sort()
    
    if args.max_cases:
        case_dirs = case_dirs[:args.max_cases]
    
    print(f"Found {len(case_dirs)} cases")
    
    # Process cases
    if not args.evaluation_mode:
        os.makedirs(args.output_dir, exist_ok=True)
    
    successful = 0
    all_metrics = {modality: [] for modality in MODALITIES}
    
    for case_dir_name in case_dirs:
        case_dir = os.path.join(args.input_dir, case_dir_name)
        success, metrics = process_case(
            case_dir, args.output_dir, args.checkpoint_dir, device,
            metrics_calculator, args.evaluation_mode, args.target_modality
        )
        
        if success:
            successful += 1
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
    
    print(f"\n=== Summary ===")
    print(f"Successful: {successful}/{len(case_dirs)}")
    
    # Print comprehensive metrics summary
    if args.evaluate_metrics and any(all_metrics.values()):
        print(f"\n=== COMPREHENSIVE METRICS SUMMARY ===")
        for modality, metrics_list in all_metrics.items():
            if metrics_list:
                print(f"\n{modality.upper()} Synthesis:")
                avg_metrics = {
                    'l1': np.mean([m['l1'] for m in metrics_list]),
                    'mse': np.mean([m['mse'] for m in metrics_list]),
                    'psnr': np.mean([m['psnr'] for m in metrics_list]),
                    'ssim': np.mean([m['ssim'] for m in metrics_list])
                }
                std_metrics = {
                    'l1': np.std([m['l1'] for m in metrics_list]),
                    'mse': np.std([m['mse'] for m in metrics_list]),
                    'psnr': np.std([m['psnr'] for m in metrics_list]),
                    'ssim': np.std([m['ssim'] for m in metrics_list])
                }
                print(f"  L1:   {avg_metrics['l1']:.6f} ± {std_metrics['l1']:.6f}")
                print(f"  MSE:  {avg_metrics['mse']:.6f} ± {std_metrics['mse']:.6f}")
                print(f"  PSNR: {avg_metrics['psnr']:.2f} ± {std_metrics['psnr']:.2f} dB")
                print(f"  SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim']:.4f}")
                print(f"  Cases: {len(metrics_list)}")


if __name__ == "__main__":
    main()