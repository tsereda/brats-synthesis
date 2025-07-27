"""
Shared synthesis utilities for both training validation and inference.
Contains comprehensive metrics calculation with brain masking.
"""

import torch as th
import numpy as np
from fast_cwdm.DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
from monai.metrics import SSIMMetric, PSNRMetric
import torch.nn.functional as F

# VALIDATED WAVELET-FRIENDLY CROP BOUNDS
CROP_BOUNDS = {
    'x_min': 39, 'x_max': 199,  # width: 160 (divisible by 16)
    'y_min': 9, 'y_max': 233,  # height: 224 (divisible by 16)
    'z_min': 0,  'z_max': 160   # depth: 160 (original)
}

ORIGINAL_SHAPE = (240, 240, 160)
CROPPED_SHAPE = (160, 224, 160)
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
            
            # 🧠 CREATE BRAIN MASK FROM TARGET (GROUND TRUTH)
            brain_mask = create_brain_mask_from_target(target, threshold=0.01)
            
            # 🧠 APPLY MASK TO BOTH PREDICTED AND TARGET
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
            
            # SSIM on masked images - KEY METRIC FOR VALIDATION!
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
            
        return metrics


def prepare_conditioning(available_modalities, missing_modality, device):
    """Prepare conditioning tensor from available modalities with FIXED dimensions."""
    dwt = DWT_3D("haar")
    
    # Get modalities in consistent order
    available_order = [m for m in MODALITIES if m != missing_modality]
    
    cond_list = []
    
    for modality in available_order:
        # Get tensor and add channel dimension
        tensor = available_modalities[modality].to(device)
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)  # [B, 1, H, W, D]
        
        # Apply DWT - should work perfectly with (160, 224, 160) dimensions
        dwt_components = dwt(tensor)
        
        # All components should have consistent dimensions now
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt_components
        
        # Create DWT conditioning tensor
        modality_cond = th.cat([
            LLL / 3.,  # Divide LLL by 3 as per training
            LLH, LHL, LHH, HLL, HLH, HHL, HHH
        ], dim=1)
        
        cond_list.append(modality_cond)
    
    # Concatenate all modalities
    cond = th.cat(cond_list, dim=1)
    return cond


def synthesize_modality_shared(model, diffusion, available_modalities, missing_modality, device, 
                              metrics_calculator=None, target_data=None):
    """
    FIXED: Synthesis with proper noise/conditioning separation for i2i mode
    """
    print(f"\n=== Synthesizing {missing_modality} (FIXED PIPELINE) ===")
    
    try:
        # Step 1: Prepare conditioning with detailed checks
        print("Step 1: Preparing conditioning...")
        cond = prepare_conditioning_debug(available_modalities, missing_modality, device)
        print(f"✅ Conditioning shape: {cond.shape}")
        print(f"   Conditioning device: {cond.device}")
        print(f"   Conditioning dtype: {cond.dtype}")
        print(f"   Conditioning range: [{cond.min():.4f}, {cond.max():.4f}]")
        
        # Step 2: Create noise tensor with validation
        print("Step 2: Creating noise tensor...")
        _, _, cond_d, cond_h, cond_w = cond.shape
        noise_shape = (1, 8, cond_d, cond_h, cond_w)  # Only 8 channels for noise
        print(f"   Target noise shape: {noise_shape}")
        
        noise = th.randn(*noise_shape, device=device, dtype=cond.dtype)
        print(f"✅ Noise shape: {noise.shape}")
        print(f"   Noise device: {noise.device}")
        print(f"   Noise dtype: {noise.dtype}")
        print(f"   Noise range: [{noise.min():.4f}, {noise.max():.4f}]")
        
        # Step 3: No concatenation - keep noise and conditioning separate!
        print("Step 3: Keeping noise and conditioning separate...")
        print(f"   Noise: {noise.shape}")
        print(f"   Conditioning: {cond.shape}")
        print(f"   These will be handled separately in the diffusion process")
        
        # Step 4: Run diffusion sampling with monitoring
        print("Step 4: Running diffusion sampling...")
        print(f"   Model mode: {getattr(diffusion, 'mode', 'default')}")
        print(f"   Timesteps: {diffusion.num_timesteps}")
        
        model.eval()
        
        with th.no_grad():
            final_sample = None
            step_count = 0
            
            try:
                # FIXED: Pass noise and conditioning separately
                for sample_dict in diffusion.p_sample_loop_progressive(
                    model=model,
                    shape=noise.shape,  # Only noise shape (8 channels)
                    time=diffusion.num_timesteps,
                    noise=noise,  # Only noise (8 channels)
                    cond=cond,    # Conditioning passed separately
                    clip_denoised=True,
                    model_kwargs={}
                ):
                    final_sample = sample_dict
                    step_count += 1
                    
                    # Progress monitoring (every 10 steps)
                    if step_count % 10 == 0 or step_count == 1:
                        print(f"   Sampling step {step_count}/{diffusion.num_timesteps}")
                
                print(f"✅ Sampling completed in {step_count} steps")
                
            except Exception as sampling_error:
                print(f"❌ Sampling failed at step {step_count}:")
                print(f"   Error: {str(sampling_error)}")
                raise sampling_error
            
            if final_sample is None:
                raise ValueError("Sampling returned None - no samples generated")
            
            sample = final_sample["sample"]
            print(f"✅ Raw sample shape: {sample.shape}")
        
        # Step 5: Extract wavelet components
        print("Step 5: Extracting wavelet components...")
        sample_dwt = sample  # Should already be 8 channels
        print(f"✅ DWT sample shape: {sample_dwt.shape}")
        print(f"   Sample range: [{sample_dwt.min():.4f}, {sample_dwt.max():.4f}]")
        
        # Step 6: Convert back to spatial domain
        print("Step 6: Converting to spatial domain...")
        idwt = IDWT_3D("haar")
        
        B, _, D, H, W = sample_dwt.shape
        print(f"   IDWT input dims: B={B}, D={D}, H={H}, W={W}")
        
        try:
            spatial_sample = idwt(
                sample_dwt[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                sample_dwt[:, 1, :, :, :].view(B, 1, D, H, W),
                sample_dwt[:, 2, :, :, :].view(B, 1, D, H, W),
                sample_dwt[:, 3, :, :, :].view(B, 1, D, H, W),
                sample_dwt[:, 4, :, :, :].view(B, 1, D, H, W),
                sample_dwt[:, 5, :, :, :].view(B, 1, D, H, W),
                sample_dwt[:, 6, :, :, :].view(B, 1, D, H, W),
                sample_dwt[:, 7, :, :, :].view(B, 1, D, H, W)
            )
            print(f"✅ IDWT completed. Spatial shape: {spatial_sample.shape}")
            
        except Exception as idwt_error:
            print(f"❌ IDWT failed:")
            print(f"   Error: {str(idwt_error)}")
            print(f"   Input shapes for IDWT:")
            for i in range(8):
                component = sample_dwt[:, i, :, :, :].view(B, 1, D, H, W)
                print(f"     Component {i}: {component.shape}")
            raise idwt_error
        
        # Step 7: Post-processing
        print("Step 7: Post-processing...")
        spatial_sample = th.clamp(spatial_sample, 0, 1)
        print(f"   After clamping range: [{spatial_sample.min():.4f}, {spatial_sample.max():.4f}]")
        
        # Apply brain mask from first available modality
        first_modality = list(available_modalities.values())[0].to(device)
        if first_modality.dim() == 4:
            first_modality = first_modality.unsqueeze(1)
        
        print(f"   Brain mask shape: {first_modality.shape}")
        print(f"   Brain mask device: {first_modality.device}")
        
        # Apply mask
        brain_voxels_before = (spatial_sample > 0).sum().item()
        spatial_sample[first_modality == 0] = 0
        brain_voxels_after = (spatial_sample > 0).sum().item()
        print(f"   Brain voxels: {brain_voxels_before} -> {brain_voxels_after}")
        
        # Remove batch and channel dimensions
        if spatial_sample.dim() == 5:
            spatial_sample = spatial_sample.squeeze(1)
        spatial_sample = spatial_sample[0]
        
        print(f"✅ Final output shape: {spatial_sample.shape}")
        
        # Step 8: Calculate metrics if provided
        metrics = {}
        if metrics_calculator is not None and target_data is not None:
            print("Step 8: Calculating metrics...")
            try:
                metrics = metrics_calculator.calculate_metrics(
                    spatial_sample, target_data, f"{missing_modality}_synthesis"
                )
                print(f"✅ Metrics calculated: {list(metrics.keys())}")
                if 'ssim' in metrics:
                    print(f"   SSIM: {metrics['ssim']:.4f}")
                    
            except Exception as metrics_error:
                print(f"❌ Metrics calculation failed: {str(metrics_error)}")
                # Don't fail the whole process for metrics errors
                metrics = {}
        
        print(f"✅ Synthesis completed successfully for {missing_modality}")
        return spatial_sample, metrics
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in synthesis pipeline:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        import traceback
        print("   Full traceback:")
        traceback.print_exc()
        raise e


def prepare_conditioning_debug(available_modalities, missing_modality, device):
    """Enhanced conditioning preparation with debugging"""
    dwt = DWT_3D("haar")
    
    # Get modalities in consistent order
    available_order = [m for m in MODALITIES if m != missing_modality]
    print(f"   Processing modalities in order: {available_order}")
    
    cond_list = []
    
    for i, modality in enumerate(available_order):
        print(f"   Processing modality {i+1}/3: {modality}")
        
        # Get tensor and validate
        tensor = available_modalities[modality].to(device)
        print(f"     Input shape: {tensor.shape}")
        print(f"     Input device: {tensor.device}")
        print(f"     Input range: [{tensor.min():.4f}, {tensor.max():.4f}]")
        
        if tensor.dim() == 4:
            tensor = tensor.unsqueeze(1)  # [B, 1, H, W, D]
            print(f"     After unsqueeze: {tensor.shape}")
        
        # Apply DWT with error checking
        try:
            dwt_components = dwt(tensor)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt_components
            
            print(f"     DWT components shapes:")
            print(f"       LLL: {LLL.shape}")
            print(f"       Others: {LLH.shape} (assuming all same)")
            
            # Create DWT conditioning tensor
            modality_cond = th.cat([
                LLL / 3.,  # Divide LLL by 3 as per training
                LLH, LHL, LHH, HLL, HLH, HHL, HHH
            ], dim=1)
            
            print(f"     Modality cond shape: {modality_cond.shape}")
            cond_list.append(modality_cond)
            
        except Exception as dwt_error:
            print(f"     ❌ DWT failed for {modality}: {str(dwt_error)}")
            raise dwt_error
    
    # Concatenate all modalities
    print(f"   Concatenating {len(cond_list)} modality conditions...")
    for i, cond in enumerate(cond_list):
        print(f"     Condition {i}: {cond.shape}")
    
    cond = th.cat(cond_list, dim=1)
    print(f"   Final conditioning shape: {cond.shape}")
    
    # Validate expected shape
    expected_channels = len(available_order) * 8  # Should be 24 for 3 modalities
    if cond.shape[1] != expected_channels:
        raise ValueError(f"Expected {expected_channels} conditioning channels, got {cond.shape[1]}")
    
    return cond

def apply_uncrop_to_original(cropped_output):
    """Uncrop from (160,224,160) back to (240,240,160)"""
    if isinstance(cropped_output, th.Tensor):
        uncropped = th.zeros(ORIGINAL_SHAPE, dtype=cropped_output.dtype, device=cropped_output.device)
    else:
        uncropped = np.zeros(ORIGINAL_SHAPE, dtype=cropped_output.dtype)
    
    # Place cropped output back in original position
    uncropped[
        CROP_BOUNDS['x_min']:CROP_BOUNDS['x_max'],
        CROP_BOUNDS['y_min']:CROP_BOUNDS['y_max'],
        CROP_BOUNDS['z_min']:CROP_BOUNDS['z_max']
    ] = cropped_output
    
    return uncropped