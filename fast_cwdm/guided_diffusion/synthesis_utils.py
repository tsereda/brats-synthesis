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
    FIXED: Proper channel concatenation for validation synthesis
    """
    print(f"\n=== Synthesizing {missing_modality} (FIXED PIPELINE) ===")
    
    # Prepare conditioning - this should give us 24 channels (3 modalities × 8 each)
    cond = prepare_conditioning(available_modalities, missing_modality, device)
    print(f"Conditioning shape: {cond.shape}")  # Should be [1, 24, D, H, W]
    
    # Create noise tensor - this should be 8 channels
    _, _, cond_d, cond_h, cond_w = cond.shape
    noise_shape = (1, 8, cond_d, cond_h, cond_w)
    noise = th.randn(*noise_shape, device=device)
    print(f"Noise shape: {noise.shape}")  # Should be [1, 8, D, H, W]
    
    # CRITICAL FIX: Manually concatenate conditioning and noise to get 32 channels
    # This matches exactly what happens during training in p_mean_variance()
    combined_input = th.cat([noise, cond], dim=1)  # [1, 32, D, H, W]
    print(f"Combined input shape: {combined_input.shape}")  # Should be [1, 32, D, H, W]
    
    # Use the combined input as the "noise" parameter
    # The sampling function will treat this as the starting point
    with th.no_grad():
        final_sample = None
        for sample_dict in diffusion.p_sample_loop_progressive(
            model=model,
            shape=combined_input.shape,  # Use the 32-channel shape
            time=diffusion.num_timesteps,
            noise=combined_input,  # Pass the concatenated tensor
            clip_denoised=True,
            model_kwargs={}
            # Note: No separate 'cond' parameter - it's already in the noise tensor
        ):
            final_sample = sample_dict
        
        sample = final_sample["sample"]
    
    # Extract only the first 8 channels (the synthesized DWT components)
    # The conditioning channels will also be in the output, but we only want the first 8
    sample = sample[:, :8, :, :, :]  # Take only first 8 channels
    print(f"Sample shape after extraction: {sample.shape}")
    
    # Convert back to spatial domain using IDWT
    idwt = IDWT_3D("haar")
    B, _, D, H, W = sample.shape
    
    spatial_sample = idwt(
        sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
        sample[:, 1, :, :, :].view(B, 1, D, H, W),
        sample[:, 2, :, :, :].view(B, 1, D, H, W),
        sample[:, 3, :, :, :].view(B, 1, D, H, W),
        sample[:, 4, :, :, :].view(B, 1, D, H, W),
        sample[:, 5, :, :, :].view(B, 1, D, H, W),
        sample[:, 6, :, :, :].view(B, 1, D, H, W),
        sample[:, 7, :, :, :].view(B, 1, D, H, W)
    )
    
    # Post-process
    spatial_sample = th.clamp(spatial_sample, 0, 1)
    
    # Apply brain mask from first available modality
    first_modality = list(available_modalities.values())[0].to(device)
    if first_modality.dim() == 4:
        first_modality = first_modality.unsqueeze(1)
    
    spatial_sample[first_modality == 0] = 0
    
    # Remove batch and channel dimensions
    if spatial_sample.dim() == 5:
        spatial_sample = spatial_sample.squeeze(1)
    spatial_sample = spatial_sample[0]
    
    # Calculate metrics if provided
    metrics = {}
    if metrics_calculator is not None and target_data is not None:
        metrics = metrics_calculator.calculate_metrics(
            spatial_sample, target_data, f"{missing_modality}_synthesis"
        )
    
    return spatial_sample, metrics

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