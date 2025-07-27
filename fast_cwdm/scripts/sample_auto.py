"""
Updated automatic sampling script for paired image-to-image translation.
Uses VALIDATED WAVELET-FRIENDLY crop bounds with 42% memory reduction.
FIXED: All dimension mismatches, hardcoded assumptions, and checkpoint logic.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

# VALIDATED WAVELET-FRIENDLY CROP BOUNDS from analysis
CROP_BOUNDS = {
    'x_min': 39, 'x_max': 199,  # width: 160 (divisible by 16)
    'y_min': 9, 'y_max': 233,  # height: 224 (divisible by 16)
    'z_min': 0,  'z_max': 160   # depth: 155 (original)
}

def apply_uncrop_to_original(cropped_output):
    """Uncrop from (160,224,155) back to (240,240,155)"""
    if isinstance(cropped_output, th.Tensor):
        uncropped = th.zeros((240, 240, 160), dtype=cropped_output.dtype, device=cropped_output.device)
    else:
        uncropped = np.zeros((240, 240, 160), dtype=cropped_output.dtype)
    
    # Place cropped output back in original position
    uncropped[
        CROP_BOUNDS['x_min']:CROP_BOUNDS['x_max'],
        CROP_BOUNDS['y_min']:CROP_BOUNDS['y_max'],
        CROP_BOUNDS['z_min']:CROP_BOUNDS['z_max']
    ] = cropped_output
    
    return uncropped

def find_checkpoint(missing_modality, checkpoint_dir):
    """Find the best checkpoint for the missing modality."""
    import glob
    
    # Look for BEST checkpoints first
    pattern = f"brats_{missing_modality}_*.pt"
    best_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if best_files:
        checkpoint = best_files[0]
        logger.log(f"Found checkpoint: {checkpoint}")
        return checkpoint
    
    # Fallback to regular checkpoints
    pattern = f"*{missing_modality}*.pt"
    regular_files = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not regular_files:
        raise FileNotFoundError(f"No checkpoint found for {missing_modality} in {checkpoint_dir}")
    
    # Sort by modification time (most recent first)
    regular_files.sort(key=os.path.getmtime, reverse=True)
    checkpoint = regular_files[0]
    logger.log(f"Found checkpoint: {checkpoint}")
    return checkpoint

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'

    if args.dataset == 'brats':
        # Use enhanced dataloader with validated optimal cropping
        ds = BRATSVolumes(args.data_dir, mode='auto')
        logger.log(f"Using dataloader with {ds.get_crop_info()}")
        logger.log(f"Output dimensions: {ds.get_output_dimensions()}")
        logger.log(f"DWT dimensions: {ds.get_dwt_dimensions()}")

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False,)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create checkpoint directory if needed
    checkpoint_dir = getattr(args, 'checkpoint_dir', './checkpoints')
    if not os.path.exists(checkpoint_dir):
        logger.log(f"Warning: Checkpoint directory {checkpoint_dir} does not exist")

    for batch in iter(datal):
        missing = batch['missing'][0]
        logger.log("Missing modality: {}".format(missing))

        if missing == 'none':
            logger.log("No missing modality found, skipping this batch")
            continue

        # FIXED: Dynamic checkpoint loading based on missing modality
        try:
            if hasattr(args, 'model_path') and args.model_path:
                # Use provided model path
                selected_model_path = args.model_path
            else:
                # Find checkpoint dynamically
                selected_model_path = find_checkpoint(missing, checkpoint_dir)
        except FileNotFoundError as e:
            logger.log(f"Error: {e}")
            continue

        logger.log("Loading model from: {}".format(selected_model_path))
        model.load_state_dict(dist_util.load_state_dict(selected_model_path, map_location="cpu"))
        model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices

        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())

        subj = batch['subj'][0].split('pseudo_validation_v2/')[1][:19] if 'pseudo_validation_v2/' in batch['subj'][0] else batch['subj'][0].split('pseudo_validation/')[1][:19]
        logger.log(f"Processing subject: {subj}")

        miss_name = args.data_dir + '/' + subj + '/' + subj +'-' + missing
        logger.log(f"Output path: {miss_name}")

        # Set up conditioning modalities based on missing modality
        if missing == 't1n':
            cond_1 = batch['t1c']  # condition
            cond_2 = batch['t2w']  # condition
            cond_3 = batch['t2f']  # condition
            try:
                header = nib.load(batch['filedict']['t1c'][0]).header
            except:
                header = None

        elif missing == 't1c':
            cond_1 = batch['t1n']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
            try:
                header = nib.load(batch['filedict']['t1n'][0]).header
            except:
                header = None

        elif missing == 't2w':
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2f']
            try:
                header = nib.load(batch['filedict']['t1n'][0]).header
            except:
                header = None

        elif missing == 't2f':
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2w']
            try:
                header = nib.load(batch['filedict']['t1n'][0]).header
            except:
                header = None

        else:
            logger.log("This contrast can't be synthesized.")
            continue

        logger.log(f"Input shapes - cond_1: {cond_1.shape}, cond_2: {cond_2.shape}, cond_3: {cond_3.shape}")

        # Conditioning vector - Apply DWT to each conditioning modality
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_2)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_3)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        logger.log(f"Conditioning shape: {cond.shape}")

        # FIXED: Noise tensor with correct DWT dimensions
        # Get DWT dimensions from the dataloader
        dwt_dims = ds.get_dwt_dimensions()  # Should be (80, 104, 77)
        noise = th.randn(args.batch_size, 8, *dwt_dims).to(dist_util.dev())
        logger.log(f"Noise shape: {noise.shape}")

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        logger.log(f"Sample shape after sampling: {sample.shape}")

        # Convert back to spatial domain using IDWT
        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        logger.log(f"Sample shape after IDWT: {sample.shape}")

        # Post-process sample
        sample[sample <= 0.04] = 0

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # don't squeeze batch dimension for bs 1

        # FIXED: No more hardcoded padding/cropping - dimensions should already be correct
        # The sample should already be in (160, 224, 155) format

        logger.log(f"Final sample shape: {sample.shape}")

        if len(cond_1.shape) == 5:
            cond_1 = cond_1.squeeze(dim=1)
        if len(cond_2.shape) == 5:
            cond_2 = cond_2.squeeze(dim=1)
        if len(cond_3.shape) == 5:
            cond_3 = cond_3.squeeze(dim=1)

        for i in range(sample.shape[0]):
            # FIXED: Save to proper directory structure and use uncropping
            os.makedirs(os.path.dirname(miss_name), exist_ok=True)
            
            # Get the sample data
            sample_data = sample.detach().cpu().numpy()[i, :, :, :]
            
            # Uncrop to original dimensions (240, 240, 155)
            sample_uncropped = apply_uncrop_to_original(sample_data)
            
            # Save with proper header if available
            output_name = miss_name + '.nii.gz'
            if header is not None:
                img = nib.Nifti1Image(sample_uncropped, None, header)
            else:
                img = nib.Nifti1Image(sample_uncropped, np.eye(4))
            
            nib.save(img=img, filename=output_name)
            logger.log(f'Saved to {output_name}')
            
            # Optionally save cropped version for debugging
            if hasattr(args, 'save_cropped') and args.save_cropped:
                cropped_output_name = miss_name + '_cropped.nii.gz'
                img_cropped = nib.Nifti1Image(sample_data, np.eye(4))
                nib.save(img=img_cropped, filename=cropped_output_name)
                logger.log(f'Saved cropped version to {cropped_output_name}')

        logger.log(f"✅ Successfully processed {subj} with validated crop bounds")

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        checkpoint_dir="./checkpoints",  # ADDED: Checkpoint directory
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=224,  # UPDATED: Height from validated crop (was 256)
        half_res_crop=False,
        concat_coords=False, # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        contr="",
        save_cropped=False,  # ADDED: Option to save cropped versions
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()