#!/usr/bin/env python3
import os
import numpy as np
import nibabel as nib
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
import glob

def load_nifti(path):
    """Load and return nifti data"""
    nii = nib.load(str(path))
    return nii.get_fdata()

def compute_segmentation_metrics(pred_seg, gt_seg):
    """Compute segmentation metrics"""
    # Convert to tensor format for MONAI metrics
    pred_tensor = torch.from_numpy(pred_seg).unsqueeze(0).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_seg).unsqueeze(0).unsqueeze(0)
    
    # Convert to one-hot for multi-class metrics
    pred_onehot = torch.zeros(1, 4, *pred_seg.shape)  # 4 classes (0,1,2,3)
    gt_onehot = torch.zeros(1, 4, *gt_seg.shape)
    
    for i in range(4):
        pred_onehot[0, i] = (pred_tensor[0, 0] == i).float()
        gt_onehot[0, i] = (gt_tensor[0, 0] == i).float()
        
    # Compute metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_scores = dice_metric(pred_onehot, gt_onehot)
    
    # Individual class dice scores
    dice_et = float(dice_scores[0, 2]) if len(dice_scores[0]) > 2 else 0.0  # Enhancing tumor  
    dice_tc = float(dice_scores[0, 1]) if len(dice_scores[0]) > 1 else 0.0  # Tumor core
    dice_wt = float(dice_scores[0, 3]) if len(dice_scores[0]) > 3 else 0.0  # Whole tumor
    
    return {
        "dice_et": dice_et,
        "dice_tc": dice_tc, 
        "dice_wt": dice_wt,
        "dice_mean": float(np.mean([dice_et, dice_tc, dice_wt]))
    }

def create_segmentation_overlay(image, pred_seg, gt_seg, case_id):
    """Create segmentation comparison visualization"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    slice_idx = image.shape[2] // 2
    img_slice = image[:, :, slice_idx]
    pred_slice = pred_seg[:, :, slice_idx]
    gt_slice = gt_seg[:, :, slice_idx]
    
    # Original image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Predicted segmentation
    axes[1].imshow(pred_slice, cmap='jet', vmin=0, vmax=3)
    axes[1].set_title('Predicted Seg')
    axes[1].axis('off')
    
    # Ground truth segmentation
    axes[2].imshow(gt_slice, cmap='jet', vmin=0, vmax=3)
    axes[2].set_title('Ground Truth Seg')
    axes[2].axis('off')
    
    # Overlay comparison
    axes[3].imshow(img_slice, cmap='gray', alpha=0.7)
    axes[3].imshow(pred_slice, cmap='Reds', alpha=0.3, vmin=0, vmax=3)
    axes[3].imshow(gt_slice, cmap='Blues', alpha=0.3, vmin=0, vmax=3)
    axes[3].set_title('Overlay (Red=Pred, Blue=GT)')
    axes[3].axis('off')
    
    plt.suptitle(f'Segmentation Comparison - {case_id}')
    plt.tight_layout()
    return fig

def main():
    # Initialize wandb (continue the existing run)
    wandb.init(project="fast-cwmd-eval", entity="timgsereda", job_type="seg_evaluation")
    
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Find prediction files - try multiple possible locations
    possible_pred_paths = [
        "./outputs/*.nii.gz",
        "../outputs/*.nii.gz", 
        "../brats-synthesis/outputs/*.nii.gz"
    ]
    
    pred_files = []
    for pattern in possible_pred_paths:
        pred_files = glob.glob(pattern)
        if pred_files:
            print(f"Found {len(pred_files)} prediction files at {pattern}")
            break
    
    if not pred_files:
        print("No prediction files found!")
        return

    # Original training data directory - try multiple possible locations
    possible_original_dirs = [
        Path("ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData"),
        Path("../BraSyn_tutorial/ASNR-MICCAI-BraTS2023-GLI-MET-TrainingData"),
        Path("BraTS2023-TrainingData"),
        Path("../BraSyn_tutorial/BraTS2023-TrainingData")
    ]
    
    original_dir = None
    for possible_dir in possible_original_dirs:
        if possible_dir.exists():
            original_dir = possible_dir
            break
            
    if original_dir is None:
        print("Error: Could not find original training data directory")
        print("Current working directory:", os.getcwd())
        print("Available directories:")
        for item in Path(".").iterdir():
            if item.is_dir():
                print(f"  {item}")
        return
    
    print(f"Using original directory: {original_dir}")
    
    # Debug: Check what's actually in the original directory
    print(f"Original directory contents (first 10 items):")
    for item in sorted(original_dir.iterdir())[:10]:
        print(f"  {item.name}")

    all_dice_scores = {"dice_et": [], "dice_tc": [], "dice_wt": [], "dice_mean": []}

    for pred_file in pred_files[:250]:  # Limit for testing
        # Extract case name properly by removing both .nii.gz extensions
        case_name = Path(pred_file).name.replace('.nii.gz', '').replace('.nii', '')
        print(f"Evaluating {case_name}")
        
        # Load predicted segmentation
        pred_seg = load_nifti(pred_file)
        
        # Find corresponding ground truth segmentation
        # FIXED: Look in the case subdirectory
        case_dir = original_dir / case_name
        if not case_dir.exists():
            print(f"Warning: Case directory not found: {case_dir}")
            continue
            
        possible_gt_files = [
            case_dir / f"{case_name}-seg.nii.gz",
            case_dir / f"{case_name}_seg.nii.gz",
        ]
        
        gt_seg_file = None
        for possible_file in possible_gt_files:
            if possible_file.exists():
                gt_seg_file = possible_file
                break
                
        if gt_seg_file is None:
            print(f"Warning: GT segmentation not found for {case_name}")
            print(f"  Looked in: {case_dir}")
            print(f"  Available files:")
            for file in case_dir.glob("*.nii.gz"):
                print(f"    {file.name}")
            continue
            
        gt_seg = load_nifti(gt_seg_file)
        
        # Load one of the original modalities for visualization
        # FIXED: Look in the case subdirectory
        possible_t1_files = [
            case_dir / f"{case_name}-t1n.nii.gz",
            case_dir / f"{case_name}_t1n.nii.gz",
            case_dir / f"{case_name}-t1.nii.gz",
            case_dir / f"{case_name}_t1.nii.gz",
        ]
        
        t1_img = None
        for possible_file in possible_t1_files:
            if possible_file.exists():
                t1_img = load_nifti(possible_file)
                break
        
        # Compute metrics
        metrics = compute_segmentation_metrics(pred_seg, gt_seg)
        
        # Add to aggregated results
        for key, value in metrics.items():
            if key in all_dice_scores:
                all_dice_scores[key].append(value)
        
        # Create visualization
        if t1_img is not None:
            fig = create_segmentation_overlay(t1_img, pred_seg, gt_seg, case_name)
            
            # Log to wandb
            wandb.log({
                f"seg_eval/{case_name}/comparison": wandb.Image(fig),
                f"seg_eval/{case_name}/dice_et": metrics["dice_et"],
                f"seg_eval/{case_name}/dice_tc": metrics["dice_tc"],
                f"seg_eval/{case_name}/dice_wt": metrics["dice_wt"],
                f"seg_eval/{case_name}/dice_mean": metrics["dice_mean"]
            })
            
            plt.close(fig)
        
        print(f"  Dice ET: {metrics['dice_et']:.3f}, TC: {metrics['dice_tc']:.3f}, WT: {metrics['dice_wt']:.3f}")

    # Compute and log summary statistics
    summary_stats = {}
    for metric, values in all_dice_scores.items():
        if values:
            summary_stats[f"seg_eval/summary/{metric}_mean"] = np.mean(values)
            summary_stats[f"seg_eval/summary/{metric}_std"] = np.std(values)

    wandb.log(summary_stats)

    print("=== SEGMENTATION EVALUATION SUMMARY ===")
    for metric, values in all_dice_scores.items():
        if values:
            print(f"{metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    wandb.finish()

if __name__ == "__main__":
    main()