#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import random
import argparse

def load_nifti(path):
    """Load and return nifti data and the nifti object"""
    nii = nib.load(str(path))
    return nii.get_fdata(), nii

def fix_floating_point_labels(segmentation):
    """Fix floating point precision issues by rounding to nearest integers"""
    # Round to nearest integer and clip to valid range [0, 4]
    fixed_seg = np.round(segmentation).astype(np.int16)
    fixed_seg = np.clip(fixed_seg, 0, 4)
    return fixed_seg

def swap_labels_1_2(segmentation):
    """Swap labels 1 and 2 in segmentation (with floating point fix)"""
    # First fix any floating point precision issues
    seg_fixed = fix_floating_point_labels(segmentation)
    
    # Now create copy for swapping
    seg_copy = seg_fixed.copy()
    
    # Create masks for each label (now using integer comparison)
    mask1 = seg_fixed == 1
    mask2 = seg_fixed == 2
    
    # Swap the labels
    seg_copy[mask1] = 2
    seg_copy[mask2] = 1
    
    return seg_copy

def find_best_slice(segmentation):
    """Find slice with most non-background content"""
    # Fix floating point issues first for accurate counting
    seg_fixed = fix_floating_point_labels(segmentation)
    
    slice_scores = []
    for i in range(seg_fixed.shape[2]):
        slice_data = seg_fixed[:, :, i]
        # Count non-zero pixels
        non_zero = np.sum(slice_data > 0)
        slice_scores.append(non_zero)
    
    # Return slice with most content
    return np.argmax(slice_scores)

def create_comparison_grid(file_examples, input_dir_name):
    """Create a 2x10 grid showing original vs swapped for 10 random files"""
    
    fig, axes = plt.subplots(2, 10, figsize=(25, 6))
    
    for col, (filename, original_seg, swapped_seg) in enumerate(file_examples):
        # Find best slice for visualization
        best_slice = find_best_slice(original_seg)
        
        # Fix floating point issues for visualization
        orig_fixed = fix_floating_point_labels(original_seg)
        
        # Original (top row)
        orig_slice = orig_fixed[:, :, best_slice]
        axes[0, col].imshow(orig_slice, cmap='jet', vmin=0, vmax=4)
        axes[0, col].set_title(f'{filename}\n(Original)', fontsize=8)
        axes[0, col].axis('off')
        
        # Swapped (bottom row)
        swap_slice = swapped_seg[:, :, best_slice]
        axes[1, col].imshow(swap_slice, cmap='jet', vmin=0, vmax=4)
        axes[1, col].set_title('After 1↔2 Swap', fontsize=8)
        axes[1, col].axis('off')
    
    plt.suptitle(f'Random Sample from "{input_dir_name}": Label 1↔2 Swap Preview (10 Examples)', fontsize=16)
    plt.tight_layout()
    
    # Save the grid
    output_file = "label_swap_preview_grid.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"📸 Preview grid saved as: {output_file}")
    
    return fig

def analyze_file_before_after(seg_data, swapped_seg, filename):
    """Analyze label distribution before and after swapping"""
    # Fix original data first
    orig_fixed = fix_floating_point_labels(seg_data)
    
    print(f"\n📊 Analysis for {filename}:")
    print("  Original distribution:")
    for label in [0, 1, 2, 3]:
        count = np.sum(orig_fixed == label)
        if count > 0:
            pct = 100 * count / orig_fixed.size
            print(f"    Label {label}: {count:,} voxels ({pct:.2f}%)")
    
    print("  After 1↔2 swap:")
    for label in [0, 1, 2, 3]:
        count = np.sum(swapped_seg == label)
        if count > 0:
            pct = 100 * count / swapped_seg.size
            print(f"    Label {label}: {count:,} voxels ({pct:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Preview and apply a 1↔2 label swap on NIfTI segmentation files (with floating point fix).")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing .nii.gz files.")
    parser.add_argument("--show-analysis", action="store_true", help="Show detailed before/after analysis for each file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    print("🎲 Label 1↔2 Swap Preview Tool (Floating Point Fixed)")
    print("=" * 60)
    
    # Find all segmentation files
    pred_files = glob.glob(f"{input_dir}/*.nii.gz")
    
    if not pred_files:
        print(f"❌ No .nii.gz files found in {input_dir}/")
        return
    
    print(f"Found {len(pred_files)} total files")
    
    # Randomly select 10 files
    if len(pred_files) < 10:
        selected_files = pred_files
        print(f"Using all {len(pred_files)} available files")
    else:
        selected_files = random.sample(pred_files, 10)
        print(f"Randomly selected 10 files for preview")
    
    print("\n🔄 Processing selected files...")
    
    file_examples = []
    
    for i, seg_file in enumerate(selected_files):
        filename = Path(seg_file).stem  # Remove .nii.gz extension
        print(f"  {i+1}/{len(selected_files)}: {filename}")
        
        try:
            # Load segmentation
            seg_data, _ = load_nifti(seg_file)
            
            # Apply 1↔2 swap (now with floating point fix)
            swapped_seg = swap_labels_1_2(seg_data)
            
            # Show analysis if requested
            if args.show_analysis:
                analyze_file_before_after(seg_data, swapped_seg, filename)
            
            # Store for visualization
            file_examples.append((filename, seg_data, swapped_seg))
            
        except Exception as e:
            print(f"    ❌ Error loading {filename}: {e}")
            continue
    
    if not file_examples:
        print("❌ No files could be processed!")
        return
    
    print(f"\n📊 Creating comparison grid with {len(file_examples)} examples...")
    
    # Create the comparison visualization
    create_comparison_grid(file_examples, input_dir.name)
    
    print("\n" + "=" * 60)
    print("🎉 PREVIEW COMPLETE!")
    print("📸 Check 'label_swap_preview_grid.png' to see the results")
    print("👀 Compare the top row (original) vs bottom row (swapped)")
    print("🔍 Look for:")
    print("   - Are labels 1 & 2 (different colors) swapping positions?")
    print("   - Does the swapped version look more reasonable?")
    print("   - Are labels 3 (core regions) staying the same?")
    print("💡 NOTE: Floating point precision issues have been automatically fixed!")
    
    response = input(f"\nIf the preview looks good, run the full batch on all {len(pred_files)} files? (y/n): ")
    
    if response.lower() == 'y':
        print(f"\n🚀 Processing all {len(pred_files)} files...")
        
        # Create output directory
        output_dir = Path("fixed_nii_gz_files")
        output_dir.mkdir(exist_ok=True)
        
        successful = 0
        total_files_analyzed = 0
        
        for i, seg_file in enumerate(pred_files):
            filename = Path(seg_file).name
            print(f"Processing {i+1}/{len(pred_files)}: {filename}", end=" ... ")
            
            try:
                # Load, swap, save
                seg_data, nii_obj = load_nifti(seg_file)
                fixed_seg = swap_labels_1_2(seg_data)  # This now includes floating point fix
                
                output_path = output_dir / filename
                
                # Save as int16 to ensure clean integer labels
                fixed_nii = nib.Nifti1Image(fixed_seg.astype(np.int16), 
                                            nii_obj.affine, nii_obj.header)
                nib.save(fixed_nii, output_path)
                
                # Quick verification
                unique_labels = np.unique(fixed_seg)
                print(f"✅ (labels: {unique_labels})")
                successful += 1
                total_files_analyzed += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
                total_files_analyzed += 1
        
        print(f"\n🎉 Batch processing complete! {successful}/{total_files_analyzed} files processed")
        print(f"📁 Fixed files saved to: {output_dir}/")
        print("💡 All files now have clean integer labels AND 1↔2 swap applied!")
        print(f"🔍 Test with: python app/utils/check_single_seg.py {output_dir}/<filename>")
    else:
        print("👋 Preview only - no batch processing performed")

if __name__ == "__main__":
    main()