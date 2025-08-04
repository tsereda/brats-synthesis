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

def swap_labels_1_2(segmentation):
    """Swap labels 1 and 2 in segmentation"""
    seg_copy = segmentation.copy()
    
    # Create masks for each label
    mask1 = segmentation == 1
    mask2 = segmentation == 2
    
    # Swap the labels
    seg_copy[mask1] = 2
    seg_copy[mask2] = 1
    
    return seg_copy

def find_best_slice(segmentation):
    """Find slice with most non-background content"""
    slice_scores = []
    for i in range(segmentation.shape[2]):
        slice_data = segmentation[:, :, i]
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
        
        # Original (top row)
        orig_slice = original_seg[:, :, best_slice]
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

def main():
    parser = argparse.ArgumentParser(description="Preview and apply a 1↔2 label swap on NIfTI segmentation files.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing .nii.gz files.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    print("🎲 Label 1↔2 Swap Preview Tool")
    print("=" * 50)
    
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
            
            # Apply 1↔2 swap
            swapped_seg = swap_labels_1_2(seg_data)
            
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
    print("   - Are labels 3 & 4 (core regions) staying the same?")
    
    response = input(f"\nIf the preview looks good, run the full batch on all {len(pred_files)} files? (y/n): ")
    
    if response.lower() == 'y':
        print(f"\n🚀 Processing all {len(pred_files)} files...")
        
        # Create output directory
        output_dir = Path("fixed_nii_gz_files")
        output_dir.mkdir(exist_ok=True)
        
        successful = 0
        for i, seg_file in enumerate(pred_files):
            filename = Path(seg_file).name
            print(f"Processing {i+1}/{len(pred_files)}: {filename}", end=" ... ")
            
            try:
                # Load, swap, save
                seg_data, nii_obj = load_nifti(seg_file)
                fixed_seg = swap_labels_1_2(seg_data)
                
                output_path = output_dir / filename
                fixed_nii = nib.Nifti1Image(fixed_seg.astype(seg_data.dtype), 
                                            nii_obj.affine, nii_obj.header)
                nib.save(fixed_nii, output_path)
                
                print("✅")
                successful += 1
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🎉 Batch processing complete! {successful}/{len(pred_files)} files processed")
        print(f"📁 Fixed files saved to: {output_dir}/")
    else:
        print("👋 Preview only - no batch processing performed")

if __name__ == "__main__":
    main()