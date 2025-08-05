#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import sys

def diagnose_segmentation(file_path):
    """Diagnose segmentation file for floating point precision issues"""
    print(f"🔍 Diagnosing: {file_path}")
    print("=" * 60)
    
    # Load the file
    nii = nib.load(file_path)
    data = nii.get_fdata()
    
    # Basic info
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{data.min():.6f}, {data.max():.6f}]")
    
    # Find all unique values
    unique_vals = np.unique(data)
    print(f"\nUnique values found: {len(unique_vals)}")
    
    # Show first 20 unique values with more precision
    print("First 20 unique values (high precision):")
    for i, val in enumerate(unique_vals[:20]):
        count = np.sum(data == val)
        print(f"  {val:.10f} -> {count:,} voxels")
    
    # Check for values close to integers
    print("\n🎯 Values close to expected labels:")
    for target in [0, 1, 2, 3, 4]:
        close_vals = unique_vals[np.abs(unique_vals - target) < 0.01]
        if len(close_vals) > 0:
            for val in close_vals:
                count = np.sum(data == val)
                diff = val - target
                print(f"  {val:.10f} (≈{target}, diff: {diff:+.2e}) -> {count:,} voxels")
    
    # Check for non-integer values
    non_integers = unique_vals[np.abs(unique_vals - np.round(unique_vals)) > 1e-10]
    if len(non_integers) > 0:
        print(f"\n⚠️  Found {len(non_integers)} non-integer values:")
        for val in non_integers[:10]:  # Show first 10
            count = np.sum(data == val)
            print(f"  {val:.10f} -> {count:,} voxels")
        if len(non_integers) > 10:
            print(f"  ... and {len(non_integers) - 10} more")
    
    # Suggest fix
    print("\n💡 Suggested fix:")
    rounded_data = np.round(data).astype(np.int16)
    rounded_unique = np.unique(rounded_data)
    print(f"After rounding: {rounded_unique}")
    
    # Check if rounding fixes the issue
    tumor_voxels_original = np.sum(data > 0)
    tumor_voxels_rounded = np.sum(rounded_data > 0)
    print(f"Tumor voxels before rounding: {tumor_voxels_original:,}")
    print(f"Tumor voxels after rounding: {tumor_voxels_rounded:,}")
    
    # Show distribution after rounding
    print("\nLabel distribution after rounding:")
    for label in [0, 1, 2, 3, 4]:
        count = np.sum(rounded_data == label)
        if count > 0:
            pct = 100 * count / rounded_data.size
            print(f"  Label {label}: {count:,} voxels ({pct:.2f}%)")

def fix_segmentation(input_path, output_path=None):
    """Fix segmentation by rounding to nearest integers"""
    if output_path is None:
        output_path = input_path.replace('.nii.gz', '_fixed.nii.gz')
    
    print(f"🔧 Fixing: {input_path}")
    
    # Load
    nii = nib.load(input_path)
    data = nii.get_fdata()
    
    # Round and convert to appropriate integer type
    fixed_data = np.round(data).astype(np.int16)
    
    # Create new NIfTI
    fixed_nii = nib.Nifti1Image(fixed_data, nii.affine, nii.header)
    
    # Save
    nib.save(fixed_nii, output_path)
    print(f"✅ Fixed file saved as: {output_path}")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose.py <segmentation_file.nii.gz> [--fix]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Diagnose
    diagnose_segmentation(file_path)
    
    # Fix if requested
    if "--fix" in sys.argv:
        print("\n" + "="*60)
        fix_segmentation(file_path)