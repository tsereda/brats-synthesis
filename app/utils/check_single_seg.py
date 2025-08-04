#!/usr/bin/env python3

import sys
import numpy as np
import nibabel as nib
import os

def analyze_segmentation_file(seg_file_path):
    """
    Analyze a single segmentation file to check its label format
    """
    
    if not os.path.exists(seg_file_path):
        print(f"File not found: {seg_file_path}")
        return False
    
    if not seg_file_path.endswith('.nii.gz'):
        print(f"File must be a .nii.gz file: {seg_file_path}")
        return False
    
    print(f"Analyzing segmentation file: {os.path.basename(seg_file_path)}")
    print(f"Full path: {seg_file_path}")
    print("=" * 60)
    
    try:
        # Load the segmentation
        img = nib.load(seg_file_path)
        data = img.get_fdata()
        
        # Basic info
        unique_labels = np.unique(data).astype(int)
        non_zero_count = np.sum(data > 0)
        total_voxels = data.size
        tumor_percentage = (non_zero_count / total_voxels) * 100
        
        print(f"Basic Information:")
        print(f"   - Shape: {data.shape}")
        print(f"   - Data type: {data.dtype}")
        print(f"   - Value range: [{data.min():.1f}, {data.max():.1f}]")
        print(f"   - File size: {os.path.getsize(seg_file_path) / 1024:.1f} KB")
        
        print(f"\nLabel Analysis:")
        print(f"   - Unique labels: {sorted(unique_labels)}")
        print(f"   - Number of label classes: {len(unique_labels)}")
        print(f"   - Background voxels (label 0): {np.sum(data == 0):,}")
        print(f"   - Tumor voxels (non-zero): {non_zero_count:,}")
        print(f"   - Tumor percentage: {tumor_percentage:.2f}%")
        
        # Detailed breakdown by label
        print(f"\nLabel Distribution:")
        for label in sorted(unique_labels):
            count = np.sum(data == label)
            percentage = (count / total_voxels) * 100
            if label == 0:
                print(f"   - Label {int(label)} (Background): {count:,} voxels ({percentage:.2f}%)")
            else:
                print(f"   - Label {int(label)} (Tumor class): {count:,} voxels ({percentage:.2f}%)")
        
        # Compare with common formats
        print(f"\n🔍 Format Comparison:")
        print(f"This file uses: {sorted(unique_labels)}")
        
        # Determine if conversion is needed
        your_labels = [0, 1, 2, 3]
        brats_standard = [0, 1, 2, 4]
        file_labels = sorted(unique_labels)
        
        if file_labels == your_labels:
            print(f" Your outputs match this training example.")
        elif file_labels == brats_standard:
            print(f" Training uses BraTS standard [0,1,2,4]")
        else:
            print(f" Unexpected label pattern: {file_labels}")
        
        return True
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_single_seg.py <segmentation_file.nii.gz>")
        print("\nExample:")
        print("  python check_single_seg.py ./training_data/BraTS-GLI-00001-000-seg.nii.gz")
        print("  python check_single_seg.py ./outputs/BraTS-GLI-00001-000.nii.gz")
        sys.exit(1)
    
    seg_file = sys.argv[1]
    print("BraSyn Challenge - Single Segmentation Analysis")
    print("=" * 60)
    
    success = analyze_segmentation_file(seg_file)
    
    if success:
        print(f"\nAnalysis complete!")
    else:
        print(f"\nAnalysis failed!")