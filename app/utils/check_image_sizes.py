#!/usr/bin/env python3

import os
import sys
import nibabel as nib
import numpy as np
from pathlib import Path

def check_image_dimensions(file_path):
    """Check dimensions of a single NIfTI file"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data.shape, img.header.get_zooms()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None, None

def analyze_case_directory(case_dir):
    """Analyze all modalities in a case directory"""
    print(f"\n=== Analyzing case: {os.path.basename(case_dir)} ===")
    
    modalities = ['t1n', 't1c', 't2w', 't2f']
    case_name = os.path.basename(case_dir)
    
    shapes = {}
    spacings = {}
    
    for mod in modalities:
        file_path = os.path.join(case_dir, f"{case_name}-{mod}.nii.gz")
        
        if os.path.exists(file_path):
            shape, spacing = check_image_dimensions(file_path)
            if shape is not None:
                shapes[mod] = shape
                spacings[mod] = spacing
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✅ {mod}: {shape} | spacing: {[f'{s:.2f}' for s in spacing]} | {size_mb:.1f}MB")
            else:
                shapes[mod] = None
        else:
            print(f"  ❌ Missing: {mod}")
            shapes[mod] = None
    
    # Check consistency
    valid_shapes = [s for s in shapes.values() if s is not None]
    if valid_shapes:
        first_shape = valid_shapes[0]
        all_same = all(s == first_shape for s in valid_shapes)
        
        if all_same:
            print(f"  ✅ All modalities have consistent shape: {first_shape}")
        else:
            print(f"  ❌ Inconsistent shapes across modalities!")
            for mod, shape in shapes.items():
                if shape:
                    print(f"      {mod}: {shape}")
    
    return shapes

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_image_sizes.py <directory_or_case_path>")
        print("\nExamples:")
        print("  # Check all cases in synthesis output directory")
        print("  python check_image_sizes.py ./datasets/BRATS2023/pseudo_validation_completed")
        print("\n  # Check a single case")
        print("  python check_image_sizes.py ./datasets/BRATS2023/pseudo_validation_completed/BraTS-MET-00001-000")
        print("\n  # Check nnUNet dataset")
        print("  python check_image_sizes.py ./Dataset137_BraTS2021_inference/imagesTs")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not os.path.exists(input_path):
        print(f"❌ Path not found: {input_path}")
        sys.exit(1)
    
    print("🔍 BraTS Image Size Checker")
    print("=" * 60)
    print(f"Target size for nnUNet: (240, 240, 155)")
    print(f"Checking: {input_path}")
    print("=" * 60)
    
    if os.path.isfile(input_path):
        # Single file
        print(f"\nAnalyzing single file:")
        shape, spacing = check_image_dimensions(input_path)
        if shape:
            size_mb = os.path.getsize(input_path) / (1024 * 1024)
            print(f"  File: {os.path.basename(input_path)}")
            print(f"  Shape: {shape}")
            print(f"  Spacing: {[f'{s:.2f}' for s in spacing]}")
            print(f"  Size: {size_mb:.1f}MB")
        
    elif os.path.isdir(input_path):
        # Check if it's a case directory or parent directory
        case_files = [f for f in os.listdir(input_path) if f.endswith('.nii.gz')]
        subdirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
        
        if case_files:
            # Single case directory
            analyze_case_directory(input_path)
        elif subdirs:
            # Parent directory with multiple cases
            case_dirs = [d for d in subdirs if d.startswith('BraTS')]
            if not case_dirs:
                # Maybe nnUNet format - check for _0000.nii.gz files
                nnunet_files = [f for f in case_files if '_0000.nii.gz' in f]
                if nnunet_files:
                    print(f"\nFound nnUNet format files:")
                    shapes_summary = {}
                    
                    for i, file in enumerate(sorted(nnunet_files)[:5]):  # Check first 5
                        file_path = os.path.join(input_path, file)
                        shape, spacing = check_image_dimensions(file_path)
                        if shape:
                            if shape not in shapes_summary:
                                shapes_summary[shape] = 0
                            shapes_summary[shape] += 1
                            
                            case_name = file.replace('_0000.nii.gz', '')
                            print(f"  {case_name}: {shape}")
                    
                    print(f"\nShape summary:")
                    for shape, count in shapes_summary.items():
                        status = "✅ Good" if shape == (240, 240, 155) else "⚠️  Needs check"
                        print(f"  {shape}: {count} files {status}")
                else:
                    print(f"❌ No recognizable image files found")
            else:
                print(f"\nFound {len(case_dirs)} case directories:")
                
                shapes_summary = {}
                issues_found = []
                
                for case_dir in sorted(case_dirs)[:30]:
                    case_path = os.path.join(input_path, case_dir)
                    shapes = analyze_case_directory(case_path)
                    
                    # Track shapes
                    valid_shapes = [s for s in shapes.values() if s is not None]
                    if valid_shapes:
                        shape = valid_shapes[0]
                        if shape not in shapes_summary:
                            shapes_summary[shape] = 0
                        shapes_summary[shape] += 1
                        
                        if shape != (240, 240, 155):
                            issues_found.append((case_dir, shape))
                
                print(f"\n📊 SUMMARY:")
                print(f"Checked: {min(10, len(case_dirs))} cases")
                for shape, count in shapes_summary.items():
                    status = "✅ Ready for nnUNet" if shape == (240, 240, 155) else "⚠️  Needs restoration"
                    print(f"  {shape}: {count} cases {status}")
                
                if issues_found:
                    print(f"\n⚠️  Cases needing size restoration:")
                    for case, shape in issues_found[:5]:
                        print(f"    {case}: {shape} → should be (240, 240, 155)")
                
                if len(case_dirs) > 10:
                    print(f"\n(Showing first 10 of {len(case_dirs)} total cases)")
        else:
            print(f"❌ No case directories or image files found in {input_path}")
    
    print(f"\n" + "=" * 60)
    print(f"✅ Size check complete!")

if __name__ == "__main__":
    main()