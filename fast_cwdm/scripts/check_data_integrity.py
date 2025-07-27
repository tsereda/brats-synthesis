#!/usr/bin/env python3
"""
Check for corrupted .nii.gz files in the dataset
"""

import os
import sys
import nibabel as nib
from pathlib import Path

def check_file_integrity(file_path):
    """Check if a single .nii.gz file can be loaded"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()  # This is where corruption usually shows up
        return True, None
    except Exception as e:
        return False, str(e)

def check_dataset_integrity(data_dir):
    """Check all .nii.gz files in dataset"""
    print(f"🔍 Checking data integrity in: {data_dir}")
    
    corrupted_files = []
    total_files = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                total_files += 1
                file_path = os.path.join(root, file)
                
                is_valid, error = check_file_integrity(file_path)
                
                if not is_valid:
                    print(f"❌ CORRUPTED: {file_path}")
                    print(f"   Error: {error}")
                    corrupted_files.append(file_path)
                else:
                    print(f"✅ OK: {os.path.basename(file_path)}")
    
    print(f"\n📊 SUMMARY:")
    print(f"Total files checked: {total_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Success rate: {((total_files - len(corrupted_files)) / total_files * 100):.1f}%")
    
    if corrupted_files:
        print(f"\n💥 CORRUPTED FILES:")
        for f in corrupted_files:
            print(f"  {f}")
        return False
    else:
        print(f"✅ All files are valid!")
        return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check BRATS dataset integrity")
    parser.add_argument("--data_dir", default="./datasets/BRATS2023/validation", 
                       help="Directory to check")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Directory not found: {args.data_dir}")
        sys.exit(1)
    
    success = check_dataset_integrity(args.data_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()