#!/usr/bin/env python3
"""
BraTS Data Validation and Repair Script
Identifies and handles corrupted NIfTI files
"""

import os
import glob
import nibabel as nib
import gzip
import tempfile
import shutil
from pathlib import Path

def validate_nifti_file(file_path):
    """
    Validate a NIfTI file and attempt to repair if corrupted
    Returns: (is_valid, error_message)
    """
    try:
        # Try to load with nibabel
        img = nib.load(file_path)
        
        # Try to access the data (this will trigger decompression)
        data = img.get_fdata()
        
        # Basic sanity checks
        if data.size == 0:
            return False, "Empty data array"
        
        if not isinstance(data, (list, tuple)) and (data.shape == (0,) or any(dim == 0 for dim in data.shape)):
            return False, "Invalid shape"
            
        return True, "Valid"
        
    except Exception as e:
        return False, str(e)

def attempt_repair_gzip(file_path):
    """
    Attempt to repair a corrupted .nii.gz file by recompressing
    """
    if not file_path.endswith('.nii.gz'):
        return False, "Not a gzipped file"
    
    try:
        # Create backup
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)
        
        # Try to decompress and recompress
        with gzip.open(file_path, 'rb') as gz_file:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                shutil.copyfileobj(gz_file, temp_file)
        
        # Recompress
        with open(temp_path, 'rb') as uncompressed:
            with gzip.open(file_path, 'wb', compresslevel=6) as gz_out:
                shutil.copyfileobj(uncompressed, gz_out)
        
        # Clean up
        os.unlink(temp_path)
        
        # Verify repair worked
        is_valid, msg = validate_nifti_file(file_path)
        if is_valid:
            os.unlink(backup_path)  # Remove backup if repair successful
            return True, "Repaired successfully"
        else:
            shutil.move(backup_path, file_path)  # Restore backup
            return False, f"Repair failed: {msg}"
            
    except Exception as e:
        # Restore backup if it exists
        backup_path = file_path + '.backup'
        if os.path.exists(backup_path):
            shutil.move(backup_path, file_path)
        return False, f"Repair attempt failed: {e}"

def validate_brats_dataset(data_dir, fix_corrupted=True):
    """
    Validate all NIfTI files in BraTS dataset
    """
    print(f"🔍 Validating BraTS dataset in: {data_dir}")
    
    all_nifti_files = []
    for pattern in ['**/*.nii.gz', '**/*.nii']:
        all_nifti_files.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    
    print(f"Found {len(all_nifti_files)} NIfTI files to validate")
    
    corrupted_files = []
    repaired_files = []
    
    for i, file_path in enumerate(all_nifti_files):
        if (i + 1) % 50 == 0:
            print(f"Validated {i + 1}/{len(all_nifti_files)} files...")
        
        is_valid, error_msg = validate_nifti_file(file_path)
        
        if not is_valid:
            print(f"❌ CORRUPTED: {file_path}")
            print(f"   Error: {error_msg}")
            corrupted_files.append((file_path, error_msg))
            
            if fix_corrupted and file_path.endswith('.nii.gz'):
                print(f"🔧 Attempting to repair: {file_path}")
                repaired, repair_msg = attempt_repair_gzip(file_path)
                if repaired:
                    print(f"✅ REPAIRED: {file_path}")
                    repaired_files.append(file_path)
                else:
                    print(f"❌ REPAIR FAILED: {repair_msg}")
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"Total files: {len(all_nifti_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Successfully repaired: {len(repaired_files)}")
    print(f"Still corrupted: {len(corrupted_files) - len(repaired_files)}")
    
    if corrupted_files:
        print(f"\n❌ CORRUPTED FILES:")
        for file_path, error in corrupted_files:
            if file_path not in repaired_files:
                print(f"  {file_path}: {error}")
    
    return len(corrupted_files) == len(repaired_files)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Validate and repair BraTS dataset')
    parser.add_argument('--data_dir', required=True, help='Path to BraTS data directory')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix corrupted files')
    args = parser.parse_args()
    
    success = validate_brats_dataset(args.data_dir, fix_corrupted=args.fix)
    
    if success:
        print("✅ All files are now valid!")
        exit(0)
    else:
        print("❌ Some files remain corrupted. Consider re-downloading the dataset.")
        exit(1)

if __name__ == "__main__":
    main()