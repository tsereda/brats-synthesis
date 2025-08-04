#!/usr/bin/env python3

import os
import shutil
import zipfile
from pathlib import Path

def prepare_validation_submission():
    """
    Prepare validation submission for BraSyn Challenge
    - Input: nnUNet segmentation outputs
    - Output: Properly formatted submission archive
    """
    
    output_dir = "fixed_nii_gz_files"
    submission_dir = "validation_submission"
    
    print("Preparing BraSyn Validation Submission...")
    
    # Create submission directory
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir)
    
    # Find all segmentation files
    seg_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.nii.gz'):
                seg_files.append(file)
    
    if not seg_files:
        print("No segmentation files found in ./outputs/")
        print("Make sure nnUNet prediction completed successfully.")
        return False
    
    print(f"Found {len(seg_files)} segmentation files:")
    
    # Process each segmentation file
    processed_count = 0
    for seg_file in seg_files:
        print(f"  Processing: {seg_file}")
        
        # Parse filename - should be like: BraTS-GLI-00001-000.nii.gz
        if seg_file.count('-') >= 3:
            # Already in correct format
            target_name = seg_file
        else:
            print(f"  ⚠️  Unexpected filename format: {seg_file}")
            # Try to keep original name
            target_name = seg_file
        
        # Copy to submission directory
        source_path = os.path.join(output_dir, seg_file)
        target_path = os.path.join(submission_dir, target_name)
        
        shutil.copy2(source_path, target_path)
        processed_count += 1
        print(f"  Copied to: {target_name}")
    
    print(f"\nSummary:")
    print(f"  - Processed {processed_count} segmentation files")
    print(f"  - Files ready in: {submission_dir}")
    
    # List final submission files
    print(f"\nSubmission files:")
    for file in sorted(os.listdir(submission_dir)):
        file_path = os.path.join(submission_dir, file)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file} ({size_mb:.1f} MB)")
    
    # Create submission archive
    archive_name = "brasyn_validation_submission.zip"
    print(f"\nCreating submission archive: {archive_name}")
    
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(submission_dir):
            file_path = os.path.join(submission_dir, file)
            zipf.write(file_path, file)  # Store with just filename, not full path
    
    archive_size = os.path.getsize(archive_name) / (1024 * 1024)
    print(f"Archive created: {archive_name} ({archive_size:.1f} MB)")
    
    print(f"\nValidation submission ready!")
    print(f"Submit this file: {archive_name}")
    print(f"Contains {processed_count} segmentation masks in NIfTI format")
    
    # Final verification
    print(f"\nFinal verification:")
    with zipfile.ZipFile(archive_name, 'r') as zipf:
        files_in_zip = zipf.namelist()
        print(f"  - Archive contains {len(files_in_zip)} files")
        for file in sorted(files_in_zip):
            print(f"    ✓ {file}")
    
    # Quality check - inspect segmentation content
    print(f"\nQuality Check - Segmentation Content:")
    try:
        import nibabel as nib
        import numpy as np
        
        for file in sorted(os.listdir(submission_dir)):
            if file.endswith('.nii.gz'):
                file_path = os.path.join(submission_dir, file)
                try:
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    
                    unique_labels = np.unique(data)
                    non_zero_voxels = np.sum(data > 0)
                    total_voxels = data.size
                    tumor_percentage = (non_zero_voxels / total_voxels) * 100
                    
                    print(f"{file}:")
                    print(f"    - Shape: {data.shape}")
                    print(f"    - Labels: {sorted(unique_labels.astype(int))}")
                    print(f"    - Tumor voxels: {non_zero_voxels:,} ({tumor_percentage:.2f}%)")
                    print(f"    - Data type: {data.dtype}")
                    print(f"    - Value range: [{data.min():.1f}, {data.max():.1f}]")
            
                except Exception as e:
                    print(f"Error reading {file}: {e}")
                    
    except ImportError:
        print("nibabel not available - skipping content check")
        print("Install with: pip install nibabel")
    except Exception as e:
        print(f" Quality check failed: {e}")
    
    return True

if __name__ == "__main__":
    success = prepare_validation_submission()
    if success:
        print("\nReady to submit to BraSyn Challenge validation!")
    else:
        print("\nSubmission preparation failed. Check your segmentation outputs.")