#!/usr/bin/env python3
"""
Debug script to find problematic files in BRATS dataset
"""

import os
import glob

def debug_brats_dataset(data_dir):
    """Find files that don't match expected naming convention"""
    
    print(f"Debugging BRATS dataset in: {data_dir}")
    print(f"Directory exists: {os.path.exists(data_dir)}")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print(f"❌ ERROR: Directory does not exist: {data_dir}")
        print("Available directories:")
        parent = os.path.dirname(data_dir)
        if os.path.exists(parent):
            for item in os.listdir(parent):
                print(f"  - {item}")
        return
    
    # Get all .nii.gz files recursively
    pattern = os.path.join(data_dir, "**", "*.nii.gz")
    print(f"Searching pattern: {pattern}")
    all_files = glob.glob(pattern, recursive=True)
    print(f"Raw glob result: {len(all_files)} files")
    
    print(f"Found {len(all_files)} .nii.gz files total")
    print()
    
    problematic_files = []
    valid_files = []
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Try the current parsing logic
        try:
            parts = filename.split('-')
            if len(parts) < 5:
                problematic_files.append((filepath, f"Only {len(parts)} parts: {parts}"))
                continue
                
            seqtype = parts[4].split('.')[0]
            valid_files.append((filepath, seqtype))
            
        except Exception as e:
            problematic_files.append((filepath, f"Error: {e}"))
    
    print(f"✅ Valid files: {len(valid_files)}")
    print(f"❌ Problematic files: {len(problematic_files)}")
    print()
    
    if problematic_files:
        print("PROBLEMATIC FILES:")
        print("-" * 40)
        for filepath, error in problematic_files:
            print(f"❌ {filepath}")
            print(f"   └─ {error}")
        print()
    
    if valid_files:
        print("SAMPLE VALID FILES:")
        print("-" * 40)
        for filepath, seqtype in valid_files[:5]:
            print(f"✅ {os.path.basename(filepath)} → seqtype: '{seqtype}'")
        
        # Show sequence type distribution
        seqtypes = [seqtype for _, seqtype in valid_files]
        from collections import Counter
        counts = Counter(seqtypes)
        print(f"\nSequence type distribution:")
        for seqtype, count in counts.items():
            print(f"  {seqtype}: {count}")

if __name__ == "__main__":
    data_dir = "./datasets/BRATS2023/training"
    debug_brats_dataset(data_dir)