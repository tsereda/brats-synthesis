"""
Fixed version of bratsloader.py with robust filename parsing
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


class BRATSVolumes(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.files = []
        
        # Get all .nii.gz files
        pattern = os.path.join(data_dir, "**", "*.nii.gz")
        all_files = glob.glob(pattern, recursive=True)
        
        print(f"Found {len(all_files)} .nii.gz files")
        
        # Group files by case and filter valid ones
        cases = {}
        valid_files = []
        problematic_files = []
        
        for filepath in all_files:
            filename = os.path.basename(filepath)
            
            # Parse filename with error handling
            try:
                parts = filename.split('-')
                
                # Check if we have enough parts
                if len(parts) < 5:
                    problematic_files.append(f"Not enough parts: {filename}")
                    continue
                
                # Extract sequence type
                seqtype_with_ext = parts[4]  # e.g., "t1n.nii.gz"
                seqtype = seqtype_with_ext.split('.')[0]  # e.g., "t1n"
                
                # Validate sequence type
                valid_seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
                if seqtype not in valid_seqtypes:
                    problematic_files.append(f"Invalid seqtype '{seqtype}': {filename}")
                    continue
                
                # Create case identifier
                case_id = '-'.join(parts[:4])  # e.g., "BraTS-GLI-01145-000"
                
                if case_id not in cases:
                    cases[case_id] = {}
                
                cases[case_id][seqtype] = filepath
                valid_files.append(filepath)
                
            except Exception as e:
                problematic_files.append(f"Parse error in {filename}: {e}")
                continue
        
        # Report results
        print(f"✅ Valid files: {len(valid_files)}")
        if problematic_files:
            print(f"❌ Problematic files: {len(problematic_files)}")
            for prob in problematic_files[:5]:  # Show first 5
                print(f"   {prob}")
            if len(problematic_files) > 5:
                print(f"   ... and {len(problematic_files) - 5} more")
        
        # Filter complete cases (having all required modalities)
        required_modalities = ['t1n', 't1c', 't2w', 't2f']
        complete_cases = []
        
        for case_id, modalities in cases.items():
            if all(mod in modalities for mod in required_modalities):
                complete_cases.append(case_id)
            else:
                missing = [mod for mod in required_modalities if mod not in modalities]
                print(f"⚠️  Incomplete case {case_id}, missing: {missing}")
        
        print(f"Complete cases: {len(complete_cases)}")
        
        # Store the complete cases
        self.cases = {case_id: cases[case_id] for case_id in complete_cases}
        self.case_ids = list(self.cases.keys())
        
        if len(self.case_ids) == 0:
            raise RuntimeError("No complete cases found! Check your dataset structure.")
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        case_files = self.cases[case_id]
        
        # Load all modalities for this case
        volumes = {}
        for modality in ['t1n', 't1c', 't2w', 't2f']:
            filepath = case_files[modality]
            nii = nib.load(filepath)
            volume = nii.get_fdata().astype(np.float32)
            volumes[modality] = torch.from_numpy(volume)
        
        return volumes


# Alternative: Quick fix for existing loader
def robust_parse_filename(filename):
    """
    Robust filename parser that handles edge cases
    """
    try:
        parts = filename.split('-')
        
        if len(parts) < 5:
            print(f"Warning: Filename '{filename}' has only {len(parts)} parts, skipping")
            return None
        
        seqtype = parts[4].split('.')[0]
        
        # Validate sequence type
        valid_seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        if seqtype not in valid_seqtypes:
            print(f"Warning: Unknown sequence type '{seqtype}' in '{filename}', skipping")
            return None
        
        return seqtype
        
    except Exception as e:
        print(f"Error parsing filename '{filename}': {e}")
        return None


# Example of how to patch the existing loader:
"""
In your existing bratsloader.py, replace line 36:

OLD:
seqtype = f.split('-')[4].split('.')[0]

NEW:
seqtype = robust_parse_filename(f)
if seqtype is None:
    continue  # Skip this file
"""