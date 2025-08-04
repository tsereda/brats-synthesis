import multiprocessing
import shutil
import os
import argparse
from multiprocessing import Pool
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def convert_labels_back_to_BraTS(seg: np.ndarray):
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 4
    new_seg[seg == 2] = 1
    return new_seg

def load_convert_labels_back_to_BraTS(filename, input_folder, output_folder):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))

def convert_folder_with_preds_back_to_BraTS_labeling_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))

if __name__ == '__main__':
    # ✅ ADD ARGUMENT PARSING
    parser = argparse.ArgumentParser(description='Convert BraTS data to nnUNet format')
    parser.add_argument('--input_dir', required=True, help='Input directory with BraTS data')
    parser.add_argument('--max_cases', type=int, default=None, help='Maximum number of cases to process')
    args = parser.parse_args()
    
    # ✅ USE THE PARSED ARGUMENT INSTEAD OF HARDCODED PATH
    brats_data_dir = args.input_dir
    
    print(f"Using input directory: {brats_data_dir}")
    
    # ✅ CHECK IF DIRECTORY EXISTS
    if not os.path.exists(brats_data_dir):
        print(f"ERROR: Directory {brats_data_dir} does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print("Available files/directories:")
        try:
            print(os.listdir('.'))
        except:
            print("Cannot list current directory")
        exit(1)
    
    task_id = 137
    task_name = "BraTS2021_inference"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)
    
    # setting up nnU-Net folders (images only - no labels for inference)
    out_base = join('./', foldername)
    imagestr = join(out_base, "imagesTs")
    maybe_mkdir_p(imagestr)
    
    # Get case directories
    try:
        case_ids = subdirs(brats_data_dir, prefix='BraTS', join=False)
    except Exception as e:
        print(f"ERROR: Cannot read directories from {brats_data_dir}: {e}")
        print("Directory contents:")
        try:
            print(os.listdir(brats_data_dir))
        except:
            print("Cannot list directory contents")
        exit(1)
    
    print(f"Found {len(case_ids)} cases for inference")
    
    # ✅ APPLY MAX_CASES LIMIT IF SPECIFIED
    if args.max_cases and len(case_ids) > args.max_cases:
        case_ids = case_ids[:args.max_cases]
        print(f"Limited to {len(case_ids)} cases due to max_cases={args.max_cases}")
    
    if len(case_ids) == 0:
        print("ERROR: No BraTS cases found!")
        print(f"Looking for directories starting with 'BraTS' in: {brats_data_dir}")
        print("Available subdirectories:")
        try:
            all_subdirs = [d for d in os.listdir(brats_data_dir) if os.path.isdir(os.path.join(brats_data_dir, d))]
            print(all_subdirs[:10])  # Show first 10
        except:
            print("Cannot list subdoirectories")
        exit(1)
    
    print(f"First few cases: {case_ids[:3]}")
    
    processed_count = 0
    for c in case_ids:
        print(f"Processing case: {c}")
        
        # Check if all 4 modalities exist
        t1n_file = join(brats_data_dir, c, c + "-t1n.nii.gz")
        t1c_file = join(brats_data_dir, c, c + "-t1c.nii.gz")
        t2w_file = join(brats_data_dir, c, c + "-t2w.nii.gz")
        t2f_file = join(brats_data_dir, c, c + "-t2f.nii.gz")
        
        missing_files = []
        if not os.path.exists(t1n_file): missing_files.append("t1n")
        if not os.path.exists(t1c_file): missing_files.append("t1c")
        if not os.path.exists(t2w_file): missing_files.append("t2w")
        if not os.path.exists(t2f_file): missing_files.append("t2f")
        
        if missing_files:
            print(f"   ❌ Skipping {c} - missing: {missing_files}")
            continue
            
        # Copy files in nnUNet format
        shutil.copy(t1n_file, join(imagestr, c + '_0000.nii.gz'))
        shutil.copy(t1c_file, join(imagestr, c + '_0001.nii.gz'))
        shutil.copy(t2w_file, join(imagestr, c + '_0002.nii.gz'))
        shutil.copy(t2f_file, join(imagestr, c + '_0003.nii.gz'))
        
        processed_count += 1
        print(f"   ✅ Converted {c}")
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 nnUNet inference data ready at: {imagestr}")
    print(f"Processed {processed_count} cases successfully")
    
    # Create symlink for nnUNet (if needed) - with error handling
    try:
        nnunet_dataset_path = join(nnUNet_raw, foldername)
        if not os.path.exists(nnunet_dataset_path):
            os.symlink(os.path.abspath(out_base), nnunet_dataset_path)
            print(f"🔗 Created symlink: {nnunet_dataset_path}")
        else:
            print(f"📁 nnUNet dataset already exists at: {nnunet_dataset_path}")
    except Exception as e:
        print(f"⚠️  Could not create nnUNet symlink: {e}")
        print(f"    You can manually copy the dataset to your nnUNet_raw folder when ready.")