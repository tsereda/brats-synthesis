"""
Create a pseudo validation set by randomly dropping one modality per case.
Adapted from BraSyn tutorial for fast-cwdm project structure.
Enhanced with command line argument support.
"""
import os
import random
import numpy as np
import shutil
import argparse

def create_pseudo_validation(input_dir, output_dir, drop_modality=None, seed=123456):
    """
    Create pseudo validation set by dropping modalities.
    
    Args:
        input_dir: Source directory with complete cases
        output_dir: Output directory for incomplete cases
        drop_modality: Specific modality to drop ('t1c', 't1n', 't2f', 't2w') or None for random
        seed: Random seed for reproducibility
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    modality_list = ['t1c', 't1n', 't2f', 't2w']  # Available modalities
    
    # Validate drop_modality if specified
    if drop_modality and drop_modality not in modality_list:
        raise ValueError(f"Invalid modality '{drop_modality}'. Must be one of: {modality_list}")
   
    folder_list = os.listdir(input_dir)
    folder_list.sort()
   
    if drop_modality:
        # Drop the same modality for all cases
        print(f"Dropping {drop_modality} for ALL cases")
        drop_indices = [modality_list.index(drop_modality)] * len(folder_list)
    else:
        # Randomly assign which modality to drop for each case
        print("Randomly dropping modalities per case")
        drop_indices = np.random.randint(0, 4, size=len(folder_list))
   
    print(f"Processing {len(folder_list)} cases...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
   
    modality_counts = {mod: 0 for mod in modality_list}
   
    for count, case_folder in enumerate(folder_list):
        case_src = os.path.join(input_dir, case_folder)
        case_dst = os.path.join(output_dir, case_folder)
       
        if not os.path.isdir(case_src):
            continue
           
        if not os.path.exists(case_dst):
            os.makedirs(case_dst)
       
        file_list = os.listdir(case_src)
        dropped_modality = modality_list[drop_indices[count]]
        modality_counts[dropped_modality] += 1
       
        print(f"Case {case_folder}: dropping {dropped_modality}")
       
        # Copy all files except the dropped modality
        copied_files = 0
        for filename in file_list:
            if dropped_modality not in filename:  # Keep files that don't contain dropped modality
                src_file = os.path.join(case_src, filename)
                dst_file = os.path.join(case_dst, filename)
                try:
                    shutil.copyfile(src_file, dst_file)
                    copied_files += 1
                except Exception as e:
                    print(f"  Warning: Failed to copy {filename}: {e}")
        
        print(f"  Copied {copied_files} files")
               
        # Create a marker file to indicate which modality is missing
        marker_file = os.path.join(case_dst, f"missing_{dropped_modality}.txt")
        with open(marker_file, 'w') as f:
            f.write(f"Missing modality: {dropped_modality}\n")
            f.write(f"Seed used: {seed}\n")
            f.write(f"Mode: {'specific' if drop_modality else 'random'}\n")
   
    print(f"\nPseudo validation set created successfully!")
    print(f"Total cases processed: {len(folder_list)}")
    print(f"\nModality drop distribution:")
    for modality, count in modality_counts.items():
        percentage = (count / len(folder_list)) * 100 if folder_list else 0
        print(f"  {modality}: {count} cases ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(
        description="Create pseudo validation set by dropping modalities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random dropping (original behavior):
  python drop_modality.py
  
  # Drop specific modality for all cases:
  python drop_modality.py --drop_modality t1n
  python drop_modality.py --drop_modality t2w
  
  # Custom directories:
  python drop_modality.py --input_dir /path/to/data --output_dir custom_output
  
  # Different random seed:
  python drop_modality.py --seed 42
        """
    )
    
    parser.add_argument(
        "--input_dir", 
        default="ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData",
        help="Input directory containing complete cases (default: ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData)"
    )
    
    parser.add_argument(
        "--output_dir", 
        default="pseudo_validation",
        help="Output directory for incomplete cases (default: pseudo_validation)"
    )
    
    parser.add_argument(
        "--drop_modality", 
        choices=['t1c', 't1n', 't2f', 't2w'],
        default=None,
        help="Specific modality to drop for ALL cases. If not specified, randomly drops modalities."
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123456,
        help="Random seed for reproducibility (default: 123456)"
    )
    
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Show what would be done without actually copying files"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory!")
        return 1
    
    # Show configuration
    print("=== Configuration ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Drop modality: {args.drop_modality or 'Random'}")
    print(f"Random seed: {args.seed}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 20)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be copied")
        # TODO: Implement dry run logic
        return 0
    
    try:
        create_pseudo_validation(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            drop_modality=args.drop_modality,
            seed=args.seed
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())