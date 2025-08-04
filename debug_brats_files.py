#!/usr/bin/env python3
"""
Debug script that exactly mimics the original BRATSVolumes logic
"""

import os

def debug_exact_loader_logic(directory):
    """Mimic the EXACT logic from the original BRATSVolumes class"""
    
    print(f"Debugging EXACT loader logic in: {directory}")
    print("=" * 60)
    
    database = []
    problematic_files = []
    
    for root, dirs, files in os.walk(directory):
        print(f"\nProcessing directory: {root}")
        print(f"  Subdirectories: {dirs}")
        print(f"  Files: {len(files)}")
        
        # This is the exact condition from the original code
        if not dirs:
            print(f"  ✅ Processing files (no subdirs): {files}")
            files.sort()
            datapoint = dict()
            
            # Process each file exactly like the original
            for f in files:
                print(f"    Processing file: {f}")
                try:
                    # This is the exact line that's failing
                    parts = f.split('-')
                    print(f"      Parts: {parts} (count: {len(parts)})")
                    
                    if len(parts) < 5:
                        problematic_files.append(f"❌ {f}: Only {len(parts)} parts")
                        continue
                    
                    seqtype = parts[4].split('.')[0]
                    print(f"      ✅ seqtype: '{seqtype}'")
                    datapoint[seqtype] = os.path.join(root, f)
                    
                except Exception as e:
                    problematic_files.append(f"❌ {f}: {e}")
                    print(f"      ❌ ERROR: {e}")
            
            if datapoint:
                database.append(datapoint)
                print(f"    Added datapoint with keys: {list(datapoint.keys())}")
        else:
            print(f"  ⏭️  Skipping (has subdirs)")
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"✅ Successfully processed datapoints: {len(database)}")
    print(f"❌ Problematic files: {len(problematic_files)}")
    
    if problematic_files:
        print(f"\nPROBLEMATIC FILES:")
        for prob in problematic_files:
            print(f"  {prob}")
    
    if database:
        print(f"\nSAMPLE DATAPOINTS:")
        for i, dp in enumerate(database[:3]):
            print(f"  {i+1}. Keys: {list(dp.keys())}")

if __name__ == "__main__":
    directory = "./datasets/BRATS2023/training"
    debug_exact_loader_logic(directory)