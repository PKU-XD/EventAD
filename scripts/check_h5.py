#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import h5py
import argparse
from tqdm import tqdm
import traceback

def check_h5_file(filepath):
    """
    Check if a single H5 file can be opened and read normally
    
    Args:
        filepath: H5 file path
        
    Returns:
        (bool, str): (success status, error message)
    """
    try:
        with h5py.File(filepath, 'r') as f:
            # Check if file can be opened
            keys = list(f.keys())
            # Read some data to ensure file content is accessible
            for key in keys:
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    # Check if dataset is readable
                    shape = data.shape
                    # Read a small portion of data
                    if len(shape) > 0 and shape[0] > 0:
                        sample = data[0:min(10, shape[0])]
        return True, ""
    except OSError as e:
        if "bad object header version number" in str(e):
            return False, f"Bad object header version: {str(e)}"
        else:
            return False, f"OSError: {str(e)}"
    except Exception as e:
        return False, f"Exception: {str(e)}\n{traceback.format_exc()}"

def find_events_files(root_dir):
    """
    Find all events_2x.h5 files in the specified directory
    
    Args:
        root_dir: Root directory
        
    Returns:
        list: List of paths containing all found events_2x.h5 files
    """
    events_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "events_2x.h5":
                events_files.append(os.path.join(root, file))
    return events_files

def main():
    parser = argparse.ArgumentParser(description="Check events_2x.h5 files for issues")
    parser.add_argument("--dir", type=str, default="/home/handsomexd/EventAD/data/detector/ROL/train", help="Root directory containing events_2x.h5 files")
    parser.add_argument("--output", type=str, default="problematic_files.txt", help="Output path for list of problematic files")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix problematic files")
    args = parser.parse_args()
    
    # Find all events_2x.h5 files
    print(f"Searching for events_2x.h5 files in directory '{args.dir}'...")
    events_files = find_events_files(args.dir)
    print(f"Found {len(events_files)} events_2x.h5 files")
    
    if len(events_files) == 0:
        print("No events_2x.h5 files found, please check the directory path")
        return
    
    # Check each file
    problematic_files = []
    print("Starting file checks...")
    
    for filepath in tqdm(events_files, desc="Checking files"):
        success, error_msg = check_h5_file(filepath)
        if not success:
            problematic_files.append((filepath, error_msg))
            print(f"\nProblematic file: {filepath}")
            print(f"Error: {error_msg}")
    
    # Output list of problematic files
    if problematic_files:
        print(f"\nFound {len(problematic_files)} problematic files")
        with open(args.output, "w") as f:
            f.write("# List of problematic events_2x.h5 files\n")
            for filepath, error_msg in problematic_files:
                f.write(f"{filepath}\t{error_msg}\n")
        print(f"List of problematic files saved to {args.output}")
        
        # Attempt to fix files
        if args.fix:
            print("\nAttempting to fix problematic files...")
            for filepath, _ in tqdm(problematic_files, desc="Fixing files"):
                try:
                    # Backup original file
                    backup_path = filepath + ".bak"
                    if not os.path.exists(backup_path):
                        os.rename(filepath, backup_path)
                        print(f"Backed up: {filepath} -> {backup_path}")
                    
                    # Attempt to fix using h5repack
                    temp_path = filepath + ".temp"
                    os.system(f"h5repack -i {backup_path} -o {temp_path}")
                    
                    # Check fixed file
                    success, _ = check_h5_file(temp_path)
                    if success:
                        os.rename(temp_path, filepath)
                        print(f"Successfully fixed: {filepath}")
                    else:
                        print(f"Fix failed: {filepath}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                except Exception as e:
                    print(f"Error during fix process: {filepath}, Error: {str(e)}")
    else:
        print("\nAll files checked successfully, no issues found")

if __name__ == "__main__":
    main()