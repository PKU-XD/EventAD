import os
import numpy as np
import json
import re

def extract_toa_values(feature_dir):
    """Extract TOA values for each video from feature files"""
    toa_dict = {}
    
    # Get all npz files
    npz_files = [f for f in os.listdir(feature_dir) if f.endswith('.npz')]
    
    print(f"Found {len(npz_files)} video feature files")
    
    for npz_file in npz_files:
        file_path = os.path.join(feature_dir, npz_file)
        try:
            # Load npz file
            data = np.load(file_path, allow_pickle=True)
            
            # Extract video ID and TOA value
            vid_id = str(data['vid_id']) if 'vid_id' in data else os.path.splitext(npz_file)[0]
            toa = int(data['toa']) if 'toa' in data else None
            
            if toa is not None:
                # Save original video ID and TOA value
                toa_dict[vid_id] = toa
                
                # Also save possible alternative format video IDs
                vid_num = re.match(r'(\d+)', vid_id)
                if vid_num:
                    alt_id = f"video_{vid_num.group(1)}"
                    toa_dict[alt_id] = toa
                    # Prevent duplicate additions
                    if vid_id != alt_id:
                        toa_dict[f"video_{vid_id}"] = toa
                
                print(f"Video {vid_id}: TOA = {toa}")
            else:
                print(f"Warning: Could not extract TOA value from file {npz_file}")
        
        except Exception as e:
            print(f"Error processing file {npz_file}: {e}")
    
    print(f"Successfully extracted TOA values for {len(toa_dict)} videos")
    return toa_dict

if __name__ == "__main__":
    # Feature files directory
    feature_dir = "/home/handsomexd/EventAD/data/feature/ROL/val"
    
    # Extract TOA values
    toa_dict = extract_toa_values(feature_dir)
    
    # Save TOA values to JSON file
    output_file = "/home/handsomexd/EventAD/data/toa_values.json"
    with open(output_file, 'w') as f:
        json.dump(toa_dict, f, indent=4)
    
    print(f"TOA values saved to {output_file}")