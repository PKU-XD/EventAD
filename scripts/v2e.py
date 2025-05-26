# generate v2e command
import os
import subprocess

def process_videos(input_folder):
    """Process all MP4 files in the directory"""
    # Iterate through all files in the specified directory
    for filename in os.listdir(input_folder):
        # Check if file extension is .mp4
        if filename.lower().endswith('.mp4'):
            # Construct complete file path
            input_path = os.path.join(input_folder, filename)
            
            # Extract filename (without extension) as dvs_h5 name
            base_name = os.path.splitext(filename)[0]
            dvs_h5_name = f"{base_name}.h5"
            
            # Construct command line
            command = [
                'python', '/home/handsomexd/v2e/v2e.py',
                '--input', input_path,
                '--dvs_h5', dvs_h5_name
            ]
            
            # Print command being executed (optional)
            print(f"Executing command: {' '.join(command)}")
            
            # Execute command
            subprocess.run(command, check=True)

if __name__ == '__main__':
    # Specify directory containing MP4 files
    input_folder = '/home/handsomexd/EventAD/data/video/ROL/val'
    
    # Process video files
    process_videos(input_folder)