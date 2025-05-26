import h5py
import os

# Root path
base_path = '/home/handsomexd/EventAD/data/detector/ROL'
folders = ['train', 'val']  # Folders to process

# Set frame rate and timestamp interval
fps = 20
time_interval = 50000  # 50ms = 50000 timestamp units

# Iterate through train and val folders
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    # Iterate through all subfolders
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # Confirm that `events/events.h5` exists in the subfolder
        h5_file_path = os.path.join(subfolder_path, 'events', 'events.h5')
        if not os.path.exists(h5_file_path):
            print(f"File does not exist: {h5_file_path}")
            continue

        # Create output directory
        output_dir = os.path.join(subfolder_path, 'images')
        os.makedirs(output_dir, exist_ok=True)

        output_txt_path = os.path.join(output_dir, 'timestamps.txt')

        # Read h5 file
        with h5py.File(h5_file_path, 'r') as f:
            events = f['events'][:]  # Read data, assuming timestamp is the first column
            timestamps = events[:, 0]  # Extract timestamp column

        # Generate time periods and find the minimum timestamp for each period
        frame_timestamps = []
        for start in range(0, 5000000, time_interval):
            # Find the minimum timestamp for the current time period
            mask = (timestamps >= start) & (timestamps < start + time_interval)
            if mask.any():
                frame_timestamps.append(timestamps[mask].min())

        # Write results to file
        with open(output_txt_path, 'w') as f:
            for ts in frame_timestamps:
                f.write(f"{ts}\n")

        print(f"Timestamps written to: {output_txt_path}")