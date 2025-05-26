import numpy as np
import os

# Root path
base_path = '/home/handsomexd/EventAD/data/feature/ROL'
detector_base_path = '/home/handsomexd/EventAD/data/detector/ROL'
folders = ['train']  # Folders to process

# Iterate through train and val folders
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    # Iterate through all .npz files in the folder
    for npz_file_name in os.listdir(folder_path):
        if not npz_file_name.endswith('.npz'):
            continue  # Skip non-npz files
        
        npz_file_path = os.path.join(folder_path, npz_file_name)
        
        # Read npz file
        npz_file = np.load(npz_file_path)
        detection = npz_file['detection']  # Assuming shape is (100, 30, 6)
        vid_id = npz_file['vid_id']  # Video ID
        
        # Find corresponding timestamps.txt file
        video_name = os.path.splitext(npz_file_name)[0]  # Get video name
        timestamps_path = os.path.join(detector_base_path, folder, video_name, 'images', 'timestamps.txt')
        if not os.path.exists(timestamps_path):
            print(f"Timestamp file does not exist: {timestamps_path}")
            continue
        
        # Read timestamps file
        timestamps = np.loadtxt(timestamps_path, dtype=np.uint64)
        
        # Verify that timestamps match the number of detection frames
        # assert len(timestamps) == detection.shape[0], f"Timestamps length does not match number of frames in detection for {npz_file_name}."
        
        # Initialize list to save data
        tracks_list = []
        
        # Iterate through each frame
        for i, ts in enumerate(timestamps):
            for j in range(detection.shape[1]):  # Iterate through each detection box in the frame
                track_data = detection[i, j]  # 6 values for each detection box (track_id, x1, y1, x2, y2)
                
                track_id = track_data[0]  # track_id
                if track_id == 0:  # Assume detection boxes with track_id=0 are invalid
                    continue
                
                # Convert bbox (x1, y1, x2, y2) to (x, y, w, h)
                x1, y1, x2, y2 = track_data[1:5]
                x = x1  # Center x
                y = y1  # Center y
                w = x2 - x1  # Width
                h = y2 - y1  # Height
                # if track_data[5] == 0:
                #     class_id = 0  # Set all class_id to 2
                # if track_data[5] == 1:
                class_id = track_data[5]
                print(class_id)
                confidence = 1  # Set all class_confidence to 0.88
                
                # Add this detection box to tracks_list
                tracks_list.append((ts, x, y, w, h, class_id, confidence, track_id))
        
        # Convert data to numpy array and save as .npy file
        dtype = [('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')]
        tracks_array = np.array(tracks_list, dtype=dtype)
        
        # Determine output file path
        output_dir = os.path.join(detector_base_path, folder, video_name, 'object_detections', 'left')
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, 'tracks.npy')
        
        # Save as tracks.npy file
        np.save(output_file_path, tracks_array)
        
        print(f"{output_file_path} file generated, containing {len(tracks_array)} entries.")