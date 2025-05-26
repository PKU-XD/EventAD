import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_rate):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images_folder = os.path.join(output_folder, "images", "left", "distorted")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video file {video_path}")
        return

    # Get original frame rate of the video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // frame_rate)
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save images based on frame interval
        if frame_count % frame_interval == 0:
            
            img_name = f"{saved_frame_count:06d}.png"
            img_path = os.path.join(images_folder, img_name)
            cv2.imwrite(img_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Video {video_path} extraction completed, saved {saved_frame_count} frames.")


def convert_videos_in_folder(input_folder, output_base_folder, frame_rate):
    # Iterate through all .mp4 files
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp4'):
            video_path = os.path.join(input_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(output_base_folder, video_name)
            
            # Extract video frames
            extract_frames_from_video(video_path, output_folder, frame_rate)


if __name__ == "__main__":
    input_folder = "/home/handsomexd/EventAD/data/video/ROL/train"  # Input video folder path
    output_base_folder = "/home/handsomexd/EventAD/data/detector/ROL/train"  # Output folder path
    frame_rate = 20

    convert_videos_in_folder(input_folder, output_base_folder, frame_rate)