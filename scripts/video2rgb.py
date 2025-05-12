import cv2
import os

def extract_frames_from_video(video_path, output_folder, frame_rate):
    # 创建output目录，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images_folder = os.path.join(output_folder, "images", "left", "distorted")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件 {video_path}")
        return

    # 获取视频的原始帧率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // frame_rate)
    
    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据帧间隔保存图片
        if frame_count % frame_interval == 0:
            
            img_name = f"{saved_frame_count:06d}.png"
            img_path = os.path.join(images_folder, img_name)
            cv2.imwrite(img_path, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"视频 {video_path} 提取完成, 共保存 {saved_frame_count} 帧.")


def convert_videos_in_folder(input_folder, output_base_folder, frame_rate):
    # 遍历所有.mp4文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.mp4'):
            video_path = os.path.join(input_folder, file_name)
            video_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(output_base_folder, video_name)
            
            # 提取视频帧
            extract_frames_from_video(video_path, output_folder, frame_rate)


if __name__ == "__main__":
    input_folder = "/home/handsomexd/EventAD/data/video/ROL/train"  # 输入视频文件夹路径
    output_base_folder = "/home/handsomexd/EventAD/data/detector/ROL/train"  # 输出文件夹路径
    frame_rate = 20

    convert_videos_in_folder(input_folder, output_base_folder, frame_rate)