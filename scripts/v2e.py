# generate v2e command
import os
import subprocess

def process_videos(input_folder):
    """处理目录中的所有 MP4 文件"""
    # 遍历指定目录中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名是否为 .mp4
        if filename.lower().endswith('.mp4'):
            # 构造完整的文件路径
            input_path = os.path.join(input_folder, filename)
            
            # 提取文件名（不包括扩展名）作为 dvs_h5 的名称
            base_name = os.path.splitext(filename)[0]
            dvs_h5_name = f"{base_name}.h5"
            
            # 构造命令行
            command = [
                'python', '/home/handsomexd/v2e/v2e.py',
                '--input', input_path,
                '--dvs_h5', dvs_h5_name
            ]
            
            # 打印正在执行的命令（可选）
            print(f"Executing command: {' '.join(command)}")
            
            # 执行命令
            subprocess.run(command, check=True)

if __name__ == '__main__':
    # 指定包含 MP4 文件的目录
    input_folder = '/home/handsomexd/EventAD/data/video/ROL/val'
    
    # 处理视频文件
    process_videos(input_folder)
