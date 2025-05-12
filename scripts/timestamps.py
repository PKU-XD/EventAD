import h5py
import os

# 根路径
base_path = '/home/handsomexd/EventAD/data/detector/ROL'
folders = ['train', 'val']  # 要处理的文件夹

# 设定帧率与时间戳间隔
fps = 20
time_interval = 50000  # 50ms = 50000的时间戳

# 遍历 train 和 val 文件夹
for folder in folders:
    folder_path = os.path.join(base_path, folder)
    
    # 遍历文件夹中的所有子文件夹
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        # 确认子文件夹存在 `events/events.h5`
        h5_file_path = os.path.join(subfolder_path, 'events', 'events.h5')
        if not os.path.exists(h5_file_path):
            print(f"文件不存在: {h5_file_path}")
            continue

        # 创建输出目录
        output_dir = os.path.join(subfolder_path, 'images')
        os.makedirs(output_dir, exist_ok=True)

        output_txt_path = os.path.join(output_dir, 'timestamps.txt')

        # 读取 h5 文件
        with h5py.File(h5_file_path, 'r') as f:
            events = f['events'][:]  # 读取数据，假设时间戳是第一列
            timestamps = events[:, 0]  # 提取时间戳列

        # 生成时间段，并为每个时间段找到对应的最小时间戳
        frame_timestamps = []
        for start in range(0, 5000000, time_interval):
            # 查找当前时间段的最小时间戳
            mask = (timestamps >= start) & (timestamps < start + time_interval)
            if mask.any():
                frame_timestamps.append(timestamps[mask].min())

        # 将结果写入文件
        with open(output_txt_path, 'w') as f:
            for ts in frame_timestamps:
                f.write(f"{ts}\n")

        print(f"时间戳已写入: {output_txt_path}")