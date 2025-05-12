import os
import shutil

# 定义源文件夹路径
src_base_dir = '/home/handsomexd/EventAD/data/Event/ROL/h5'
# 定义目标文件夹路径
dst_base_dir = '/home/handsomexd/EventAD/data/detector/ROL'

# 遍历文件夹中的所有h5文件
for phase in ['train', 'val']:
    # 构造源和目标的完整路径
    src_dir = os.path.join(src_base_dir, phase)
    dst_dir = os.path.join(dst_base_dir, phase)

    # 如果目标文件夹不存在，跳过该文件夹
    if not os.path.exists(dst_dir):
        print(f"目标路径 {dst_dir} 不存在，跳过。")
        continue

    # 遍历源路径中的所有h5文件
    for filename in os.listdir(src_dir):
        if filename.endswith('.h5'):
            src_file = os.path.join(src_dir, filename)
            # 获取不带扩展名的文件名
            file_basename = os.path.splitext(filename)[0]
            dst_folder = os.path.join(dst_dir, file_basename)

            if os.path.exists(dst_folder):
                # 创建events文件夹
                events_folder = os.path.join(dst_folder, 'events')
                os.makedirs(events_folder, exist_ok=True)

                # 将文件复制并重命名为 events.h5
                dst_file = os.path.join(events_folder, 'events.h5')
                shutil.copy2(src_file, dst_file)
                print(f"文件 {filename} 已复制并重命名为 {dst_file}")
            else:
                print(f"目标文件夹 {dst_folder} 不存在，跳过 {filename}")