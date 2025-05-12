import os
import yaml

# 设置根目录路径
root_dir = '/home/handsomexd/EventAD/data/detector/ROL'

# 定义子文件夹路径
train_dir = os.path.join(root_dir, 'train')
val_dir = os.path.join(root_dir, 'val')
test_dir = os.path.join(root_dir, 'test')

def get_folder_names(directory):
    """获取指定目录下的文件夹名称列表，并将数字名称转换为字符串"""
    folder_names = []
    for folder in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder)):
            # 如果文件夹名是纯数字，则转换为字符串格式
            if folder.isdigit():
                folder_names.append(f'{folder}')
            else:
                folder_names.append(folder)
    return sorted(folder_names)

# 获取train, val, test的文件夹名称
train_folders = get_folder_names(train_dir)
val_folders = get_folder_names(val_dir)
test_folders = get_folder_names(test_dir)

# 创建一个字典，保存train, val, test文件夹名称
data = {
    'train': train_folders,
    'val': val_folders,
    'test': test_folders
}

# 输出到 YAML 文件
yaml_file_path = '/home/handsomexd/EventAD/config/rol_split.yaml'
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)

print(f"YAML 文件已保存到: {yaml_file_path}")