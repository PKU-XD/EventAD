import h5py

def print_h5_info(h5_file, indent=0):
    """递归地打印H5文件的信息"""
    for key in h5_file.keys():
        item = h5_file[key]
        print("  " * indent + f"键名: {key}")
        
        if isinstance(item, h5py.Dataset):
            # 打印数据集的信息
            print("  " * indent + f"  数据集形状: {item.shape}")
            print("  " * indent + f"  数据集类型: {item.dtype}")
            print("  " * indent + f"  数据集大小: {item.size}")
            print("  " * indent + f"  数据占用内存: {item.nbytes} 字节")
            print("  " * indent + f"  部分数据预览: {item[()]}")  # 只显示前5个数据
        elif isinstance(item, h5py.Group):
            # 如果是组（类似文件夹），递归打印
            print("  " * indent + "  组 (Group):")
            print_h5_info(item, indent + 1)

def read_h5_file(file_path):
    # 打开并读取h5文件
    with h5py.File(file_path, 'r') as h5_file:
        print(f"文件名: {file_path}")
        print(f"包含的数据结构:")
        print_h5_info(h5_file)

if __name__ == "__main__":
    # h5_file_path = "/home/handsomexd/EventAD/data/detector/ROL/test/thun_01_a/events/left/events_2x.h5"  
    h5_file_path = "/home/handsomexd/EventAD/data/detector/ROL/val/31_M/events/left/events_2x.h5" # x.max 319 y.max 239 
    # h5_file_path = "/home/handsomexd/EventAD/data/detector/zurich_city_13_b/events/left/events_2x.h5"
    read_h5_file(h5_file_path)