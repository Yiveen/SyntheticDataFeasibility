import os
import shutil

# 定义图片存储的根目录
root_dir = '/shared-network/yliu/projects/disef/images/'
new_root_dir = '/shared-network/yliu/projects/disef/fine-tune/data/OxfordPets'
# 获取所有图片文件的列表
image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

# 遍历所有图片文件
for image_file in image_files:
    # 提取类别名称，假设类别名称和文件名之间用下划线分隔
    class_name = '_'.join(image_file.split('_')[:-1])
    # 创建类别文件夹的路径
    class_dir = os.path.join(new_root_dir, class_name)

    # 如果类别文件夹不存在，则创建它
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # 源文件路径
    source = os.path.join(root_dir, image_file)

    # 目标文件路径
    destination = os.path.join(class_dir, image_file)

    # 移动文件到相应的类别文件夹
    shutil.copy2(source, destination)

print("文件整理完成！")