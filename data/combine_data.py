import os
import shutil

# 源文件夹路径
source_folder = r"D:\DATA\DQ2"
dir_list = os.listdir(source_folder)
# 目标文件夹路径
target_folder = r"D:\DATA\DQ2"
ann_path = os.path.join(target_folder, "anno")
jpg_path = os.path.join(target_folder, "jpg")
os.makedirs(ann_path, exist_ok=True)
os.makedirs(jpg_path, exist_ok=True)
# 遍历源文件夹下的子文件夹
for subfolder in dir_list:
    subfolder_path = os.path.join(source_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue

    # 获取ann文件夹和jpg文件夹的路径
    ann_folder = os.path.join(subfolder_path, "anno")
    jpg_folder = os.path.join(subfolder_path, "jpg")


    # 遍历ann文件夹中的文件
    for file_name in os.listdir(ann_folder):
        if not file_name.endswith(".png"):
            print("-----------------------------")
            continue

        # 构建新的文件名，加上原子文件夹的名字前缀
        new_file_name = subfolder + "_" + file_name

        # 源文件路径和目标文件路径
        src_file = os.path.join(ann_folder, file_name)
        dst_file = os.path.join(ann_path, new_file_name)

        # 复制文件到目标文件夹
        shutil.copy(src_file, dst_file)
        shutil.copy(src_file.replace("anno", "jpg").replace("png", "jpg"), dst_file.replace("anno", "jpg").replace("png", "jpg"))  # anno-->jpg png-->jpg

    # # 遍历jpg文件夹中的文件
    # for file_name in os.listdir(jpg_folder):
    #     if not file_name.endswith(".jpg"):
    #         continue
    #
    #     # 构建新的文件名，加上原子文件夹的名字前缀
    #     new_file_name = subfolder + "_" + file_name
    #
    #     # 源文件路径和目标文件路径
    #     src_file = os.path.join(jpg_folder, file_name)
    #     dst_file = os.path.join(target_folder, "jpg", new_file_name)
    #
    #     # 复制文件到目标文件夹
    #     shutil.copy(src_file, dst_file)