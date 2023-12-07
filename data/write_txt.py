# --coding:utf-8--
import os, glob
import random

def copy_first_100_lines(input_file, output_file):
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        for i, line in enumerate(in_file):
            if i < 20000:
                out_file.write(line)
            else:
                break

# 使用函数



split_rate = 0.8
unlabel_root = ''

root = r'E:\python\ZEV\U2PL\data\crop5'
save_path= root
# copy_first_100_lines(os.path.join(save_path, 'unlabel.txt'), os.path.join(save_path, 'unlabel1.txt'))
data_names =  os.listdir(os.path.join(root, "jpg"))
# unlabel_data_names = os.listdir(os.path.join(root, "image_A"))
# data_names = [i for i in data_names]
random.shuffle(data_names)
# random.shuffle(unlabel_data_names)
split_index = int(len(data_names) * split_rate)
train_lines = data_names[:split_index]
val_lines = data_names[split_index:]
# with open(os.path.join(save_path, 'unlabel.txt'), 'w') as f:
#     f.write('\n'.join(unlabel_data_names))

with open(os.path.join(save_path, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_lines))

with open(os.path.join(save_path, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_lines))