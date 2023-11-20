# --coding:utf-8--
import os
import random
split_rate = 0.8
unlabel_root = ''

root = r'E:\python\ZEV\U2PL\data\st\train'
save_path= root
data_names = os.listdir(os.path.join(root, "jpg"))
unlabel_data_names = os.listdir(os.path.join(root, "image_A"))
# data_names = [i for i in data_names]
random.shuffle(data_names)
random.shuffle(unlabel_data_names)
split_index = int(len(data_names) * split_rate)
train_lines = data_names[:split_index]
val_lines = data_names[split_index:]
with open(os.path.join(save_path, 'unlabel.txt'), 'w') as f:
    f.write('\n'.join(unlabel_data_names))

with open(os.path.join(save_path, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_lines))

with open(os.path.join(save_path, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_lines))