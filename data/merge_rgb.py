import glob
import os
import os.path as op

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np


# 读取RGB波段数据
def merge_rgb(a):
    # global max_r, max_b, max_g, min_r, min_g, min_b
    # global num
    # print()
    if os.path.exists(op.join(img_path, "jpg", a.split("\\")[-1])):
        # num+=1
        return 0
    # a = r"E:\python\ZEV\Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection\Data\Test\test4_band1\band1_oSV1-03_20210809_L1B0001352451_6042300610390028_01-MUX.tif"
    with rasterio.open(a) as gt:
        gt = gt.read(1)
        if gt.max() <10:
            return 0
    with rasterio.open(a.replace("gt_", "red_")) as red:

        red_band = red.read(1) // 255
        red_profile = red.profile
        transform = red.transform
        # m_r = red_band.max()
        # mi_r = red_band.min()
        # if m_r>max_r:
        #     max_r = m_r
        # if mi_r < min_r:
        #     min_r = mi_r

    with rasterio.open(a.replace("gt_", "green_")) as green:
        green_band = green.read(1) // 255
        # m_g = green_band.max()
        # mi_g = green_band.min()
        # if m_g > max_g:
        #     max_g = m_g
        # if mi_g < min_g:
        #     min_g = mi_g
    with rasterio.open(a.replace("gt_", "blue_")) as blue:
        blue_band = blue.read(1) // 255
        # m_b = blue_band.max()
        # mi_b= blue_band.min()
        # if m_b > max_b:
        #     max_b = m_b
        # if mi_b < min_b:
        #     min_b = mi_b


    #     unique = np.unique(gt)
    #     [label.add(i) for i in unique]

    # 合并RGB波段数据
    rgb = np.dstack((red_band, green_band, blue_band))
    red_profile.update(count=3)
    # rasterio.Affine(transform.a, transform.b, transform.c, transform.d, transform.e * -1, transform.f * -1)red_profile.data['transform'].e
    red_profile.update(dtype='uint8')
    # red_profile.data['transform'].e *= -1
    # red_profile.data['transform'].f *= -1
    new_transform = rasterio.Affine(transform.a, transform.b, transform.c, transform.d, -transform.e, -transform.f)
    red_profile.update(transform=new_transform)
    # 显示合并后的RGB图像
    # show(rgb.transpose(2, 0, 1))
    # 保存合并后的RGB图像
    with rasterio.open(op.join(img_path, "jpg", a.split("\\")[-1]), 'w', **red_profile) as dst:
        dst.write(rgb.transpose(2, 0, 1))

def main():
    # import csv
    # with open(csv_file, 'r') as f:
    #     reader = csv.reader(f)
    #     next(reader)  # Skip the header row.
    #     data_list = [row[0] for row in reader]
    data_list = glob.glob(r"E:\dataset\cloud_95\95-cloud_training_only_additional_to38-cloud\train_gt_additional_to38cloud\*.tif")
    for i, data in enumerate(data_list):
        merge_rgb(data)
    # print(max_r, max_b, max_g, min_r, min_g, min_b)
    # print(num)
    print()
if __name__ =="__main__":

    img_path = r"E:\dataset\cloud_95\95-cloud_training_only_additional_to38-cloud"

    label = set()
    # max_i = 0
    # global max_r, max_b, max_g, min_r, min_g, min_b
    # max_r = 0
    # max_g = 0
    # max_b = 0
    # min_r = 100000
    # min_g = 100000
    # min_b = 100000
    # num = 0
    csv_file = r"E:\dataset\cloud_95\95-cloud_training_only_additional_to38-cloud\training_patches_95-cloud_nonempty.csv"
    main()

