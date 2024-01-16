import glob, os

import loguru
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import rasterio


def draw_hist(arr, x, title, bins=10, save_path=None):
    # 计算每个bins的数量
    # counts, bins = np.histogram(arr, bins=bins)
    # # 绘制直方图
    # plt.hist(arr, bins=bins, edgecolor='black')

    # 创建一个直方图
    n, bins, patches = plt.hist(arr, bins=bins, edgecolor='black')

    # 设置标题和标签
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel('num')
    # 在每个bin上添加文本
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2., patches[i].get_height(),
                 '%d' % int(patches[i].get_height()), ha='center', va='bottom')
    # 在每个柱子上显示具体数量
    # for i in range(len(bins) - 1):
    #     plt.text(bins[i], counts[i] + 2, str(counts[i]), color='black', fontweight='bold')

    # 显示图形
    plt.savefig(os.path.join(save_path, title + '.png'))
    # plt.show()
    plt.close()


def mask_static(mask_path, save_path):
    masks = glob.glob(mask_path + "/*.png")
    mask_list = []  # 洪水占比
    for one_mask in masks:
        mask_arr = cv2.imread(one_mask)[..., 0]
        flood_rate = (mask_arr > 0).sum() / sum_pixels
        mask_list.append([os.path.basename(one_mask), flood_rate])

    # 创建DataFrame
    df = pd.DataFrame(mask_list, columns=['name', 'flood_rate'])

    draw_hist(df['flood_rate'], 'rate', 'flood_rate')
    df.to_excel(save_path, index=False)


"""
统计光学数据的数据分布情况
1：mask分布,写入表格，绘制图形
2：输入图像的分布：
    1): rgb...等通道的取值范围，每个通道的最大取值范围
"""
mask_path = r"E:\dataset\FLOOD\Track2\train\labels"
sum_pixels = 128 ** 2
mask_save_path = 'flood_rate.xlsx'
# mask_static(mask_path, mask_save_path)  # 1:

img_path = r"E:\dataset\FLOOD\Track2\val\images"

list_name = ['img_name', 'Coastal Aerosol max', 'Coastal Aerosol min', 'Blue max',
             'Blue min', 'Green max', 'Green min', 'Red max', 'Red min',
             'NIR max', 'NIR min', 'SWIR1 max', 'SWIR1 min', 'SWIR2 max',
             'SWIR2 min', 'QA band max', 'QA band min', 'Merit DEM max',
             'Merit DEM min', 'Copernicus DEM max', 'Copernicus DEM min',
             'ESA World Cover Map max', 'ESA World Cover Map min',
             'Water occ pro mean', 'Water occ pro mean thr0', 'Water occ pro mean thr10',
             'NDWI thr 0 mean', 'NDWI thr 0.2 mean', 'Flood rate'
             ]


def img_static(img_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    img_list = glob.glob(img_path + "/*.tif")
    img_static_list = []
    for img_path in img_list:
        img_static_list.append(static_and_vis(img_path, if_vis=True, save_path=save_path))  # 获取每张图的统计数据
    # compute the static
    statics = ['static']
    for i in range(1, len(img_static_list[0])):
        one_st = [item[i] for item in img_static_list]
        draw_hist(one_st, 'value', list_name[i], 10, save_path)  # 绘制直方图
        if i >= len(img_static_list[0]) - 6:
            mins = np.mean(one_st)
        elif i % 2 == 0:
            mins = min(one_st)
        else:
            mins = max(one_st)
        statics.append(mins)
    img_static_list.append(statics)
    df = pd.DataFrame(img_static_list, columns=list_name)
    df.to_excel(os.path.join(save_path, 'img.xlsx'), index=False)
    loguru.logger.info("end")


def compute_NDWI(path_input_img):
    """
    The Normalized Difference Water Index (NDWI) is a common spectral index used to highlight open water features in a
    satellite image, allowing a water area to stand out against the other types of ground cover.

    It is defined as follows:

    NDWI = (Green – NIR)/(Green + NIR)


    The NDWI values correspond to the following ranges:

    - [0,2 ; 1] –> Water surface,
    - [0.0 ; 0,2] –> Flooding, humidity,
    - [-0,3 ; 0.0] –> Moderate drought, non-aqueous surfaces,
    - [-1 ; -0.3] –> Drought, non-aqueous surfaces
    """
    with rasterio.open(path_input_img) as dataset:
        green = dataset.read(3)
        nir = dataset.read(5)

        # Conversion from uint16 to float
        GREEN = np.float32(green)  # R = np.float32(R_U)
        NIR = np.float32(nir)  # NIR = np.float32(NIR_U)

        # print('np.amin(R) = ', np.amin(GREEN), ' ; np.amax(R) = ', np.amax(NIR))
        # print('np.amin(NIR) = ', np.amin(NIR), ' ; np.amax(NIR) = ', np.amax(NIR))

        # # if the denominateur = 0, we don't calculate the NDVI, otherwise it's ok
        NDWI = np.divide((GREEN - NIR), (GREEN + NIR), where=(NIR + GREEN) != 0)

        # print("type(NDWI[0,0]) = ", type(NDWI[0, 0]))
        # print('np.amin(NDWI) = ', np.amin(NDWI))
        # print('np.amax(NDWI) = ', np.amax(NDWI))

    return NDWI


def threshold(input_NDWI, min_limit):
    thresh = np.where(input_NDWI >= min_limit, 1, 0)
    return thresh


def static_and_vis(img_file, if_vis=True, save_path=None):
    """
    展示img每个通道的情况，
    """
    # for i in range(len(img_files)):
    path_msk = img_file.replace('images', 'labels').replace(".tif", '.png')
    path_img = img_file
    # print("path_msk = ", path_msk)
    # print("path_img = ", path_img)
    # Water mask (.png file)
    img1 = rasterio.open(path_img).read(1)
    if os.path.exists(path_msk):
        mask = cv2.imread(path_msk, cv2.IMREAD_UNCHANGED)
    else:
        mask = np.zeros_like(img1)
    flood_rate = (mask > 0).sum() / sum_pixels
    # Spectral bands inside each .tif file
    img2 = rasterio.open(path_img).read(2)
    img3 = rasterio.open(path_img).read(3)
    img4 = rasterio.open(path_img).read(4)
    img5 = rasterio.open(path_img).read(5)
    img6 = rasterio.open(path_img).read(6)
    img7 = rasterio.open(path_img).read(7)
    img8 = rasterio.open(path_img).read(8)
    img9 = rasterio.open(path_img).read(9)
    img10 = rasterio.open(path_img).read(10)
    img11 = rasterio.open(path_img).read(11)
    img12 = rasterio.open(path_img).read(12)

    GREEN = np.float32(img3)  # R = np.float32(R_U)
    NIR = np.float32(img5)  # NIR = np.float32(NIR_U)
    NDWI = np.divide((GREEN - NIR), (GREEN + NIR), where=(NIR + GREEN) != 0)
    # NDWI = compute_NDWI(path_img)
    statics = [os.path.basename(path_img)]
    for img in [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11]:
        # img = rasterio.open(path_img).read(i + 1)
        channel_min = img.min()
        channel_max = img.max()
        statics.append(channel_max)
        statics.append(channel_min)
    water_occ_p = img12.mean()
    water_occ_thr10 = (img12 > 10)
    water_occ_thr10_p = water_occ_thr10.mean()
    water_occ_thr0 = (img12 > 0)
    water_occ_thr0_p = water_occ_thr0.mean()
    thr_0 = threshold(NDWI, 0.)
    thr_02 = threshold(NDWI, 0.2)
    statics.append(water_occ_p)  # mean(Water occurence probability)
    statics.append(water_occ_thr0_p)  #
    statics.append(water_occ_thr10_p)  #
    statics.append(round(thr_0.mean(), 4))
    statics.append(round(thr_02.mean(), 4))
    statics.append(round(flood_rate, 4))

    if if_vis:
        fig, ax = plt.subplots(4, 5, figsize=(24, 20))
        ax[0, 0].imshow(img1, cmap='gray')
        ax[0, 1].imshow(img2, cmap='gray')
        ax[0, 2].imshow(img3, cmap='gray')
        ax[0, 3].imshow(img4, cmap='gray')
        ax[0, 4].imshow(img5, cmap='gray')
        ax[1, 0].imshow(img6, cmap='gray')
        ax[1, 1].imshow(img7, cmap='gray')
        ax[1, 2].imshow(img8, cmap='gray')
        ax[1, 3].imshow(img9, cmap='gray')
        ax[1, 4].imshow(img10, cmap='gray')
        ax[2, 0].imshow(img11, cmap='gray')
        ax[2, 1].imshow(img12, cmap='gray')
        ax[2, 2].imshow(mask, cmap='gray')
        ax[2, 3].imshow(NDWI, cmap='gray')
        ax[2, 4].imshow(thr_0, cmap='gray')
        ax[3, 1].imshow(water_occ_thr0, cmap='gray')
        ax[3, 2].imshow(water_occ_thr10, cmap='gray')
        ax[3, 3].imshow(thr_02, cmap='gray')

        # Set a title for the figures
        ax[0, 0].set_title('Coastal Aerosol')
        ax[0, 1].set_title('Blue')
        ax[0, 2].set_title('Green')
        ax[0, 3].set_title('Red')
        ax[0, 4].set_title('NIR')
        ax[1, 0].set_title('SWIR1')
        ax[1, 1].set_title('SWIR2')
        ax[1, 2].set_title('QA band')
        ax[1, 3].set_title('Merit DEM')
        ax[1, 4].set_title('Copernicus DEM')
        ax[2, 0].set_title('ESA World Cover Map')
        ax[2, 1].set_title('Water occurence probability')
        ax[2, 2].set_title('Water mask')
        ax[2, 3].set_title('NDWI')
        ax[2, 4].set_title('Thresholded NDWI, min_limit=0.0')
        ax[3, 1].set_title('Thresholded Water occ pro, min_limit=0')
        ax[3, 2].set_title('Thresholded Water occ pro, min_limit=10')
        ax[3, 3].set_title('Thresholded NDWI, min_limit=0.2')
        os.makedirs(os.path.join(save_path, 'img'), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'img', os.path.basename(path_msk)))
        plt.close()
    # plt.show()

    return statics


img_static(img_path, 'static_val')
