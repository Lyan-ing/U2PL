# import rasterio
# from rasterio.windows import Window
#
# # 打开原始tif图像
# with rasterio.open(r"E:\python\ZEV\U2PL\cloud\unlabel\cloud.tif") as src:
#     # 获取图像的宽度、高度和波段数
#     width, height = src.width, src.height
#     bands = src.count
#
#     # 设置每个小块的大小
#     block_size = 512
#
#     # 计算需要切分的行数和列数
#     num_blocks_x = (width + block_size - 1) // block_size
#     num_blocks_y = (height + block_size - 1) // block_size
#
#     # 遍历每个小块并保存为tif文件
#     for i in range(num_blocks_y):
#         for j in range(num_blocks_x):
#             # 计算当前小块的左上角坐标和右下角坐标
#             left = j * block_size
#             top = i * block_size
#             right = min((j + 1) * block_size, width)
#             bottom = min((i + 1) * block_size, height)
#
#             # 读取当前小块的数据
#             window = Window(left, top, right - left, bottom - top)
#             block = src.read(window=window)
#             # 获取小块的转换信息
#             # transform = rasterio.windows.transform(window, src.transform)
#             transform = src.window_transform(window)
#             transform = rasterio.Affine(transform.a, transform.b, transform.c, transform.d, transform.e * -1, transform.f * -1)
#
#             # 生成输出文件名
#             output_filename = f"E:/python/ZEV/U2PL/cloud/unlabel/output_{i}_{j}.tif"
#
#             # 保存当前小块为tif文件
#             with rasterio.open(output_filename, "w", driver="GTiff", height=block.shape[1], width=block.shape[2], count=bands, dtype=block.dtype, crs=src.crs, transform=transform) as dst:
#                 dst.write(block)


from osgeo import gdal
import numpy as np

def split_tiff(input_file, output_prefix, tile_size):
    # 打开输入的四波段TIFF图像
    dataset = gdal.Open(input_file)
    if dataset is None:
        print("无法打开输入文件。")
        return

    # 获取图像的宽度和高度
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 计算图像的列数和行数
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # 切分图像并保存小块
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # 计算当前小块的起始像素位置
            x_offset = i * tile_size
            y_offset = j * tile_size

            # 读取小块数据
            tile_data = dataset.ReadAsArray(x_offset, y_offset, tile_size, tile_size)

            # 创建输出文件名
            output_file = f"{output_prefix}_{i}_{j}.tiff"

            # 创建输出驱动程序和数据集
            driver = gdal.GetDriverByName("GTiff")
            output_dataset = driver.Create(output_file, tile_size, tile_size, 4, gdal.GDT_Float32)

            # 将小块数据写入输出数据集的波段
            for band in range(4):
                output_band = output_dataset.GetRasterBand(band + 1)
                output_band.WriteArray(tile_data[band])

            # 设置地理转换信息和投影信息
            output_dataset.SetGeoTransform((x_offset, tile_size, 0, y_offset, 0, -tile_size))
            output_dataset.SetProjection(dataset.GetProjection())

            # 关闭输出数据集
            output_dataset = None

    # 关闭输入数据集
    dataset = None

# 示例用法
input_file = r'GF7_DLC_W11.4_N11.4_20220622_L1A0000852849-BWDMUX.tiff'  # 输入的四波段TIFF图像文件名
output_prefix = r'E:\python\ZEV\U2PL\cloud\unlabel\GF7'  # 输出小块的文件名前缀
tile_size = 512  # 小块的尺寸（宽度和高度）

split_tiff(input_file, output_prefix, tile_size)