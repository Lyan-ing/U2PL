import os
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

def cut_georeferenced_image(input_path, output_dir, tile_size):
    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height

        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                window = Window(i, j, min(tile_size, width - i), min(tile_size, height - j))
                transform = src.window_transform(window)
                # transform = rasterio.Affine(transform.a, transform.b, transform.c, transform.d, transform.e * -1, transform.f * -1)
                #
                output_path = os.path.join(output_dir, f'GF7_DLC_W111_{i}_{j}.tiff')

                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    width=window.width,
                    height=window.height,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=transform
                ) as dst:
                    data = src.read(window=window, out_shape=(src.count, window.height, window.width), resampling=Resampling.nearest)
                    dst.write(data)

# 用法示例
input_tif = r'E:\dataset\20231108_影像挑选\GF7_DLC_W11.4_N11.4_20220622_L1A0000852849-BWDMUX_mask.tiff'
output_directory = r'E:\python\ZEV\U2PL\cloud\anno'
tile_size = 512  # 调整为适当的大小


from osgeo import gdal
ds = gdal.Open(input_tif)
geotrans = ds.GetGeoTransform()
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cut_georeferenced_image(input_tif, output_directory, tile_size)
