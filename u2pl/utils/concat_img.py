from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def concatenate_images(image_paths, direction='horizontal'):
    if isinstance(image_paths[0], str):
        images = [Image.open(i) for i in image_paths]
    else:
        images = image_paths
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_image = Image.new('L', (new_width, new_height))

    x_offset = 0
    y_offset = 0

    for im in images:
        new_image.paste(im, (x_offset, y_offset))
        if direction=='horizontal':
            x_offset += im.size[0]
        else:
            y_offset += im.size[1]

    return new_image

# 使用方法
image_paths = [r'E:\python\ZEV\U2PL\onnx_predict1\pre_0_0.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_20480_0.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_40960_0.png', r"E:\python\ZEV\U2PL\onnx_predict1\pre_61440_0.png"]
image_paths1 = [r'E:\python\ZEV\U2PL\onnx_predict1\pre_0_20480.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_20480_20480.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_40960_20480.png', r"E:\python\ZEV\U2PL\onnx_predict1\pre_61440_20480.png"]
image_paths2 = [r'E:\python\ZEV\U2PL\onnx_predict1\pre_0_40960.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_20480_40960.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_40960_40960.png', r"E:\python\ZEV\U2PL\onnx_predict1\pre_61440_40960.png"]
image_paths3 = [r'E:\python\ZEV\U2PL\onnx_predict1\pre_0_61440.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_20480_61440.png', r'E:\python\ZEV\U2PL\onnx_predict1\pre_40960_61440.png', r"E:\python\ZEV\U2PL\onnx_predict1\pre_61440_61440.png"]
concatenated_image = []
for image_path in [image_paths, image_paths1, image_paths2, image_paths3]:
    concatenated_image.append(concatenate_images(image_path, 'horizontal'))
# for con in concatenated_image:
con = concatenate_images(concatenated_image, direction='re')
con.save('path_to_save_image0.png')