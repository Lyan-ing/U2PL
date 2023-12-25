import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch
# from common_tools import set_seed
def test(
        model,
        data_root,
        label_root,
        save_root,
):
    import cv2
    from torchvision import transforms
    # 加载模型
    os.makedirs(save_root, exist_ok=True)
    model.eval()

    # 预处理图像
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 如果有GPU可用，则将模型移到GPU上
    if torch.cuda.is_available():
        model.to('cuda')

    # 遍历文件夹中的所有图像
    # 创建一个颜色图，以便于可视化
    # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    # colors = torch.as_tensor([i for i in range(3)])[:, None] * palette
    # colors = (colors % 255).numpy().astype("uint8")
    # 定义三原色
    red = np.array([255, 255, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])

    # 创建一个1x3的图像，并填充三原色
    colors = np.array([red, green, blue])
    for filename in os.listdir(data_root):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 加载图像
            image = cv2.imread(os.path.join(data_root, filename))
            label = cv2.imread(os.path.join(label_root, filename))
            vis_label = deepcopy(label)
            vis_label[label > 1] = 2
            vis_label = vis_label[:, :, 0]

            # 预处理图像
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

            # 如果有GPU可用，则将输入张量移到GPU上
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')

            # 进行预测
            with torch.no_grad():
                output = model(input_batch)['pred'][0]
            output_predictions = output.argmax(0)

            # 将预测结果映射到颜色图上
            output_predictions = output_predictions.byte().cpu().numpy()
            output_vis = colors[output_predictions]
            # ori_vis = colors[image]
            label_vis = colors[vis_label]
            label_vis[label == 14] = 0
            output_vis[label == 14] = 0
            merge = np.concatenate((image, label_vis, output_vis), axis=1)

            # 保存结果
            cv2.imwrite(os.path.join(save_root, filename), merge)