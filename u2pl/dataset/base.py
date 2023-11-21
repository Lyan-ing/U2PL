import logging

from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, d_list, data_type, **kwargs):
        # parse the input list
        self.parse_input_list(d_list, data_type, **kwargs)

    def parse_input_list(self, d_list, data_type='costum', max_sample=-1, start_idx=-1, end_idx=-1):
        logger = logging.getLogger("global")
        assert isinstance(d_list, str)
        if "cityscapes" in d_list:
            self.list_sample = [
                [
                    line.strip(),
                    "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
                ]
                for line in open(d_list, "r")
            ]
        elif "pascal" in d_list or "VOC" in d_list:
            self.list_sample = [
                [
                    "JPEGImages/{}.jpg".format(line.strip()),
                    "SegmentationClass/{}.png".format(line.strip()),
                ]
                for line in open(d_list, "r")
            ]
        elif data_type == 'costum':  # 自定義數據集的處理：不固定尾綴
            if self.mode == 'label':
                self.list_sample = [
                    [
                        "jpg/{}".format(line.strip()),
                        "anno/{}.png".format(line.strip().split('.')[0]),
                    ]
                    for line in open(d_list, "r")
                ]
            elif self.mode == 'unlabel':
                self.list_sample = [
                    [
                        "image_A/{}".format(line.strip()),
                    ]
                    for line in open(d_list, "r")
                ]
        else:
            error_info = "unknown dataset!"
            raise error_info

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        logger.info("# samples: {}".format(self.num_sample))

    def img_loader(self, path, mode):
        with open(path, "rb") as f:
            img = Image.open(f)
            # if mode=='L':
            #     return img
            return img.convert(mode)

    def __len__(self):
        return self.num_sample
