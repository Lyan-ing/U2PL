import copy
import math
import os
import os.path
import random

import loguru
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from osgeo import gdal
# import augmentation as psp_trsforms
from . import augmentation as psp_trsform
# from base import BaseDataset as BaseDatasets
from .base import BaseDataset


class voc_dset(BaseDataset):
    def __init__(
            self, data_root, data_list, data_type, trs_form, seed=0, n_sup=10582, split="val", mode='label'
    ):
        self.mode = mode
        super(voc_dset, self).__init__(data_list, data_type)
        self.data_root = data_root
        # self.mode = mode
        self.transform = trs_form
        random.seed(seed)
        if split == "train" and len(self.list_sample) > n_sup:
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif split == "train" and len(self.list_sample) < n_sup:
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

        del self.list_sample  # del掉原始list，減少無用數據

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])

        image = self.img_loader(image_path, "RGB")
        if self.mode == 'label':  # 無標簽數據生成全0mask
            label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
            label = Image.open(label_path)
        else:
            label = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


class tif_dset(BaseDataset):
    def __init__(
            self, data_root, data_list, data_type, trs_form, seed=0, n_sup=10582, split="val", mode='label'
    ):
        self.mode = mode
        super(tif_dset, self).__init__(data_list, data_type)
        self.data_root = data_root
        # self.mode = mode
        self.transform = trs_form
        random.seed(seed)
        if split == "train" and len(self.list_sample) > n_sup:
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif split == "train" and len(self.list_sample) < n_sup:
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

        del self.list_sample  # del掉原始list，減少無用數據

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        image_tif = gdal.Open(image_path).ReadAsArray()
        # scale to 255
        image_rgb = image_tif[:3]
        image_rgb = (image_rgb - image_rgb.min()) / (image_rgb.max() - image_rgb.min()) *255
        image_nir = image_tif[3]
        image_nir = (image_nir - image_nir.min()) / (image_nir.max() - image_nir.min()) * 255
        image = Image.fromarray(image_rgb.transpose(1, 2, 0), mode="RGB")
        img_nir = Image.fromarray(image_nir, mode="L")
        if self.mode == 'label':  # 無標簽數據生成全0mask
            label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
            label = Image.open(label_path)
        else:
            label = Image.fromarray(np.zeros((image.size[1], image.size[0]), dtype=np.uint8))
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg):
    # if cfg["saver"]["task_name"]=="guangxi":
    #     from . import augmentation_label_convert as psp_trsform
    trs_form = []
    ignore_label = cfg["ignore_label"]
    mean, std = None, None
    if cfg.get("mean"):
        mean = cfg["mean"]
    if cfg.get("std"):
        std = cfg["std"]
    if cfg.get("ColorJitter", False) and cfg["ColorJitter"]:
        trs_form.append(psp_trsform.RandomColorJitter())
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
        trs_form.append(psp_trsform.RandomVerticalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    return psp_trsform.Compose(trs_form)


def build_vocloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], os.path.join(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"])),
                    trs_form=trs_form, seed=seed, n_sup=n_sup)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_costum_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    if split == "val":
        batch_size = int(batch_size * 1.5)
    # n_sup = cfg.get("n_sup", 10582)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"]), 'costum', trs_form=trs_form,
                    seed=seed)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_voc_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 10582 - cfg.get("n_sup", 10582)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"]), trs_form=trs_form, seed=seed,
                    n_sup=n_sup, split=split)

    if split == "val":
        # build sampler
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = os.path.join(cfg["data_root"], cfg["data_list"]).replace("labeled.txt", "unlabeled.txt")
        dset_unsup = voc_dset(
            cfg["data_root"], data_list_unsup, trs_form=trs_form_unsup, seed=seed, n_sup=n_sup, split=split,
            mode='unlabel'
        )

        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup


def build_costum_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)

    data_list = os.path.join(cfg["data_root"], cfg["data_list"])
    n_sup = 0
    if split != 'val':
        # build sampler for unlabeled set
        data_list_unsup = os.path.join(cfg["data_root"], cfg["unlabel_data_list"])
        with open(data_list_unsup, 'r') as f:
            lines = f.readlines()
            line_count_unsup = len(lines)
        with open(data_list, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)
        n_sup = line_count if line_count_unsup < line_count else line_count_unsup
        loguru.logger.info(f"训练样本数量为 2 * {n_sup}")
        # .replace("labeled.txt", "unlabeled.txt")
        dset_unsup = voc_dset(
            cfg["data_root"], data_list_unsup, 'costum', trs_form_unsup, seed, n_sup, split=split, mode="unlabel"
        )
        # n_sup = len(dset_unsup.list_sample_new)  # 計算一下無標記圖像的數量，將標籤圖像重複採樣至於無標記圖像相同

    dset = voc_dset(cfg["data_root"], data_list, 'costum', trs_form, seed, n_sup,
                    split)

    if split == "val":
        # build sampler
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup


def main():
    import torch
    import torchvision.transforms as f
    import yaml
    from PIL import Image
    cfg = yaml.load(open(r'E:\python\ZEV\U2PL\config\config_cos_sup_unet.yaml', "r"), Loader=yaml.Loader)["dataset"]
    cfg["data_root"] = r"E:\python\ZEV\U2PL\cloud"
    cfg.update(cfg.get('train', {}))
    trs_form = build_transfrom(cfg)
    dset = tif_dset(cfg["data_root"], os.path.join(cfg["data_root"], os.path.join(cfg["data_root"], cfg["data_list"])),
                    'costum', trs_form=trs_form, seed=0, mode='unlabel')
    # sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=1,
        num_workers=0,
        # sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    for i, batch in enumerate(loader):
        a = batch
        print()
        pass

    # img_path = tif_dset()
    # img = Image.open(img_path)
    # trans = f.ColorJitter(brightness=0.8)
    # image = trans(img)
    # image.show()


if __name__ == "__main__":
    main()
