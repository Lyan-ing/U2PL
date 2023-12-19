import logging

from .cityscapes import build_city_semi_loader, build_cityloader
from .pascal_voc import build_voc_semi_loader, build_vocloader, build_costum_loader, build_costum_semi_loader
from .tif import build_costum_tif_loader
logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    cfg_dataset = cfg["dataset"]

    if cfg_dataset["type"] == "cityscapes_semi":
        train_loader_sup, train_loader_unsup = build_city_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_cityloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "cityscapes":
        train_loader_sup = build_cityloader("train", cfg, seed=seed)
        val_loader = build_cityloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    elif cfg_dataset["type"] == "pascal_semi":
        train_loader_sup, train_loader_unsup = build_voc_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_vocloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader
    # 實現自定義數據集半監督訓練加載模式
    elif cfg_dataset["type"] == "costum_semi":
        train_loader_sup, train_loader_unsup = build_costum_semi_loader(
            "train", cfg, seed=seed
        )
        val_loader = build_costum_loader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, train_loader_unsup, val_loader

    elif cfg_dataset["type"] == "pascal":
        train_loader_sup = build_vocloader("train", cfg, seed=seed)
        val_loader = build_vocloader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    # 實現自定義數據集加載模式
    elif cfg_dataset["type"] == "costum":
        train_loader_sup = build_costum_loader("train", cfg, seed=seed)
        val_loader = build_costum_loader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader

    elif cfg_dataset["type"] == "costum_tif":
        train_loader_sup = build_costum_tif_loader("train", cfg, seed=seed)
        val_loader = build_costum_tif_loader("val", cfg)
        logger.info("Get loader Done...")
        return train_loader_sup, val_loader
    else:
        raise NotImplementedError(
            "dataset type {} is not supported".format(cfg_dataset)
        )
