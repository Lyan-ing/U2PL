import argparse
import json
import logging
import os
import os.path as osp
import pprint
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import get_criterion, MulticlassDiceLoss
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    load_state,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--resume", action='store_true')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=10081)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--pretrain_path", default='', type=str)
logger = init_log("global", logging.INFO)
logger.propagate = 0


def main():
    from loguru import logger as loger
    global args, cfg
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    # loger.info(args.port)

    cfg["exp_path"] = cfg["saver"]["exp_path"]
    cfg["log_path"] = cfg["saver"]["log_path"]
    cfg["task_id"] = osp.join(cfg["dataset"]["type"], cfg["saver"]["task_name"], cfg["saver"]["task_idx"])
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["task_id"])
    cfg["log_save_path"] = osp.join(cfg["log_path"], cfg["task_id"])

    # cfg["resume"] = True if args.resume else False
    wether_resume = os.path.exists(osp.join(cfg["save_path"], "ckpt.pth"))
    if not wether_resume:
        args.resume = False

    if args.pretrain_path != '':
        cfg["net"]["pretrain"] = args.pretrain_path

    cudnn.enabled = True
    cudnn.benchmark = True

    rank, word_size = setup_distributed(port=args.port, backend='gloo')

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_logger = SummaryWriter(osp.join(cfg["log_save_path"], current_time))
    else:
        tb_logger = None

    if args.seed is not None:
        loger.info(f"==> set random seed to {args.seed}")
        set_random_seed(args.seed)

    if rank == 0:
        os.makedirs(cfg["save_path"], exist_ok=True)
        os.makedirs(cfg["log_save_path"], exist_ok=True)
    model_path = None

    if cfg['net']['base_model'] == "naic":
        # model_path = cfg['trainer']['naic_path']
        from u2pl.naic.deeplabv3_plus import DeepLabv3_plus
        import torch.nn as nn
        in_channels = cfg["net"]["in_channels"] if cfg["net"].get("in_channels") else 3
        model = DeepLabv3_plus(in_channels=in_channels, num_classes=cfg["net"]["num_classes"],
                               backend=cfg["net"]["backbone"],
                               os=16, pretrained=cfg["net"]["pretrain"] == '', norm_layer=nn.BatchNorm2d)
        # if not args.resume:
        #     loger.info(f"Model from NAIC DeeplabV3Plus from {model_path}..........")
        #     load_state(model_path, model, key="naic")
        # model.load_state_dict(torch.load(model_path,  map_location='cpu'), strict=False)
        modules_back = [model.backend]
        modules_head = [model.aspp_pooling, model.cbr_low, model.cbr_last]
    elif cfg['net']["base_model"] == "unet":
        from u2pl.unet.unet import Unet
        in_channels = cfg["net"]["in_channels"] if cfg["net"].get("in_channels") else 3
        model = Unet(num_classes=cfg["net"]["num_classes"], pretrained=False, backbone=cfg["net"]["backbone"],
                     in_channels=in_channels)
        modules_back = [model.resnet]
        modules_head = [model.up_concat1, model.up_concat2, model.up_concat3, model.up_concat4, model.final]
        # model.freeze_backbone()
    else:
        # Create network.
        model = ModelBuilder(cfg["net"])
        modules_back = [model.encoder]
        if cfg["net"].get("aux_loss", False):
            modules_head = [model.auxor, model.decoder]
        else:
            modules_head = [model.decoder]

        if cfg["net"].get("sync_bn", True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion = get_criterion(cfg)
    dice_loss = None
    if cfg["criterion"].get("dice_loss", False) and cfg["criterion"]["dice_loss"]:
        dice_loss = MulticlassDiceLoss(num_cls=cfg["net"]["num_classes"])

    train_loader_sup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10  # if "pascal" in cfg["dataset"]["type"] else 1  #  这里要修改

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    best_prec = 0
    last_epoch = 0
    # if not model_path:
    # auto_resume > pretrain
    if args.resume:
        lastest_model = osp.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:  # resume
            loger.info(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )

    elif cfg["net"].get("pretrain", False):
        if cfg["net"]["base_model"] == "naic":
            load_state(cfg["net"]["pretrain"], model, key="naic", type="need_module")
        elif cfg["net"]["base_model"] == "unet":
            load_state(cfg["net"]["pretrain"], model, key="naic", type="need_module")
        else:
            load_state(cfg["net"]["pretrain"], model, key="model_state")

    optimizer_old = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_old, start_epoch=last_epoch
    )
    with open(osp.join(cfg["save_path"], 'config.yaml'), 'w', encoding='utf-8') as yf:
        yaml.dump(cfg, yf)

    # Start to train model
    if not cfg['dataset'].get('category', False):
        CLASSES_need = {
            0: "background",
            1: "building",
            2: "grass",
            3: "tree",
            4: "water",
            5: "tea"
        }
    else:
        if isinstance(cfg['dataset']['category'], dict):
            CLASSES_need = {
                0: "background"}
            CLASSES_need.update(cfg['dataset']['category'])
        else:
            loger.error("Need category")
    # CLASSES_need = {
    #     0: "laingshi",
    #     1: "laohuang",
    #     2: "feiliang",
    #     # 3: "tree",
    #     # 4: "water",
    #     # 5: "tea"
    # }

    # test(model, osp.join(cfg["dataset"]["val"]["data_root"], "jpg"), osp.join(cfg["dataset"]["val"]["data_root"], "anno"), osp.join(cfg["dataset"]["val"]["data_root"], "pred3"))
    now = datetime.now()
    # 格式化日期和时间
    datetime_begin = now.strftime("%Y-%m-%d %H:%M:%S")
    if not args.resume:
        with open(osp.join(cfg["save_path"], 'log.json'), 'w', encoding='utf-8') as log:
            json.dump(f"==================BEGIN the Training at {datetime_begin}=================", log)
            log.write('\n')
            log.flush()
    else:
        with open(osp.join(cfg["save_path"], 'log.json'), 'a', encoding='utf-8') as log:
            json.dump(f"=================RESUME the Training at {datetime_begin}=================", log)
            log.write('\n')
            log.flush()
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # prec, iou_cls = validate(model, val_loader, epoch)
        # Training
        train(
            model,
            optimizer,
            lr_scheduler,
            criterion,
            dice_loss,
            train_loader_sup,
            epoch,
            tb_logger,
        )

        # Validation and store checkpoint
        prec, iou_cls = validate(model, val_loader, epoch)

        if rank == 0:
            if rank == 0:
                for i, iou in enumerate(iou_cls):
                    logger.info(" * class [{}] IoU {:.2f}".format(CLASSES_need[i], iou * 100))  #
                logger.info(" * epoch {} mIoU {:.2f}".format(epoch, prec * 100))

            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou": best_prec,
            }

            if prec > best_prec:
                best_prec = prec
                state["best_miou"] = prec
                torch.save(
                    state, osp.join(cfg["save_path"], "ckpt_best.pth")
                )
                # 只保留权重，不保留优化器等
                torch.save(model.module.state_dict(), osp.join(cfg["save_path"], "best_epoch_weights.pth"))

                # write the model_param.json
                now = datetime.now()

                # 格式化日期和时间
                datetime_end = now.strftime("%Y-%m-%d %H:%M:%S")
                model_type = cfg["net"]["base_model"].capitalize()
                task_type = cfg["dataset"]["type"].capitalize()
                model_param_dict = {"modelName": f"{model_type}-{task_type}-01",
                                    "baseModel": f"{model_type}",
                                    "backbone": cfg["net"]["backbone"],
                                    "modelType": "landcover-classfication",
                                    "modelVersion": "1.0.0",
                                    "modelDescription": "模型说明",
                                    "category": list(CLASSES_need.values())[1:],
                                    "Accuray": round(best_prec * 100, 2),
                                    "author": "...",
                                    "create-time": datetime_end,
                                    # "end-time": datetime_end
                                    }

                with open(os.path.join(cfg["save_path"], 'model_param.json'), 'w', encoding='utf-8') as ff:
                    json.dump(model_param_dict, ff, indent=4, ensure_ascii=False)

            torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))
            logger.info(
                "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                    best_prec * 100
                )
            )
            tb_logger.add_scalar("AmIoU", prec, epoch)
            for i, iou in enumerate(iou_cls):
                tb_logger.add_scalar(f"IoU/{CLASSES_need[i]}", iou, epoch)
    with open(osp.join(cfg["save_path"], 'log.json'), 'a', encoding='utf-8') as log:
        json.dump(f"==================END the Training at {datetime_end}=================", log)
        log.write('\n')
        log.flush()


def train(  # 蓝，青，绿
        model,
        optimizer,
        lr_scheduler,
        criterion,
        dice_loss,
        data_loader,
        epoch,
        tb_logger,
):
    model.train()

    data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    rank, world_size = dist.get_rank(), dist.get_world_size()

    losses = AverageMeter(20)
    if dice_loss:
        loss_criterion = AverageMeter(20)
        losses_dice = AverageMeter(20)
    data_times = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)

    batch_end = time.time()
    with open(osp.join(cfg["save_path"], 'log.json'), 'a', encoding='utf-8') as log:
        # json.dump(cfg, log)
        all_epoch = cfg["trainer"]["epochs"]
        len_iter = len(data_loader)
        all_iter = cfg["trainer"]["epochs"] * len_iter
        for step in range(len_iter):
            batch_start = time.time()
            data_times.update(batch_start - batch_end)

            i_iter = epoch * len_iter + step
            lr = lr_scheduler.get_lr()
            learning_rates.update(lr[0])
            lr_scheduler.step()

            image, label = next(data_loader_iter)
            batch_size, h, w = label.size()
            image, label = image.cuda(), label.cuda()
            outs = model(image)
            pred = outs["pred"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                loss = criterion([pred, aux], label)
            else:
                loss = criterion(pred, label)
            # use_dice = True
            if dice_loss is not None:
                loss_criterion.update(loss.item())
                loss_dice = dice_loss(pred, label)
                losses_dice.update(loss_dice.item())
                loss += loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # gather all loss from different gpus
            reduced_loss = loss.clone().detach()
            dist.all_reduce(reduced_loss)
            losses.update(reduced_loss.item())

            batch_end = time.time()
            batch_times.update(batch_end - batch_start)

            if i_iter % 20 == 0 and rank == 0:
                if dice_loss:
                    logger.info(
                        "Epoch [{}/{}]\t"
                        "Iter [{}/{}]\t"
                        "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "CE Loss {ce.val:.4f} ({ce.avg:.4f})\t"
                        "Dice Loss {dice.val:.4f} ({dice.avg:.4f})\t"
                        "LR {lr.val:.5f} ({lr.avg:.5f})\t".format(
                            epoch,
                            all_epoch,
                            step,
                            len_iter,
                            batch_time=batch_times,
                            loss=losses,
                            ce=loss_criterion,
                            dice=losses_dice,
                            lr=learning_rates,
                        )
                    )
                    dict_json = {"FLAG": "[Train]", "Epoch": f"[{epoch}/{all_epoch}]", "Iter": f"[{step}/{len_iter}]",
                                 "Loss": f"{losses.val:.4f} ({losses.avg:.4f}) ",
                                 "CE Loss": f"{loss_criterion.val:.4f} ({loss_criterion.avg:.4f})",
                                 "Dice Loss": f"{losses_dice.val:.4f} ({losses_dice.avg:.4f})",
                                 "LR": f"{learning_rates.val:.5f} ({learning_rates.avg:.5f})",
                                 "Time": f"{batch_times.val:.2f} ({batch_times.avg:.2f})"}
                else:
                    logger.info(
                        "Epoch [{}/{}]\t"
                        "Iter [{}/{}]\t"
                        "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "LR {lr.val:.5f} ({lr.avg:.5f})\t".format(
                            epoch,
                            all_epoch,
                            step,
                            len_iter,
                            batch_time=batch_times,
                            loss=losses,
                            lr=learning_rates,
                        )
                    )
                    dict_json = {"FLAG": "[Train]", "Epoch": f"[{epoch}/{all_epoch}]", "Iter": f"[{step}/{len_iter}]",
                                 "Loss": f"{losses.val:.4f} ({losses.avg:.4f}) ",
                                 "LR": f"{learning_rates.val:.5f} ({learning_rates.avg:.5f})",
                                 "Time": f"{batch_times.val:.2f} ({batch_times.avg:.2f})"}
                json.dump(dict_json, log)
                # "FLAG: [Train]    Epoch [{}/{}]    Iter [{}/{}]    "
                # "Data {data_time.val:.2f} ({data_time.avg:.2f})    "
                # "Time {batch_time.val:.2f} ({batch_time.avg:.2f})    "
                # "Loss {loss.val:.4f} ({loss.avg:.4f})    "
                # "LR {lr.val:.5f} ({lr.avg:.5f})".format(
                #     epoch,
                #     cfg["trainer"]["epochs"],
                #     i_iter,
                #     cfg["trainer"]["epochs"] * len(data_loader),
                #     data_time=data_times,
                #     batch_time=batch_times,
                #     loss=losses,
                #     lr=learning_rates,
                # )
                log.write('\n')
                log.flush()
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("loss/Loss", losses.avg, i_iter)
                if dice_loss:
                    tb_logger.add_scalar("loss/CE Loss", loss_criterion.avg, i_iter)
                    tb_logger.add_scalar("loss/Dice Loss", losses_dice.avg, i_iter)


def validate(
        model,
        data_loader,
        epoch,
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    with open(osp.join(cfg["save_path"], 'log.json'), 'a', encoding='utf-8') as log:
        for batch in tqdm(data_loader):
            images, labels = batch
            images = images.cuda()
            labels = labels.long().cuda()
            batch_size, h, w = labels.shape

            with torch.no_grad():
                outs = model(images)

            # get the output produced by model_teacher
            output = outs["pred"]
            output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
            output = output.data.max(1)[1].cpu().numpy()
            target_origin = labels.cpu().numpy()

            # start to calculate miou
            intersection, union, target = intersectionAndUnion(
                output, target_origin, num_classes, ignore_label
            )

            # gather all validation information
            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        total_epoch = cfg["trainer"]["epochs"]
        dict_json = {"FLAG": "[Val]", "Epoch": f"[{epoch}/{total_epoch}]", "mIoU": f"[{round(mIoU * 100, 2)}]"}
        json.dump(dict_json, log)
        log.write('\n')
        log.flush()

        # if rank == 0:
        #     for i, iou in enumerate(iou_class):
        #         logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        #     logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

        return mIoU, iou_class


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


if __name__ == "__main__":
    main()
