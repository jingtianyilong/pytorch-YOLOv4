# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 15:07
@Author        : Tianxiaomo
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import time
import random
import logging
import os, sys, math
import argparse
import pprint
from collections import deque
import datetime
import pdb
import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import Yolo_dataset
from config import cfg
from config import update_config
from models import Yolov4
from tool.darknet2pytorch import Darknet
from utils.yolo_loss import Yolo_loss, bboxes_iou
from utils.utils import init_seed
from tool.tv_reference.utils import collate_fn as val_collate
from tool.tv_reference.coco_utils import convert_to_coco_api
from tool.tv_reference.coco_eval import CocoEvaluator

def collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    if len(bboxes) != 0:
        bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images, bboxes


def train(model, device, config, tb_log_dir, anchors, epochs=5, batch_size=1, save_cp=True, log_step=5, img_scale=0.5):
    train_dataset = Yolo_dataset(config.train_label, config, train=True)
    val_dataset = Yolo_dataset(config.val_label, config, train=False)
    # scaler = torch.cuda.amp.GradScaler()

    n_train = len(train_dataset)
    n_val = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config.batch // config.subdivisions, shuffle=True,
                              num_workers=16, pin_memory=True, drop_last=True, collate_fn=collate)

    val_loader = DataLoader(val_dataset, batch_size=config.batch // config.subdivisions, shuffle=True, num_workers=16,
                            pin_memory=True, drop_last=True, collate_fn=val_collate)

    writer = SummaryWriter(log_dir=tb_log_dir,
                           filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                           comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
    # writer.add_images('legend',
    #                   torch.from_numpy(train_dataset.label2colorlegend2(cfg.DATA_CLASSES).transpose([2, 0, 1])).to(
    #                       device).unsqueeze(0))
    max_itr = epochs * n_train
    # global_step = cfg.TRAIN_MINEPOCH * n_train
    global_step = 0
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {config.batch}
        Subdivisions:    {config.subdivisions}
        Learning rate:   {config.learning_rate}
        Steps:           {max_itr//config.subdivisions}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:     {config.width}
        Optimizer:       {config.TRAIN_OPTIMIZER}
        Dataset classes: {config.classes}
        Train label path:{config.train_label}
        Pretrained:
    ''')
    def burnin_schedule(i):
        # pdb.set_trace()
        if i < n_train//config.batch:
            factor = pow(i / (n_train//config.batch), 4)
        elif i < int(max_itr*0.8//config.batch):
            factor = 1.0
        elif i < int(max_itr*0.9//config.batch):
            factor = 0.1
        else:
            factor = 0.01
        # print(factor)
        return factor
    
    if config.TRAIN_OPTIMIZER.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif config.TRAIN_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.decay,
        )
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule)
    criterion = Yolo_loss(device=device, anchors=anchors, batch=config.batch // config.subdivisions, n_classes=config.classes)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=6, min_lr=1e-7)

    save_prefix = 'Yolov4_epoch'
    saved_models = deque()
    # model.train()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_step = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', ncols=80) as pbar:
            for i, batch in enumerate(train_loader):
                global_step = global_step + 1
                epoch_step = epoch_step + 1
                images = batch[0]
                bboxes = batch[1]

                images = images.to(device=device, dtype=torch.float32)
                bboxes = bboxes.to(device=device)
                # with to
                bboxes_pred = model(images)
                loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = criterion(bboxes_pred, bboxes)
                loss.backward()

                epoch_loss = epoch_loss + loss.item()

                if global_step % config.subdivisions == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                if global_step % (log_step * config.subdivisions) == 0:
                    writer.add_scalar('train/Loss', loss.item(), global_step)
                    writer.add_scalar('train/loss_xy', loss_xy.item(), global_step)
                    writer.add_scalar('train/loss_wh', loss_wh.item(), global_step)
                    # writer.add_scalar('train/loss_obj', loss_obj.item(), global_step)
                    writer.add_scalar('train/loss_cls', loss_cls.item(), global_step)
                    # writer.add_scalar('train/loss_l2', loss_l2.item(), global_step)
                    # writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'loss_xy': loss_xy.item(),
                                        'loss_wh': loss_wh.item(),
                                        # 'loss_obj': loss_obj.item(),
                                        'loss_cls': loss_cls.item(),
                                        # 'loss_l2': loss_l2.item(),
                                        'lr': scheduler.get_last_lr()[0]
                                        })
                    logging.debug('Train step_{:.06f}: loss : {:.06f},loss xy : {:.06f},loss wh : {:.06f},'
                                  'loss obj : {:.06f}，loss cls : {:.06f},loss l2 : {},lr : {:.06f}'
                                  .format(global_step, loss.item(), loss_xy.item(),
                                          loss_wh.item(), loss_obj.item(),
                                          loss_cls.item(), loss_l2.item(),
                                          scheduler.get_last_lr()[0]))
                pbar.update(images.shape[0])

            if cfg.use_darknet_cfg:
                eval_model = Darknet(cfg.cfgfile, inference=True)
            else:
                eval_model = Yolov4(anchors,yolov4conv137weight=cfg.pretrained, n_classes=cfg.classes, inference=True)
            if torch.cuda.device_count() > 1:
                eval_model.load_state_dict(model.module.state_dict())
            else:
                eval_model.load_state_dict(model.state_dict())
            eval_model.to(device)
            evaluator = evaluate(eval_model, val_loader, config, device)
            del eval_model

            stats = evaluator.coco_eval['bbox'].stats
            writer.add_scalar('train/AP', stats[0], global_step)
            writer.add_scalar('train/AP50', stats[1], global_step)
            writer.add_scalar('train/AP75', stats[2], global_step)
            writer.add_scalar('train/AP_small', stats[3], global_step)
            writer.add_scalar('train/AP_medium', stats[4], global_step)
            writer.add_scalar('train/AP_large', stats[5], global_step)
            # writer.add_scalar('train/AR1', stats[6], global_step)
            # writer.add_scalar('train/AR10', stats[7], global_step)
            # writer.add_scalar('train/AR100', stats[8], global_step)
            # writer.add_scalar('train/AR_small', stats[9], global_step)
            # writer.add_scalar('train/AR_medium', stats[10], global_step)
            # writer.add_scalar('train/AR_large', stats[11], global_step)

            if save_cp:
                try:
                    os.makedirs(os.path.join(tb_log_dir,"checkpoints"), exist_ok=True)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                save_path = os.path.join(tb_log_dir,"checkpoints", f'{save_prefix}{epoch + 1}.pth')
                if torch.cuda.device_count() > 1: 
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                logging.info(f'Checkpoint {epoch + 1} saved !')
                saved_models.append(save_path)
                if len(saved_models) > config.keep_checkpoint_max > 0:
                    model_to_remove = saved_models.popleft()
                    try:
                        os.remove(model_to_remove)
                    except:
                        logging.info(f'failed to remove {model_to_remove}')
    writer.close()


@torch.no_grad()
def evaluate(model, data_loader, cfg, device, logger=None, **kwargs):
    """ finished, tested
    """
    # cpu_device = torch.device("cpu")
    model.train(False)
    model.eval()
    # header = 'Test:'

    coco = convert_to_coco_api(data_loader.dataset, bbox_fmt='coco')
    coco_evaluator = CocoEvaluator(coco, iou_types = ["bbox"], bbox_fmt='coco')
    
    for images, targets in tqdm(data_loader,desc="Validation Batch: ", ncols=80):
        model_input = [[cv2.resize(img, (cfg.w, cfg.h))] for img in images]
        model_input = np.concatenate(model_input, axis=0)
        model_input = model_input.transpose(0, 3, 1, 2)
        model_input = torch.from_numpy(model_input).div(255.0)
        model_input = model_input.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(model_input)

        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        # outputs = outputs.cpu().detach().numpy()
        res = {}
        # for img, target, output in zip(images, targets, outputs):
        for img, target, boxes, confs in zip(images, targets, outputs[0], outputs[1]):
            img_height, img_width = img.shape[:2]
            # boxes = output[...,:4].copy()  # output boxes in yolo format
            boxes = boxes.squeeze(2).cpu().detach().numpy()
            boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
            boxes[...,0] = boxes[...,0]*img_width
            boxes[...,1] = boxes[...,1]*img_height
            boxes[...,2] = boxes[...,2]*img_width
            boxes[...,3] = boxes[...,3]*img_height
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # confs = output[...,4:].copy()
            confs = confs.cpu().detach().numpy()
            labels = np.argmax(confs, axis=1).flatten()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            scores = np.max(confs, axis=1).flatten()
            scores = torch.as_tensor(scores, dtype=torch.float32)
            res[target["image_id"].item()] = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    return coco_evaluator


def getArgs():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_file', type=str, default="experiment/demo.yaml", help="yaml configuration file")
    parser.add_argument('--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    args = parser.parse_args()
    return args


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')

def get_anchors(cfg, num_of_anchor=9):
    f = open(cfg.train_label)        
    lines = [line.rstrip('\n') for line in f.readlines()]
    annotation_dims = []
    # pdb.set_trace()
    from PIL import Image
    for line in lines:
        line = line.rstrip().split()
        img = Image.open(os.path.join(cfg.dataset_dir,line[0]))
        img_w,img_h = img.size
        try:
            for obj in line[1:]:
                obj = obj.split(",")
                annotation_dims.append((float(obj[2])/img_w*cfg.width,float(obj[3])/img_h*cfg.height))
        except:
            pass
    annotation_dims = np.array(annotation_dims)
    from sklearn.cluster import KMeans
    kmeans_calc = KMeans(n_clusters=num_of_anchor)
    kmeans_calc.fit(annotation_dims)
    y_kmeans = kmeans_calc.predict(annotation_dims)
    anchor_list = []
    for ind in range(num_of_anchor):
        anchor_list.append(np.mean(annotation_dims[y_kmeans==ind],axis=0))

    return list(map(int,np.concatenate(anchor_list)))

if __name__ == "__main__":
    args = getArgs()
    # import config file and save it to log
    update_config(cfg, args)
    init_seed(cfg.SEED)
    anchors = get_anchors(cfg)
    log_dir = os.path.join("log",os.path.basename(args.config_file)[:-5])
    logging = init_logger(log_dir=log_dir)
    logging.info(pprint.pformat(args))
    logging.info(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if cfg.use_darknet_cfg:
        model = Darknet(cfg.cfgfile)
    else:
        model = Yolov4(anchors, yolov4conv137weight=cfg.pretrained, n_classes=cfg.classes)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=device)

    try:
        train(model=model,
              config=cfg,
              tb_log_dir=log_dir,
              anchors = anchors,
              epochs=cfg.TRAIN_EPOCHS,
              device=device)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
