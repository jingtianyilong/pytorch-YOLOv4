import os
from yacs.config import CfgNode as CN

_C = CN()

_C.use_darknet_cfg = False
_C.cfgfile = "experiment/yolov4.cfg" 
_C.dataset_dir = "/data"
_C.SEED = 358
_C.pretrained = 'yolov4.conv.137.pth'
_C.keep_checkpoint_max = 10
_C.batch = 64
_C.subdivisions = 16
_C.width = 608
_C.height = 608
_C.channels = 3
_C.momentum = 0.949
_C.decay = 0.0005
_C.angle = 0
_C.saturation = 1.5
_C.exposure = 1.5
_C.hue = .1

_C.learning_rate = 0.00261
_C.burn_in = 1000

_C.cutmix = 0
_C.mosaic = 1

_C.letter_box = 0
_C.jitter = 0.2
_C.classes = 7 
_C.track = 0
_C.w = _C.width
_C.h = _C.height
_C.flip = 1
_C.blur = 0
_C.gaussian = 0
_C.boxes = 60  # box num
_C.TRAIN_EPOCHS = 80
_C.train_label = "/data/anyverse_train.txt"
_C.val_label = "/data/anyverse_val.txt"
_C.TRAIN_OPTIMIZER = 'adam'
_C.namesfile = "data/yolo.names"
_C.anchors = None
    
_C.checkpoints = 'checkpoints'

_C.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

_C.keep_checkpoint_max = 10

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    print(args.config_file)
    if cfg.mosaic and cfg.cutmix:
        cfg.mixup = 4
    elif cfg.cutmix:
        cfg.mixup = 2
    elif cfg.mosaic:
        cfg.mixup = 3
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
