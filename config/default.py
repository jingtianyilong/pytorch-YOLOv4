import os
from yacs.config import CfgNode ad CN

_C = CN()

_C.use_darknet_cfg = True
_C.cfgfile = os.path.join(_BASE_DIR, 'cfg', 'yolov4.cfg')

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
_C.max_batches = 500500
_C.steps = [400000, 450000]
_C.policy = _C.steps
_C.scales = .1, .1

_C.cutmix = 0
_C.mosaic = 1

_C.letter_box = 0
_C.jitter = 0.2
_C.classes = 80
_C.track = 0
_C.w = _C.width
_C.h = _C.height
_C.flip = 1
_C.blur = 0
_C.gaussian = 0
_C.boxes = 60  # box num
_C.TRAIN_EPOCHS = 300
_C.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')
_C.val_label = os.path.join(_BASE_DIR, 'data' ,'val.txt')
_C.TRAIN_OPTIMIZER = 'adam'

if _C.mosaic and _C.cutmix:
    _C.mixup = 4
elif _C.cutmix:
    _C.mixup = 2
elif _C.mosaic:
    _C.mixup = 3
    
_C.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
_C.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

_C.iou_type = 'iou'  # 'giou', 'diou', 'ciou'

_C.keep_checkpoint_max = 10

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)

    # if args.modelDir:
    #     cfg.OUTPUT_DIR = args.modelDir

    if args.log_dir:
        cfg.LOG_PATH = args.log_dir

    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)