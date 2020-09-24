
import os, argparse, torch, cv2, time
from utils.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
from config import cfg
from config import update_config
from models import Yolov4
from train import get_anchors, getArgs

def get_first_10_imgs(config):
    fh = open(config.val_label)
    image_paths = []
    for line in fh.readlines():
        line = line.rstrip().split()
        if len(image_paths) < 10:
            if len(line) > 1:
                image_paths.append(os.path.join("/data",line[0]))
        else:
            break
    return image_paths


def demo(model, config, anchors):
    if torch.cuda.is_available():
        model.cuda()
    class_names = load_class_names(config.namesfile)
    fh = open(config.val_label)
    print("load val_label: {}".format(config.val_label))
    i = 0
    image_paths = get_first_10_imgs(config)
    print("Got images: ")
    print(image_paths)
    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        sized = cv2.resize(img, (config.width, config.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        start = time.time()
        boxes = do_detect(model, sized, 0.4, 0.6, torch.cuda.is_available())
        finish = time.time()
        print('{}: Predicted in {:.04f} seconds.'.format(image_path, (finish - start)))

        plot_boxes_cv2(img, boxes[0], savename=os.path.join("demo_img","predictions_{}.jpg".format(i)), class_names=class_names)
    
if __name__ == "__main__":
    args = getArgs()
    # import config file and save it to log
    update_config(cfg, args)
    print("getting anchors...")
    anchors = get_anchors(cfg)
    log_dir = os.path.join("log",os.path.basename(args.config_file)[:-5])
    if not args.load:
        latest_weight = os.path.join(log_dir,"checkpoints",os.listdir(os.path.join(log_dir,"checkpoints"))[0])
    elif os.path.exists():
        latest_weight = args.load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.use_darknet_cfg:
        eval_model = Darknet(cfg.cfgfile)
    else:
        eval_model = Yolov4(anchors, yolov4conv137weight=None, n_classes=cfg.classes,inference=True)
    os.makedirs("demo_img",exist_ok=True)
    print("load pretrained weight")
    pretrained_dict = torch.load(latest_weight, map_location=torch.device('cuda'))
    if torch.cuda.device_count() > 1:
        eval_model = torch.nn.DataParallel(eval_model)
    eval_model.to(device=device)
    eval_model.load_state_dict(pretrained_dict)

    demo(model=eval_model,
            config=cfg,
            anchors = anchors)

