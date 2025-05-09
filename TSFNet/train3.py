import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling3 import VisionTransformer as ViT_seg
from networks.vit_seg_modeling3 import CONFIGS as CONFIGS_ViT_seg
from trainer3 import trainer_dataset
import os
from networks.vit_seg_modeling_L2HNet3 import L2HNet
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dw', help='experiment_name')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4
                    , help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.0001, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--CNN_width', type=int, default=64, help='L2HNet_width_size, default is 64: light mode. Set to 128: normal mode')
parser.add_argument('--savepath', type=str, default='D:\\wxc\\Paraformer-main\\Paraformer-main\\save')
parser.add_argument('--gpu', default='0', type=str, help='Select GPU number to train' )
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if __name__ == "__main__":
    vit_patches_size=16
    img_size=224
    cudnn.benchmark = True
    cudnn.deterministic = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'dw': {  # default dataset as a example
            'list_dir': 'dataset/ht/list_new3.csv', # The path of the *.csv file
            'num_classes': 7
        }
    }# Create a config to your own dataset here
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    snapshot_path = args.savepath 
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg["ViT-B_16"]
    config_vit.n_classes = args.num_classes
    config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, backbone=L2HNet(width=args.CNN_width),img_size=img_size, num_classes=config_vit.n_classes).cuda()
    net.load_state_dict(torch.load(snapshot))
    trainer_dataset(args, net, snapshot_path)
