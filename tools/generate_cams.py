import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml
from utils.torch_utils import get_device, load_from_checkpoint
from data import get_dataloaders
from tqdm import tqdm
import glob


def generate_cams_byproto(dataloaders, config, device, steps):
    ckpt_step = glob.glob(os.path.join(config['CHECKPOINT']["save_path"], '*0' + '.pth'))
    for index in range(len(ckpt_step)):
        ckpt_step_path = os.path.basename(ckpt_step[index])
        step = os.path.splitext(ckpt_step_path)[0]
        if int(step) not in steps:
            continue
        cam_path = os.path.join(config["CHECKPOINT"]["save_cams_path"], 'cams_'+step)
        if not os.path.exists(cam_path):
            os.makedirs(cam_path)
        net = get_model(config, device)
        if ckpt_step[index]:
            load_from_checkpoint(net, ckpt_step[index], partial_restore=False)
        if len(config['local_device_ids']) > 1:
            net = nn.DataParallel(net, device_ids=config['local_device_ids'])
        net.to(device)
        net.eval()
        with torch.no_grad():
            for step, sample in tqdm(enumerate(dataloaders['train']), total=len(dataloaders['train'])):
                cls_label_gt = sample['cls_labels'].numpy()
                temporal_cam = net(x=sample['inputs'].to(torch.float32).to(device), cls_label_gt=sample['cls_labels'].cuda(), generate_cam=True)                
                temporal_cam = temporal_cam.cpu().numpy() * cls_label_gt.reshape(cls_label_gt.shape[0], cls_label_gt.shape[1], 1, 1)                
                for b, img_path in enumerate(sample["img_path"]):
                    total_dict = {}
                    temporal_dict = {}
                    for cls_id in range(cls_label_gt.shape[1]):
                        if cls_label_gt[b,cls_id] > 0:
                            norm_temporal_cam = temporal_cam[b,cls_id,:]
                            norm_temporal_cam = (norm_temporal_cam - norm_temporal_cam.min()) / (norm_temporal_cam.max() - norm_temporal_cam.min() + 1e-8)
                            temporal_dict[cls_id] = norm_temporal_cam                
                    total_dict['temporal_cam'] = temporal_dict
                    img_name = os.path.basename(img_path)
                    img_name, _ = os.path.splitext(img_name)
                    np.save(os.path.join(cam_path, img_name+".npy"), total_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', default='./configs/PASTIS24/Generate_and_Eval.yaml', help='configuration (.yaml) file to use')
    parser.add_argument('--device', nargs='+', default=[0,1,2,3,4,5,6,7], type=int,
                        help='gpu ids to use')
    parser.add_argument("--steps", nargs='+', default=[4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000], type=int)
    parser.add_argument('--save_path', type=str, help='')
    parser.add_argument('--dataset_path', type=str, help='')
    parser.add_argument('--save_cams_path', type=str, help='')
                        
    args = parser.parse_args()
    config_file = args.config
    device_ids = args.device
    steps = args.steps

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config["CHECKPOINT"]["save_path"] = args.save_path
    config["DATASETS"]["dataset_path"] = args.dataset_path
    config["CHECKPOINT"]["save_cams_path"] = args.save_cams_path

    dataloaders = get_dataloaders(config)

    if isinstance(steps, int):
        steps = [steps]

    generate_cams_byproto(dataloaders, config, device, steps)
    