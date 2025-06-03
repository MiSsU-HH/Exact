import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params
from data import get_dataloaders
from utils.summaries import write_mean_summaries
import torch.nn.functional as F
from metrics.mean_ap import mAP
from tqdm import tqdm
import logging


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):
    def train_step(net, sample, optimizer, device, abs_step, lambda_1=0.01, lambda_2=0.015):
        optimizer.zero_grad()
        cls_label_gt = sample['cls_labels'].to(device)
        res = net(x=sample['inputs'].to(torch.float32).to(device), cls_label_gt=cls_label_gt, abs_step=abs_step)
        loss_cls = F.multilabel_soft_margin_loss(res['cls_logits'], cls_label_gt)
        loss_aux_cls = (F.multilabel_soft_margin_loss(res['spatial_patch_logits'], cls_label_gt)
                         + F.multilabel_soft_margin_loss(res['temporal_patch_logits'], cls_label_gt)) * 0.5
        loss = loss_cls + loss_aux_cls
        loss_cbl = lambda_1 * res['proto_entropy'].mean() if abs_step >= 4000 else 0
        loss_tap = lambda_2 * torch.sum(torch.abs(res['fusion_cam_refine']-res['fusion_cam']) * cls_label_gt.unsqueeze(2).unsqueeze(2))/ \
            (torch.sum(cls_label_gt)*res['fusion_cam'].shape[2]*res['fusion_cam'].shape[3])
        loss = loss_cls + loss_aux_cls + loss_cbl + loss_tap      
        loss.backward()
        optimizer.step()
        return loss_cls, loss_aux_cls, loss_cbl, loss_tap, loss

    def evaluate(net, evalloader):
        predicted_all = []
        labels_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in tqdm(enumerate(evalloader), total=len(evalloader)):
                res = net(x=sample['inputs'].to(torch.float32).to(device))
                predicted_all.append(torch.sigmoid(res['cls_logits']).cpu().numpy())
                labels_all.append(sample['cls_labels'].cpu().numpy())
        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        eval_results = {}
        mAP_value_macro, mAP_value_micro, ap = mAP(predicted_classes, target_classes)
        eval_results['mAP_macro'] = mAP_value_macro
        eval_results['mAP_micro'] = mAP_value_micro
        for i in range(np.size(ap)):
            eval_results[f"AP_class{i}"] = ap[i]
        return eval_results
    
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_path = config['CHECKPOINT']["save_path"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)
    start_global = 1
    start_epoch = 1
    print("current learn rate: ", lr)
    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)
    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)
    copy_yaml(config)
    if lin_cls:
        print('Train linear classifier only')
        trainable_params = get_net_trainable_params(net)[-2:]
    else:
        print('Train network end-to-end')
        trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    scheduler = build_scheduler(config, optimizer, num_steps_train)
    writer = SummaryWriter(save_path)
    log_file = os.path.join(save_path, 'result.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            loss_cls, loss_aux_cls, loss_cbl, loss_tap, loss = train_step(net, sample, optimizer, device, abs_step)
            if abs_step % train_metrics_steps == 0:
                batch_metrics = {"abs_step":abs_step,"epoch":epoch,"step":step + 1,"loss":loss,"loss_cls":loss_cls,
                "loss_aux_cls":loss_aux_cls,"loss_cbl":loss_cbl,"loss_tap":loss_tap,"lr":optimizer.param_groups[0]["lr"]}
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                logger.info("abs_step: %d, epoch: %d, step: %d, loss: %.7f, loss_cls: %.7f, loss_aux_cls: %.7f, loss_cbl: %.7f, loss_tap: %.7f, lr: %.6f" %
                      (abs_step, epoch, step + 1, loss, loss_cls, loss_aux_cls, loss_cbl, loss_tap, optimizer.param_groups[0]["lr"]))
            if abs_step % eval_steps == 0:  # evaluate model every eval_steps batches
                eval_metrics = evaluate(net, dataloaders['eval'])
                logger.info("-------------------------------------------------------------------------------------------------------")
                metric_info = ''
                for key, value in eval_metrics.items():
                    metric_info += str(key) + ': ' + str(value) + '   '
                logger.info(metric_info)
                logger.info("-------------------------------------------------------------------------------------------------------")
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%s.pth" % (save_path,str(abs_step)))
                else:
                    torch.save(net.state_dict(), "%s/%s.pth" % (save_path,str(abs_step)))
                write_mean_summaries(writer, eval_metrics, abs_step, mode="eval", optimizer=None)
                net.train()
        scheduler.step_update(abs_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', default='./configs/PASTIS24/Exact_cls.yaml', help='configuration (.yaml) file to use')
    parser.add_argument('--device', nargs='+', default=[0,1,2,3,4,5,6,7], type=int, help='gpu ids to use')
    parser.add_argument('--lin', action='store_true', help='train linear classifier only')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    args = parser.parse_args()
    config_file = args.config
    device_ids = args.device
    lin_cls = args.lin
    device = get_device(device_ids, allow_cpu=False)
    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config["CHECKPOINT"]["save_path"] = args.save_path
    config["DATASETS"]["dataset_path"] = args.dataset_path
    dataloaders = get_dataloaders(config)
    net = get_model(config, device)
    train_and_evaluate(net, dataloaders, config, device)
