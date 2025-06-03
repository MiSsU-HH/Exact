import torch
from data.PASTIS24.dataloader import get_dataloader as get_pastis_dataloader
from data.PASTIS24.data_transforms import PASTIS_segmentation_transform
from utils.config_files_utils import get_params_values, read_yaml
import os


def get_dataloaders(config, is_dist=False):

    dataset_path = config['DATASETS']['dataset_path']
    model_config = config['MODEL']
    train_config = config['DATASETS']['train']
    train_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    eval_config  = config['DATASETS']['eval']
    eval_config['bidir_input'] = model_config['architecture'] == "ConvBiRNN"
    dataloaders = {}
    
    # TRAIN data -------------------------------------------------------------------------------------------------------
    train_config['base_dir'] = dataset_path
    train_config['paths'] = os.path.join(dataset_path, 'fold-paths/folds_1_123_paths.csv')

    if 'PASTIS' in train_config['dataset']:

        dataloaders['train'] = get_pastis_dataloader(
            paths_file=train_config['paths'], root_dir=train_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=True),
            batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'],
            pseudo_path=train_config['pseudo_path'], is_dist=is_dist)

    # EVAL data --------------------------------------------------------------------------------------------------------
    eval_config['base_dir'] = dataset_path
    eval_config['paths'] = os.path.join(dataset_path, 'fold-paths/fold_4_paths.csv')

    if 'PASTIS' in eval_config['dataset']:
        dataloaders['eval'] = get_pastis_dataloader(
            paths_file=eval_config['paths'], root_dir=eval_config['base_dir'],
            transform=PASTIS_segmentation_transform(model_config, is_training=False),
            batch_size=eval_config['batch_size'], shuffle=False, num_workers=eval_config['num_workers'],
            pseudo_path=False, is_dist=is_dist)

    return dataloaders


def get_loss_data_input(config):

    def segmentation_ground_truths(sample, device):
        labels = sample['labels'].to(device)
        if 'unk_masks' in sample.keys():
            unk_masks = sample['unk_masks'].to(device)
        else:
            unk_masks = None

        if 'edge_labels' in sample.keys():
            edge_labels = sample['edge_labels'].to(device)
            return labels, edge_labels, unk_masks
        return labels, unk_masks

    def cscl_ground_truths(sample, device, return_masks=False):
        labels = sample['cscl_labels'].to(device)
        if return_masks:
            masks = sample['cscl_labels_mask'].to(device)
            if 'edge_locs' in sample:
                wh, ww = masks.shape[-2:]
                masks = (sample['edge_locs'].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, wh, ww) * \
                         sample['cscl_labels_mask'].to(torch.float32)).to(torch.bool).to(device)
            return labels, masks
        return labels

    loss_fn = config['SOLVER']['loss_function']
    stage = get_params_values(config['MODEL'], 'train_stage', 2)

    if config['MODEL']['architecture'] in ['UNET3Df', 'UNET2D-CLSTM']:
        if stage in [0, 4]:
            if loss_fn in ["binary_cross_entropy", "binary_focal_loss", "contrastive_loss"]:
                if stage == 0:
                    return cscl_ground_truths

            if loss_fn in ["masked_binary_cross_entropy", "masked_binary_focal_loss", "masked_contrastive_loss"]:
                if stage == 0:
                    return lambda sample, device: cscl_ground_truths(sample, device, return_masks=True)

    return segmentation_ground_truths
