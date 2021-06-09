import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
from copy import deepcopy

import numpy as np
import pdb

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from visual_utils.visualize_utils import visualize_points_and_boxes


def parse_config():
    parser = argparse.ArgumentParser(description='argparser')
    # Mendatory Input Arguments
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--dir_data', type=str, default='/mnt/mnt/sdd/ysshin/nuscenes')

    # Radar Configuration
    parser.add_argument('--modality', type=str, default='radar')
    parser.add_argument('--max_sweeps', type=int, default=10)
    parser.add_argument('--sweep_version', type=str, default='version4')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    flags = parser.parse_args()
    if flags.modality == 'lidar':
        flags.cfg_file = os.path.join(os.getcwd(), 'cfgs', 'nuscenes_models', 'cbgs_pp_multihead.yaml')
    elif flags.modality == 'radar':
        flags.cfg_file = os.path.join(os.getcwd(), 'cfgs', 'nuscenes_models', 'cbgs_pp_radar_multihead.yaml')
    else:
        raise ValueError('[ERROR SPALab] Modality %s not understood'%flags.modality)

    cfg_from_yaml_file(flags.cfg_file, cfg)
    cfg.TAG = Path(flags.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(flags.cfg_file.split('/')[1:-1]) 

    if flags.set_cfgs is not None:
        cfg_from_list(flags.set_cfgs, cfg)

    return flags, cfg

def main():
    flags, cfg = parse_config()

    # Radar CFG Modification
    cfg.DATA_CONFIG.MODALITY = flags.modality
    if flags.modality == 'radar':
        # CFG radar cfgs
        cfg.DATA_CONFIG.MODALITY = flags.modality
        cfg.DATA_CONFIG.MAX_SWEEPS = flags.max_sweeps
        cfg.DATA_CONFIG.SWEEP_VERSION = flags.sweep_version

        train_view = '360' # 'front', '360'
        test_view = '360' # 'front', '360
        cfg.DATA_CONFIG.TRAIN_VIEW = train_view
        cfg.DATA_CONFIG.TEST_VIEW = test_view

        max_sweeps = flags.max_sweeps
        sweep_version = flags.sweep_version
        # CFG INFO_PATH
        dir_info_train = f'radar_infos_{sweep_version}_{max_sweeps}sweeps_train.pkl'
        dir_info_val = f'radar_infos_{sweep_version}_{max_sweeps}sweeps_val.pkl'
        cfg.DATA_CONFIG.INFO_PATH = {'train': [dir_info_train], 'test': [dir_info_val]}
        # CFG DB_INFO_PATH
        dir_dbinfo = f'radar_dbinfos_{sweep_version}_{max_sweeps}sweeps.pkl'
        cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH = [dir_dbinfo]
        PREPARE = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['PREPARE']
        filter_num = 1

        # CFG POINT_FEATURE_ENCODING
        if sweep_version in ['version3', 'version4', 'version5', 'version6']:
            feature_list = ['x', 'y', 'z', 'rcs', 'vx', 'vy', 'timestamp']
            cfg.DATA_CONFIG.POINT_FEATURE_ENCODING['used_feature_list'] = feature_list
            cfg.DATA_CONFIG.POINT_FEATURE_ENCODING['src_feature_list'] = feature_list
            cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['NUM_POINT_FEATURES'] = 7
        if train_view == 'front' and flags.modality == 'radar':
            cfg.DATA_CONFIG.POINT_CLOUD_RANGE = [-40, 0, -5.0, 40, 70.4, 3.0]
            cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[1]['ALONG_AXIS_LIST'] = ['y']

    if flags.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % flags.launcher)(
            flags.tcp_port, flags.local_rank, backend='nccl'
        )
        dist_train = True

    if not dist_train:
        os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%flags.gpu

    if flags.batch_size is None:
        flags.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert flags.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        flags.batch_size = flags.batch_size // total_gpus

    flags.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if flags.epochs is None else flags.epochs

    if flags.fix_random_seed:
        common_utils.set_random_seed(666)

    try:
        sweep_version = cfg.DATA_CONFIG.SWEEP_VERSION
    except:
        sweep_version = 'None'
    optimizer = cfg.OPTIMIZATION.OPTIMIZER
    LR = cfg.OPTIMIZATION.LR
    LR_DECAY = cfg.OPTIMIZATION.LR_DECAY
    EPOCHS = cfg.OPTIMIZATION.NUM_EPOCHS

    exp_name = 'GrifNet'
    nms = cfg.MODEL.POST_PROCESSING.NMS_CONFIG.NMS_THRESH
    #output_name = f'Modality_{cfg.DATA_CONFIG.MODALITY}_{sweep_version}_{optimizer}_LR_{LR}_DECAY_{LR_DECAY}_{EPOCHS}epochs'
    #! Only Car 
    output_name = f'Only_car_360degree_10sweeps_version4_densehead'
    output_dir = Path('/mnt/mnt/sdd/ysshin/nuscenes') / 'results' / exp_name /output_name
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * flags.batch_size))
    for key, val in vars(flags).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (flags.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=flags.batch_size,
        dist=dist_train, 
        workers=flags.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=flags.merge_all_iters_to_one_epoch,
        total_epochs=flags.epochs
    )

    ##! Dataset Statistics Analysis
    #import matplotlib.pyplot as plt
    #data_itr = iter(train_loader)

    #features = ['x', 'y', 'z', 'RCS', 'vx', 'vy']
    #features_dict = {}
    #for feature_idx in range(len(features)):
    #    features[features[feature_idx]] = []

    #num_data = 12*len(train_loader)
    #for batch_idx in range(len(train_loader)):
    #    batch_data = data_itr.next()
    #    pts = batch_data['points']
    #    for feature_idx in range(len(features)):
    #        features_dict[features[feature_idx]].extend(pts[:, feature_idx+1])
    #pdb.set_trace()

    #test_cfg = deepcopy(cfg)
    #if cfg.DATA_CONFIG.MODALITY == 'radar' and cfg.DATA_CONFIG.TEST_VIEW == 'front':
    #    test_cfg.DATA_CONFIG.POINT_CLUD_RANGE = [-40, 0, -5.0, 40, 70.4, 3.0]
    #    test_cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[1]['ALONG_AXIS_LIST'] = ['y']
    #if cfg.DATA_CONFIG.MODALITY == 'radar' and cfg.DATA_CONFIG.TEST_VIEW == '360':
    #    test_cfg.DATA_CONFIG.POINT_CLUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    #    test_cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[1]['ALONG_AXIS_LIST'] = ['x', 'y']

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=Path('/mnt/mnt/sdd/ysshin/nuscenes/v1.0-trainval'), verbose=True)

    ###! Visualize Input data
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib as mpl
    import pickle
    dir_figs = 'figs'
    if not os.path.exists(dir_figs):
        os.mkdir(dir_figs)
    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=flags.batch_size,
        dist=dist_train, 
        workers=flags.workers,
        logger=logger,
        training=False,
        merge_all_iters_to_one_epoch=flags.merge_all_iters_to_one_epoch,
        total_epochs=flags.epochs
    )

    class_names = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer']
    num_boxes = {}
    dir_figs = 'figs'
    dir_result1 = os.path.join('/mnt', 'mnt', 'sdd', 'ysshin', 'nuscenes', 'results', 'GrifNet',
                              'Only_car_360degree_10sweeps_version4/eval/eval_all_default/default/epoch_100/val/test_result.pkl')
    with open(dir_result, 'rb') as f:
        results1 = pickle.load(f)

    for class_name in class_names:
        num_boxes[class_name] = 0

    for i in range(test_set.__len__()):
        data = test_set.__getitem__(i)
        if data['gt_boxes'].shape[0] != 0:
            result1 = results1[i]
            visualize_points_and_boxes(
                                       dir_fig=os.path.join(dir_figs, '%d.png'%i),
                                       pts=data['points'],
                                       pred_boxes1=result['boxes_lidar'],
                                       score1=result1['score'],
                                       gt_boxes1=data['gt_boxes'],

                                       pred_boxes2=result['boxes_lidar'],
                                       score2=result2['score'],
                                       gt_boxes2=data['gt_boxes'],

                                       score_thresh=0.2)
            pdb.set_trace()

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)

    if flags.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if flags.pretrained_model is not None:
        model.load_params_from_file(filename=flags.pretrained_model, to_cpu=dist, logger=logger)

    if flags.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(flags.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=flags.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, flags.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=flags.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=flags.ckpt_save_interval,
        max_ckpt_save_num=flags.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=flags.merge_all_iters_to_one_epoch,
        logger=logger,
        cfg=cfg,
        flags=flags,
        dist_train=dist_train,
        output_dir=output_dir,
        eval_output_dir=eval_output_dir,
        nusc=nusc
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, flags.extra_tag))

if __name__ == '__main__':
    main()
