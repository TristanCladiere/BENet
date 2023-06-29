from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from test_in_train_mt import test_in_train_mt

import _init_paths

from models.BENet import get_pose_net
from config import cfg_mt
from config import update_config_mt
from core.loss_mt import MultiLossFactory
from core.trainer_mt import do_train
from dataset import make_dataloader
from utils.utils import create_logger
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import setup_logger
from dataset import make_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--test',
                        help='test step. Default=0 means that the model is never tested',
                        default=0,
                        type=int)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config_mt(cfg_mt, args)

    cfg_mt.defrost()
    cfg_mt.RANK = args.rank
    cfg_mt.freeze()

    unused_params = True

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg_mt, args.cfg, 'train'
    )
    os.makedirs(os.path.join(tb_log_dir, 'train'))
    if cfg_mt.DATASET.WITH_VAL:
        os.makedirs(os.path.join(tb_log_dir, 'val'))
    if args.test:
        os.makedirs(os.path.join(tb_log_dir, 'test'))
    logger.info(pprint.pformat(args))
    logger.info(cfg_mt)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or cfg_mt.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg_mt.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir, tb_log_dir, unused_params)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg_mt.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
            tb_log_dir,
            unused_params
        )


def main_worker(
        gpu, ngpus_per_node, args, final_output_dir, tb_log_dir, unused_params
):
    # cudnn related setting
    cudnn.benchmark = cfg_mt.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg_mt.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg_mt.CUDNN.ENABLED

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if cfg_mt.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg_mt.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    update_config_mt(cfg_mt, args)

    # setup logger
    logger, _ = setup_logger(final_output_dir, args.rank, 'train')

    model = get_pose_net(cfg_mt, cfg_mt.MODEL, is_train=True)

    # copy model file
    if not cfg_mt.MULTIPROCESSING_DISTRIBUTED or (
            cfg_mt.MULTIPROCESSING_DISTRIBUTED
            and args.rank % ngpus_per_node == 0
    ):
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../lib/models', cfg_mt.MODEL.NAME + '.py'),
            final_output_dir
        )
        writer_dict = {
            'train_writer': SummaryWriter(log_dir=os.path.join(tb_log_dir, 'train')),
            'global_steps': 0,
        }
        if cfg_mt.DATASET.WITH_VAL:
            writer_dict['val_writer'] = SummaryWriter(log_dir=os.path.join(tb_log_dir, 'val'))
        if args.test:
            writer_dict['test_writer'] = SummaryWriter(log_dir=os.path.join(tb_log_dir, 'test'))

        logger.info(f"\nNumber of parameters: {sum(p.numel() for p in model.parameters()):_}\n")
    else:
        writer_dict = None

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=unused_params
            )
        else:
            model.cuda()

    elif args.gpu is not None:
        torch.cuda.device(torch.cuda.current_device())
        model = model.cuda(torch.cuda.current_device())
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function
    loss_factory = MultiLossFactory(cfg_mt).cuda()

    # Data loading code
    data_loaders = {}
    train_loader = make_dataloader(
        cfg_mt, dataset_type='train', distributed=args.distributed, pre_transforms=cfg_mt.DATASET.PRE_TRANSFORMS
    )
    logger.info(train_loader.dataset)
    data_loaders['train'] = train_loader
    if cfg_mt.DATASET.WITH_VAL:
        val_loader = make_dataloader(
            cfg_mt, dataset_type='val', distributed=args.distributed, pre_transforms=cfg_mt.DATASET.PRE_TRANSFORMS
        )
        logger.info(val_loader.dataset)
        data_loaders['val'] = val_loader
    if args.test:
        test_data_loader, test_dataset = make_test_dataloader(cfg_mt)

    best_perf = -1
    best_epoch = -1
    last_epoch = -1
    optimizer = get_optimizer(cfg_mt, model)

    begin_epoch = cfg_mt.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')
    if cfg_mt.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_perf = checkpoint['best_perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

        if writer_dict is not None:
            writer_dict["global_steps"] = last_epoch

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg_mt.TRAIN.LR_STEP, cfg_mt.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg_mt.TRAIN.END_EPOCH):
        # train one epoch
        mean_val_loss = do_train(cfg_mt, model, data_loaders, loss_factory, optimizer, epoch, writer_dict,
                                 val=cfg_mt.DATASET.WITH_VAL, val_step=cfg_mt.DATASET.VAL_STEP)

        if not cfg_mt.MULTIPROCESSING_DISTRIBUTED or (
                cfg_mt.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):

            # In PyTorch 1.1.0 and later, you should call `lr_scheduler.step()` after `optimizer.step()`.
            if not mean_val_loss:
                lr_scheduler.step()

            if cfg_mt.DATASET.WITH_VAL:
                if epoch == 0 and mean_val_loss:
                    if mean_val_loss > 0:
                        best_perf = mean_val_loss
                    else:
                        best_perf = 10e5
                    best_epoch = epoch
                if mean_val_loss and mean_val_loss > 0:
                    perf_indicator = mean_val_loss
                    if perf_indicator <= best_perf:
                        best_perf = perf_indicator
                        best_epoch = epoch
                        best_model = True
                    else:
                        best_model = False
                else:
                    best_model = False

            else:
                perf_indicator = epoch
                if perf_indicator >= best_perf:
                    best_perf = perf_indicator
                    best_model = True
                else:
                    best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch,
                'best_epoch': best_epoch,
                'model': cfg_mt.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'best_perf': best_perf,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

            if args.test and (epoch % args.test == 0 or epoch == cfg_mt.TRAIN.END_EPOCH - 1):
                name_values, results_emo = test_in_train_mt(cfg_mt, model, logger, final_output_dir,
                                                            test_data_loader, test_dataset)

                if name_values is not None:
                    AP = name_values['AP'] * 100
                    writer_dict['test_writer'].add_scalar(f'AP_detection', AP, writer_dict['global_steps'])

                if results_emo is not None:
                    for key, val in results_emo.items():
                        if key in ["ind2thr", "ind2cat"]:
                            continue
                        if val is not None:
                            try:
                                writer_dict["test_writer"].add_scalar(key, val.mean(), writer_dict["global_steps"])
                            except:
                                logger.info(f"\n{key}, {val}")

                logger.info(f"Done\n")

    if not cfg_mt.MULTIPROCESSING_DISTRIBUTED or (
            cfg_mt.MULTIPROCESSING_DISTRIBUTED
            and args.rank == 0):
        final_model_state_file = os.path.join(
            final_output_dir, 'final_state{}.pth.tar'.format(gpu)
        )

        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
        writer_dict['train_writer'].close()
        if cfg_mt.DATASET.WITH_VAL:
            writer_dict['val_writer'].close()
        if args.test:
            writer_dict['test_writer'].close()


if __name__ == '__main__':
    main()
