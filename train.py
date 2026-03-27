import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import wandb
from tqdm import tqdm

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "MDE-AdaBins"
enable_logging = False  # renamed from 'logging' to avoid shadowing the built-in module


def is_rank_zero(args):
    return args.rank == 0


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    """Convert a depth tensor or numpy array to a colorized RGB image."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    # Squeeze to 2D (H, W)
    if value.ndim == 4:
        value = value[0, 0]
    elif value.ndim == 3:
        value = value[0]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.0  # Avoid 0-division

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (H x W x 4)
    img = value[:, :, :3]
    return img


def log_images(img, depth, pred, args, step):
    """Log input image, ground truth depth, and prediction to WandB."""
    # img: (C, H, W) tensor on GPU — convert to numpy for wandb
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    else:
        img_np = img

    depth_col = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred_col = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img_np)],
            "GT": [wandb.Image(depth_col)],
            "Prediction": [wandb.Image(pred_col)]
        }, step=step)


def main_worker(gpu, ngpus_per_node, args):
    print("\nControl Entered main_worker function\n")
    args.gpu = gpu

    ###################################### Load model ##############################################
    print("\nBuilding model...\n")
    model = models.UnetAdaptiveBins.build(
        n_bins=args.n_bins,
        min_val=args.min_depth,
        max_val=args.max_depth,
        norm=args.norm
    )
    print(f"Model type: {type(model)}")
    ################################################################################################

    args.multigpu = False

    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(f"GPU: {args.gpu}  Rank: {args.rank}  Batch: {args.batch_size}  Workers: {args.workers}")
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True
        )

    elif args.gpu is not None:
        # Single GPU — no parallelism
        print(f"\nUsing single GPU: {args.gpu}\n")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    else:
        # Use DataParallel across all available GPUs
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    optimizer_state_dict = None

    if args.resume != '':
        if os.path.isfile(args.resume):
            model, optimizer_state_dict, args.epoch = model_io.load_checkpoint(args.resume, model)
            args.last_epoch = args.epoch
            print(f"Loaded checkpoint '{args.resume}' (epoch {args.epoch})")
        else:
            print(f"Warning: No checkpoint found at '{args.resume}', starting from scratch.")

    train(
        model, args,
        epochs=args.epochs,
        lr=args.lr,
        device=args.gpu,
        root=args.root,
        experiment_name=args.name,
        optimizer_state_dict=optimizer_state_dict
    )


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".",
          device=None, optimizer_state_dict=None):
    print("\nControl inside train function\n")
    global PROJECT

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and enable_logging

    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### Losses ##################################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################

    model.train()

    ###################################### Optimizer ###############################################
    if args.same_lr:
        print("Using same LR for all param groups")
        params = model.parameters()
    else:
        print("Using different LR for encoder/decoder")
        m = model.module if args.multigpu else model
        params = [
            {"params": m.get_1x_lr_params(), "lr": lr / 10},
            {"params": m.get_10x_lr_params(), "lr": lr}
        ]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################

    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ##############################################
    # last_epoch handles fast-forwarding the scheduler when resuming — no extra .step() needed
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=args.last_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor
    )
    ################################################################################################

    for epoch in range(args.epoch, epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        ################################# Train loop ###############################################
        if should_log:
            wandb.log({"Epoch": epoch}, step=step)

        train_iter = tqdm(
            enumerate(train_loader),
            desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
            total=len(train_loader)
        ) if is_rank_zero(args) else enumerate(train_loader)

        for i, batch in train_iter:
            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)

            # BUG FIX: use .item() to safely convert a tensor to Python bool
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth'].item():
                    continue

            bin_edges, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.tensor(0.0, device=img.device)

            loss = l_dense + args.w_chamfer * l_chamfer
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # BUG FIX: only log criterion_bins if it is not None
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                if criterion_bins is not None:
                    wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)

            del loss
            step += 1
            scheduler.step()

            ######################################################################################

            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ################################
                model.eval()
                metrics, val_si = validate(
                    args, model, test_loader, criterion_ueff,
                    epoch, epochs, device, should_log, step
                )
                
                # Always print metrics to console regardless of WandB logging
                if should_write:
                    print(f"\nEpoch: {epoch + 1}/{epochs}  Step: {step}  "
                          f"Val Loss: {val_si.get_value():.4f}")
                    print("  Metrics: " +
                          "  ".join(f"{k}: {round(v, 4)}" for k, v in metrics.items()))

                if should_log:
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                    }, step=step)
                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)

                # FIX 1: Save latest checkpoint unconditionally (not just when WandB is on).
                # FIX 2: Save epoch+1 so that resuming starts from the NEXT epoch,
                #         not the already-completed one.
                if should_write:
                    model_io.save_checkpoint(
                        model, optimizer, epoch + 1,
                        f"{experiment_name}_{run_id}_latest.pt",
                        root=os.path.join(root, "checkpoints")
                    )

                if not metrics:
                    print("Warning: metrics dict is empty, skipping best-model checkpoint check.")
                elif metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(
                        model, optimizer, epoch + 1,
                        f"{experiment_name}_{run_id}_best.pt",
                        root=os.path.join(root, "checkpoints")
                    )
                    best_loss = metrics['abs_rel']

                model.train()
                ##################################################################################

    return model


def validate(args, model, test_loader, criterion_ueff, epoch, epochs,
             device='cpu', should_log=False, step=0):
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = utils.RunningAverageDict()
        skipped = 0

        val_iter = (
            enumerate(tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation"))
            if is_rank_zero(args)
            else enumerate(test_loader)
        )

        for i, batch in val_iter:
            # Check has_valid_depth FIRST — before touching depth at all.
            # The dataloader sets depth_gt=False (Python bool) when GT is missing,
            # which collates to a zero tensor. We must skip these early.
            if 'has_valid_depth' in batch:
                hvd = batch['has_valid_depth']
                # Handle both plain Python bool (from dataloader) and tensor
                if isinstance(hvd, torch.Tensor):
                    valid = hvd.item()
                else:
                    valid = bool(hvd)
                if not valid:
                    skipped += 1
                    continue

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)

            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred_np = pred.squeeze().cpu().numpy()
            pred_np[pred_np < args.min_depth_eval] = args.min_depth_eval
            pred_np[pred_np > args.max_depth_eval] = args.max_depth_eval
            pred_np[np.isinf(pred_np)] = args.max_depth_eval
            pred_np[np.isnan(pred_np)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

            # BUG FIX: eval_mask must be defined before it can be AND-ed with valid_mask.
            # Moved valid_mask update INSIDE the if block so eval_mask is always defined first.
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[
                        int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                        int(0.03594771 * gt_width):int(0.96405229 * gt_width)
                    ] = 1
                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[
                            int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                            int(0.0359477 * gt_width):int(0.96405229 * gt_width)
                        ] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1

                valid_mask = np.logical_and(valid_mask, eval_mask)  # only update when eval_mask exists

            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred_np[valid_mask]))

            # Log images for the very first validation batch only
            if should_log and i == 0:
                log_images(img[0], depth[0], pred_np, args, step)

        if skipped > 0:
            print(f"[Validation] Skipped {skipped} batches (no valid GT depth). "
                  f"Evaluated on {i + 1 - skipped}/{i + 1} batches.")

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(
        description='Training script. Default values of all arguments are recommended for reproducibility',
        fromfile_prefix_chars='@',
        conflict_handler='resolve'
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float,
                        help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float,
                        help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int,
                        help='validation period (in steps)')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use (None = all available)')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str,
                        help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str, help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to ground truth dataset")
    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')
    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--do_random_rotate', default=False,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images',
                        action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')
    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the ground truth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=False, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg ECCV16', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0

    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root, exist_ok=True)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')
        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])
    except KeyError:
        # Not using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        try:
            mp.set_start_method('forkserver')
        except RuntimeError:
            pass

        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(f"Distributed URL: {args.dist_url}")
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        print("\nControl entered distributed mode\n")
        args.world_size = ngpus_per_node * args.world_size
        print(f"ngpus_per_node: {ngpus_per_node}")
        print(f"args.world_size: {args.world_size}")
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
