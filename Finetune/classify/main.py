import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


import os
import sys
import json
import time
import math
import random
import datetime
import traceback
from pathlib import Path
from os.path import join as ospj
import wandb

import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # 正确导入mp
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchvision.transforms import v2

from utils import (
    fix_random_seeds,
    cosine_scheduler,
    cosine_scheduler_iter,
    MetricLogger,
)

from config import get_args
from data import get_data_loader, get_synth_train_data_loader
from models.clip import CLIP
from models.resnet50 import ResNet50
from util_data import SUBSET_NAMES

from torch.multiprocessing import Process



def load_data_loader(args):
    train_loader, test_loader = get_data_loader(
        real_train_data_dir=args.real_train_data_dir,
        real_test_data_dir=args.real_test_data_dir,
        dataset=args.dataset,
        bs=args.batch_size,
        eval_bs=args.batch_size_eval,
        n_img_per_cls=args.n_img_per_cls,
        is_synth_train=args.is_synth_train,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_real_shots=args.is_real_shots,
        model_type=args.model_type,
        csv_path=args.csv_path,
    )
    return train_loader, test_loader



def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        bs=args.batch_size,
        n_img_per_cls=args.n_img_per_cls,
        dataset=args.dataset,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_real_shots=args.is_real_shots,
        model_type=args.model_type,
        f_if=args.f_if,
    )
    return synth_train_loader



def main(args):
    # Setup the number of classes based on the dataset
    args.n_classes = len(SUBSET_NAMES[args.dataset])

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Fix random seeds for reproducibility
    fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    # ==================================================
    # Data loader
    # ==================================================

    train_loader_real, val_loader = load_data_loader(args)

    if args.is_synth_train:
        train_loader_sys = load_synth_train_data_loader(args)

    # Decide which training loader to use
    if args.is_synth_train:
        train_loader = train_loader_sys
    else:
        train_loader = train_loader_real

    # ==================================================
    # Model and optimizer
    # ==================================================
    if args.model_type == "clip":
        model = CLIP(
            dataset=args.dataset,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            clip_download_dir=args.clip_download_dir,
            clip_version=args.clip_version,
        )
        params_groups = model.learnable_params()
    elif args.model_type == "resnet50":
        model = ResNet50(n_classes=args.n_classes)
        params_groups = model.parameters()

    model = model.cuda('cuda')  # Move the model to GPU

    criterion = nn.CrossEntropyLoss().cuda()

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    # Optimizer and learning rate scheduler
    scheduler = None
    if not args.test:
        optimizer = torch.optim.AdamW(
            params_groups, lr=args.lr, weight_decay=args.wd,
        )
        # Setup the learning rate schedule based on the total iterations
        args.lr_schedule = cosine_scheduler_iter(
            args.lr,
            args.min_lr,
            total_iters=args.total_iterations,  # Use total iterations instead of epochs
            warmup_iters=int(0.1 * args.total_iterations), #int(3 * args.val_iter)
            start_warmup_value=args.min_lr,
        )

    fp16_scaler = None
    if args.use_fp16:
        # Enable mixed precision training
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ==================================================
    # Logging setup (Wandb or TensorBoard)
    # ==================================================
    if args.log == 'wandb':
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_group,
            settings=wandb.Settings(start_method='fork'),
            config=vars(args)
        )
        args.wandb_url = wandb.run.get_url()
    elif args.log == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb-{}".format(args.local_rank))
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir, flush_secs=30)

    # ==================================================
    # Training Loop
    # ==================================================
    print("=> Training starts ...")
    start_time = time.time()

    best_stats = {}
    best_top1 = 0.0
    total_iterations = args.total_iterations  # Define total iterations for training

    it = 0  # Initialize the iteration counter
    train_stats = None

    while it < total_iterations:
        # Dynamically compute epoch based on current iteration
        epoch = it // len(train_loader)

        # Train one "epoch" (dynamically handled by iteration count)
        if not args.test:
            train_stats, best_stats, best_top1 = train_one_epoch(
                model, criterion, train_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
                val_loader, best_stats, best_top1, it, total_iterations
            )
        if train_stats is None:
            break

        # Increment the global iteration counter
        it += len(train_loader)  # Increment by the number of iterations per epoch
        
        if args.log == 'wandb':
            if not args.test:
                train_stats.update({"iteration": it})
                wandb.log(train_stats, step=it)

    print('=> Final Evaluate model ...')
    test_stats = eval(
        model, criterion, val_loader, it, fp16_scaler, args, is_last=True)

    # Save the best model based on top-1 accuracy
    if test_stats["test/top1"] > best_top1 and not args.test:
        best_top1 = test_stats["test/top1"]
        best_stats = test_stats
        print('new best_top1: ', best_top1)
        save_model(args, model, optimizer, it, fp16_scaler, "best_checkpoint.pth")

    # Final evaluation stats logging
    if it >= total_iterations and not args.test:
        test_stats['test/best_top1'] = best_stats["test/top1"]
        test_stats['test/best_loss'] = best_stats["test/loss"]

    if args.log == 'wandb' and not args.test:
        wandb.log(test_stats, step=it)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1, current_iter, max_iter
):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Iteration: [{}/{}]".format(current_iter, args.total_iterations)

    model.train()

    for it, batch in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        global_iter = current_iter + it  # Global training iteration count
        
        if global_iter > max_iter:
            return None, best_stats, best_top1

        # Load data
        if args.is_real_shots:
            image, label, is_real = batch
        else:
            image, label = batch

        label_origin = label.cuda(non_blocking=True)

        # Apply CutMix and MixUp augmentation
        if args.is_mix_aug:
            p = random.random()
            if p < 0.2:
                if args.is_synth_train and args.is_real_shots:
                    new_image = torch.zeros_like(image)
                    new_label = torch.stack([torch.zeros_like(label)] * args.n_classes, dim=1).mul(1.0)

                    if len(torch.nonzero(is_real == 1, as_tuple=True)[0]) > 0:
                        image_real, label_real = image[is_real == 1], label[is_real == 1]
                        image_real, label_real = cutmix_or_mixup(image_real, label_real)
                        new_image[is_real == 1] = image_real
                        new_label[is_real == 1] = label_real
                    if len(torch.nonzero(is_real == 0, as_tuple=True)[0]) > 0:
                        image_synth, label_synth = image[is_real == 0], label[is_real == 0]
                        image_synth, label_synth = cutmix_or_mixup(image_synth, label_synth)
                        new_image[is_real == 0] = image_synth
                        new_label[is_real == 0] = label_synth

                    image = new_image
                    label = new_label
                else:
                    image, label = cutmix_or_mixup(image, label)

        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # Update learning rate and weight decay
        for i, param_group in enumerate(optimizer.param_groups):
            if global_iter < len(args.lr_schedule):
                param_group["lr"] = args.lr_schedule[global_iter]
                if i == 0:  # Only the first group is regularized
                    param_group["weight_decay"] = args.wd
            else:
                param_group["lr"] = args.min_lr

        # Forward pass
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            logit = model(image)
            if args.is_synth_train and args.is_real_shots:
                if len(torch.nonzero(is_real == 1, as_tuple=True)[0]) > 0:
                    loss_real = criterion(logit[is_real == 1], label[is_real == 1])
                else:
                    loss_real = 0
                if len(torch.nonzero(is_real == 0, as_tuple=True)[0]) > 0:
                    loss_synth = criterion(logit[is_real == 0], label[is_real == 0])
                else:
                    loss_synth = 0
                loss = args.lambda_1 * loss_real + (1 - args.lambda_1) * loss_synth
            else:
                loss = criterion(logit, label)

        # Backward and optimize
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        if fp16_scaler is None:
            loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # Logging
        with torch.no_grad():
            if args.dataset == 'waterbirds' or args.dataset == 'waterbirds_nobias':
                acc1 = get_accuracy(logit.detach(), label_origin, topk=(1,))

                # Record logs
                metric_logger.update(loss=loss.item())
                metric_logger.update(top1=acc1[0].item())
                
            else:

                acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))

                # Record logs
                metric_logger.update(loss=loss.item())
                metric_logger.update(top1=acc1.item())
                metric_logger.update(top5=acc5.item())
            
            
            # acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))
            # metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if scheduler is not None:
            scheduler.step()

        # Periodic validation
        if global_iter % args.val_iter == 0 and global_iter > 0:
            print('=> Evaluate model ...')
            test_stats = eval(
                model, criterion, val_loader, global_iter, fp16_scaler, args, is_last=False
            )
            if test_stats["test/top1"] > best_top1:
                best_top1 = test_stats["test/top1"]
                best_stats = test_stats
                print('new best_top1: ', best_top1)
                save_model(args, model, optimizer, global_iter, fp16_scaler, "best_checkpoint.pth")

            if args.log == 'wandb':
                wandb.log(test_stats, step=global_iter)

        model.train()

    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)

    return {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}, best_stats, best_top1


@torch.no_grad()
def eval(model, criterion, data_loader, iteration, fp16_scaler, args, is_last=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Iteration: [{}/{}]".format(iteration, args.total_iterations)

    if is_last is None:
        is_last = iteration + 1 == args.total_iterations
    else:
        is_last = is_last

    # If this is the final evaluation, collect targets and outputs
    if is_last:
        targets = []
        outputs = []

    model.eval()
    for it, (image, label) in enumerate(
            metric_logger.log_every(data_loader, 100, header)
    ):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # Compute output
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            output = model(image, phase="eval")
            loss = criterion(output, label)
            
        if args.dataset == 'waterbirds' or args.dataset == 'waterbirds_nobias':
            acc1 = get_accuracy(output, label, topk=(1,))

            # Record logs
            metric_logger.update(loss=loss.item())
            metric_logger.update(top1=acc1[0].item())
            
        else:

            acc1, acc5 = get_accuracy(output, label, topk=(1, 5))

            # Record logs
            metric_logger.update(loss=loss.item())
            metric_logger.update(top1=acc1.item())
            metric_logger.update(top5=acc5.item())

        if is_last:
            targets.append(label)
            outputs.append(output)

    metric_logger.synchronize_between_processes()
    print("Averaged test stats:", metric_logger)

    stat_dict = {"test/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}

    # If this is the last evaluation, calculate and log per-class accuracy
    if is_last:
        targets = torch.cat(targets)
        outputs = torch.cat(outputs)

        # Calculate per-class accuracy
        acc_per_class = [
            get_accuracy(outputs[targets == cls_idx], targets[targets == cls_idx], topk=(1,))[0].item()
            for cls_idx in range(args.n_classes)
        ]
        for cls_idx, acc in enumerate(acc_per_class):
            class_name = SUBSET_NAMES[args.dataset][cls_idx].replace('/', '_')
            print("{} [{}]: {}".format(SUBSET_NAMES[args.dataset][cls_idx], cls_idx, str(acc)))
            stat_dict[f'{class_name}' + '_cls-acc'] = acc

    return stat_dict

def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]



def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
        # world_size = torch.cuda.device_count()  # Assumes that we want to use all available GPUs
        # mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    except Exception as e:
        print(traceback.format_exc())
