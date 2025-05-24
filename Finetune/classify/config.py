import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import logging
import random
import atexit
import getpass
import shutil
import time
import os
import yaml
import json
import argparse
from os.path import join as ospj

from util_data import SUBSET_NAMES

_MODEL_TYPE = ("resnet50", "clip")


class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self.log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def str2bool(v):
    if v == "":
        return None
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2none(v):
    if v is None:
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return v

def int2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return int(v)

def float2none(v):
    if v is None or v == "":
        return v
    elif v.lower() in ('none', 'null'):
        return None
    else:
        return float(v)

def list_int2none(vs):
    return_vs = []
    for v in vs:
        if v is None:
            pass
        elif v.lower() in ('none', 'null'):
            v = None
        else:
            v = int(v)
        return_vs.append(v)
    return return_vs


def set_local(args):
    yaml_file = args.yaml_file
    with open(yaml_file, "r") as f:
        args_local = yaml.safe_load(f)
    # print(args.yaml_file)

    args.real_train_data_dir = args_local["real_train_data_dir"][args.dataset] + '/' + args.dataset
    args.real_train_fewshot_data_dir = ospj(
        args_local["real_train_fewshot_data_dir"][args.dataset]
    )
    args.real_test_data_dir = args_local["real_test_data_dir"][args.dataset]
    args.synth_train_data_dir = args_local["synth_train_data_dir"]
    args.clip_download_dir = args_local["clip_download_dir"]
    args.wandb_key = args_local["wandb_key"]


def set_output_dir(args):
    n_img_per_cls = "full" if args.n_img_per_cls is None else args.n_img_per_cls
    mid2 = f"n_img_per_cls_{n_img_per_cls}"
    if args.is_synth_train:
        mid3 = "SDXL_gen"
    else:
        mid3 = "baseline"
        if args.is_real_shots:
            mid3 += f"_shot{args.n_shot}_{args.fewshot_seed}"
    if args.is_synth_train:
        if args.n_shot == 0: # zeroshot
            mid3 = ospj(mid3, f"shot{args.n_shot}")
        else: # Finetune
            mid3 = ospj(mid3, f"shot{args.n_shot}")
            if args.no_mask:
                mid3 += "_nomask"
            else:
                mid3 += "_mask"
            if args.back:
                mid3 += "_back"
            if args.color:
                mid3 += "_color"
            if args.texture:
                mid3 += "_texture"
            if args.infeasible:
                mid3 += "_infeasible"
            if args.feasible:
                mid3 += "_feasible"
            if args.with_image:
                mid3 += "_with_image"
        if args.is_real_shots:
            mid3 += f"_lbd{args.lambda_1}"
    mixaug = "_mixuag" if args.is_mix_aug else ""
    mid4 = f"lr{args.lr}_wd{args.wd}{mixaug}"

    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version

    args.output_dir = ospj(args.output_dir, args.dataset, model_type, mid2, mid3, mid4)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


def set_synth_train_data_dir(args):
    if args.is_synth_train:
        if args.no_mask:
            mid_dir1 = "no_mask"
        else:
            mid_dir1 = "mask"
        if args.back:
            mid_dir2 = "back"
        elif args.color:
            mid_dir2 = "color"
        elif args.texture:
            mid_dir2 = "texture"
        else:
            raise RuntimeError('You must give a suitable name!')
        if args.no_mask and args.infeasible:
            if args.with_image:
                mid_dir3 = "infeasible/with_image"
            else:
                mid_dir3 = "infeasible/without_image"
        elif args.no_mask and args.feasible:
            if args.with_image:
                mid_dir3 = "feasible/with_image"
            else:
                mid_dir3 = "feasible/without_image"
        elif args.infeasible:
             mid_dir3 = "infeasible"
             
        elif args.feasible:
             mid_dir3 = "feasible"
    
        else:
            raise RuntimeError('You must give a suitable name!')

        args.synth_train_data_dir = ospj(
            args.synth_train_data_dir,
            args.dataset,
            mid_dir1,
            mid_dir2,
            mid_dir3,
        ) 
        # print(args.synth_train_data_dir)


def set_log(output_dir):
    log_file_name = ospj(output_dir, 'log.log')
    Logger(log_file_name)


def set_wandb_group(args):
    pooled = f"pool_lbd{args.lambda_1}" if args.is_real_shots else ""
    mixaug = "_mixaug" if args.is_mix_aug else ""
    synth_setting = ""
    if args.is_synth_train: 
        if args.n_shot == 0:
            synth_setting = "zeroshot"
        else:
            synth_setting = "OODdata"
            if args.is_real_shots:
                synth_setting += f"_shot{args.n_shot}"
            if args.no_mask:
                synth_setting += "_nomask"
            else:
                synth_setting += "_mask"
            if args.back:
                synth_setting += "_back"
            if args.color:
                synth_setting += "_color"
            if args.texture:
                synth_setting += "_texture"
            if args.infeasible:
                synth_setting += "_infeasible"
            if args.feasible:
                synth_setting += "_feasible"
    model_type = args.model_type
    if model_type == 'clip':
        model_type += args.clip_version

    args.wandb_group = f"{args.dataset[:4]}_{model_type}_{pooled}_{synth_setting}_nipc{args.n_img_per_cls}_lr{args.lr}_wd{args.wd}{mixaug}"

def set_follow_up_configs(args):
    if not args.test:
        set_output_dir(args)
    set_synth_train_data_dir(args)
    if not args.test:
        set_log(args.output_dir)
    if args.wandb_group is None:
        set_wandb_group(args)

def get_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_type', type=str2none, default=None,
                        choices=_MODEL_TYPE)
    parser.add_argument('--clip_version', type=str, default='ViT-B/16')

    # CLIP setting
    parser.add_argument("--is_lora_image", type=str2bool, default=True)
    parser.add_argument("--is_lora_text", type=str2bool, default=True)


    # Data
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument("--n_img_per_cls", type=int2none, default=100)
    parser.add_argument("--is_mix_aug", type=str2bool, default=False,
                        help="use mixup and cutmix")
    parser.add_argument("--is_real_shots", type=str2bool, default=False)
    parser.add_argument("--lambda_1", type=float2none, default=0,
                        help="weight for loss from real/synth data")
    parser.add_argument("--fewshot_seed", type=str2none, default="seed0",
                        help="best or seed{number}.")
    parser.add_argument("--is_dataset_wise", type=str2bool, default=False)
    parser.add_argument("--Finetune_lr", type=float2none, default=1e-4)
    parser.add_argument("--Finetune_epoch", type=int2none, default=200)
    parser.add_argument("--Finetune_train_text_encoder", type=str2bool, default=True)

    # stable diffusion
    parser.add_argument("--is_synth_train", type=str2bool, default=False)
    parser.add_argument("--sd_version", type=str2none, default=None)
    parser.add_argument("--guidance_scale", type=float2none, default=2.0)
    parser.add_argument("--num_inference_steps", type=int2none, default=50)
    # for few-shot
    parser.add_argument("--n_shot", type=int2none, default=16)
    parser.add_argument("--n_template", type=int2none, default=1)

    # Basic arguments
    parser.add_argument("--back", type=str2bool, default=False, help="use background augmentation")
    parser.add_argument("--color", type=str2bool, default=False, help="use color augmentation")
    parser.add_argument("--texture", type=str2bool, default=False, help="use texture ay7ugmentation")

    parser.add_argument("--no_mask", type=str2bool, default=False, help="whether to use sdxl-inpainting")
    parser.add_argument("--with_image", action="store_true", help="whether to use sdxl-inpainting")

    # Mutually exclusive group for infeasible and feasible
    parser.add_argument("--infeasible", type=str2bool, default=False, help="use unfeasible color")
    parser.add_argument("--feasible", type=str2bool, default=False, help="use feasible color")
    parser.add_argument("--f_if", type=str2bool, default=False, help="use feasible + infeasible")
    
    parser.add_argument("--csv_path", type=str, default="/shared-network/yliu/projects/OODData/Finetune/artifacts")
    parser.add_argument("--yaml_file", type=str, default="/shared-network/yliu/projects/OODData/Finetune/classify/local.yaml")
    parser.add_argument("--subset", action="store_true", help="whether to use subset of the dataset")
    parser.add_argument("--val_iter", type=int, default=450)
    
    parser.add_argument("--test_save_root", type=str, default=None)

    
    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        default=True,
        help="Whether or not to use mixed precision for training.",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU. Total batch size is proportional to the number of GPUs.",
    )
    parser.add_argument(
        "--batch_size_eval",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--wd",
        type=float2none,
        default=1e-4,
        help="Weight decay for the SGD optimizer.",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float2none,
        help="Maximum learning rate at the end of linear warmup.",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=25,
        type=int,
        help="Number of training epochs for the learning-rate-warm-up phase.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate at the end of training.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="Path to the output folder to save logs and checkpoints.",
    )
    parser.add_argument(
        "--saveckpt_freq",
        default=100,
        type=int,
        help="Frequency of intermediate checkpointing.",
    )
    parser.add_argument(
        "--seed",
        default=22,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        default=12,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="Url used to set up distributed training.",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore this argument; No need to set it manually.",
    )
    # wandb args
    parser.add_argument('--log', type=str, default='tensorboard', help='How to log')
    parser.add_argument('--wandb_entity', type=str, default='regaz', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='Finetune_ood', help='Wandb project name')
    parser.add_argument('--wandb_group', type=str2none, default=None, help='Name of the group for wandb runs')
    parser.add_argument('--wandb_key', default='<your_api_key_here>', type=str, help='API key for W&B.')

    parser.add_argument('--test', action="store_true", help='test version.')
    parser.add_argument('--ckpt_path', type=str, default=None, help='load ckpt path')
    parser.add_argument('--total_iterations', type=int, default=46000, help='total_training_iterations')


    args = parser.parse_args()

    if args.back and args.color:
        raise RuntimeError("Error: --back or --color can only be used once.")

    # if args.infeasible and args.feasible:
    #     raise RuntimeError("Error: --feasible or --infeasible can only be used once.")
    
    if (args.with_image):
        if (not args.no_mask):
            raise RuntimeError("Error: -with_image could only be used when no_mask is given.")

    set_local(args)
    set_follow_up_configs(args)
    
    print(f'Current experiment setting is lr: {args.lr}, warm up step is {args.warmup_epochs}, lambda is {args.lambda_1}, weight decay is {args.wd}')

    return args


