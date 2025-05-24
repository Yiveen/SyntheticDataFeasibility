import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import os
from os.path import expanduser
from os.path import join as ospj
import json
import pickle
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision as tv
from collections import defaultdict,Counter
from pathlib import Path
import random

import pandas as pd
import torchvision.datasets as dsets
import glob
import matplotlib.pyplot as plt

# from utils.utils import make_dirs
from Finetune.classify.util_data import (
    SUBSET_NAMES,
    configure_metadata, get_image_ids, get_class_labels,
    GaussianBlur, Solarization,
)
from torch.utils.data.distributed import DistributedSampler

NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)
CLIP_NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_NORM_STD = (0.26862954, 0.26130258, 0.27577711)


def get_transforms(model_type):
    """
    Return data augmentation transforms for training and testing.
    
    Args:
        model_type (str): Model type, e.g., 'clip' or 'resnet50'.

    Returns:
        Tuple[Transform, Transform]: Training and testing torchvision transforms.
    """
    if model_type != None:
        if model_type == "clip":
            norm_mean = CLIP_NORM_MEAN
            norm_std = CLIP_NORM_STD
        elif model_type == "resnet50":
            norm_mean = NORM_MEAN
            norm_std = NORM_STD

        aux_transform = tv.transforms.Compose([
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomApply(
                [
                    tv.transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                    )
                ],
                p=0.8,
            ),
            tv.transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.2),
            Solarization(0.2),
        ])
        train_transform = tv.transforms.Compose([
            tv.transforms.Lambda(lambda x: x.convert("RGB")),
            tv.transforms.RandAugment(),
            tv.transforms.RandomResizedCrop(
                224,
                scale=(0.25, 1.0),
                interpolation=tv.transforms.InterpolationMode.BICUBIC,
                antialias=None,
            ),
            aux_transform,
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(norm_mean, norm_std)
        ])

        test_transform = tv.transforms.Compose([
            tv.transforms.Lambda(lambda x: x.convert("RGB")),
            tv.transforms.Resize(
                224,
                interpolation=tv.transforms.functional.InterpolationMode.BICUBIC
            ),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(norm_mean, norm_std)
        ])
    else:
        train_transform = tv.transforms.Compose([
            tv.transforms.Lambda(lambda x: x.convert("RGB")),
            tv.transforms.ToTensor(),
        ])

        test_transform = tv.transforms.Compose([
            tv.transforms.Lambda(lambda x: x.convert("RGB")),
            tv.transforms.ToTensor(),
        ])

    return train_transform, test_transform

def get_counts(labels):
    """
    Compute inverse class frequency weights.

    Args:
        labels (List[int]): List of integer labels.

    Returns:
        torch.Tensor: Normalized inverse frequency weights.
    """
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

class Waterbirds:
    """
    Dataset wrapper for the Waterbirds dataset, with group label logic for spurious correlation analysis.
    """
    def __init__(self, root, split, transform=None):
        self.root = root
        self.df = pd.read_csv(os.path.join(root, 'metadata.csv'))
        if split == 'train':
            self.df = self.df[self.df.split == 0] 
        elif split == 'val':
            self.df = self.df[self.df.split == 1] 
        elif split == 'test':
            self.df = self.df[self.df.split == 2] 
        self.class_names = ['Landbird', 'Waterbird']
        self.group_names = ['land_landbird', 'land_waterbird', 'water_landbird', 'water_waterbird']
        self.samples = [(f, y) for f, y in zip(self.df['img_filename'], self.df['y'])]
        self.targets = [s[1] for s in self.samples]
        self.classes = ['land', 'water']
        self.transform = transform
        self.class_weights = get_counts(self.df['y']) 
        self.groups = [] 
        for (label, place) in zip(self.df['y'], self.df['place']):
            if place == 0:
                group = 0 if label == 0 else 1
            else:
                group = 2 if label == 0 else 3
            self.groups.append(group)
        print(f"{split} \t group counts: {Counter(self.groups)}")
        self.group_weights = get_counts(self.groups)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        img = Image.open(os.path.join(self.root, sample['img_filename'])).convert('RGB')
        label = int(sample['y'])
        place = int(sample['place'])
        group = self.groups[idx]
        species = sample['img_filename'].split('/')[0]
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_subset(self, groups=[0,1,2,3], num_per_class=1000):
        self.df['group'] = self.groups
        df = self.df.reset_index(drop=True)
        df['orig_idx'] = df.index
        df = df[df.group.isin(groups)]
        result = df.groupby('group').apply(
            lambda x: x.sample(n=num_per_class, random_state=42) if len(x) > num_per_class else x
        ).reset_index(drop=True)
        
        for group_id in groups:
            print(f'Group {group_id} samples: {len(result[result["group"] == group_id])}')
        
        return result['orig_idx'].values
        
class Subset(torch.utils.data.Dataset):
    """
    Subset wrapper for filtering a subset of a dataset by indices.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.samples = [self.dataset.samples[i] for i in indices]
        print(f"num samples {len(self.samples)}")
        self.groups = [self.dataset.groups[i] for i in indices]
        self.classes = self.dataset.classes
        self.group_names = self.dataset.group_names
        self.class_names = self.dataset.class_names
        self.targets = [s[1] for s in self.samples]
        self.class_weights = get_counts([s[1] for s in self.samples])

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)
    
    
class CombinedDataset(torch.utils.data.Dataset):
    """
    Combines multiple datasets into one for joint loading and visualization.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.idx_to_dataset, self.idx_mapping, curr = [], [], 0 
        for j, d in enumerate(datasets):
            self.idx_to_dataset.extend([j] * len(d))
            self.idx_mapping.extend([i for i in range(len(d))])
            curr += len(d)
        # print(self.datasets)
        # print(self.datasets[0].classes, self.datasets[1].classes)
        assert len(datasets[0].classes) == len(datasets[1].classes)
        self.samples = np.concatenate([d.samples for d in datasets])
        self.all_path = [s[0] for s in self.samples]
        self.samples = [(s[0], int(s[1])) for s in self.samples]
        print("samples new ", self.samples[:5])
        self.groups = np.concatenate([d.groups for d in datasets])
        self.targets = np.concatenate([[s[1] for s in d.samples] for d in datasets])
        self.class_weights = get_counts([s[1] for s in self.samples])
        self.classes = datasets[0].classes
        self.group_names = np.concatenate([d.group_names for d in datasets])
        self.class_names = datasets[0].class_names
        print(f"Combining datasets of size {[len(d) for d in datasets]} \t Total = {len(self.samples)}")

    def __getitem__(self, index):
        return self.datasets[self.idx_to_dataset[index]][self.idx_mapping[index]]

    def vis_dsets(self, idx, save=False):
        fig, axs = plt.subplots(1, len(self.datasets), figsize=(20, 20), constrained_layout=True)
        imgs = []
        for i, d in enumerate(self.datasets):
            img, label, _ = d[idx]
            # print(d.samples[idx][0])
            imgs.append(img)
            axs[i].imshow(img)
            axs[i].set_title(label)
            axs[i].axis('off')
        if save:
            plt.savefig(f"figs/vis_dsets_{idx}.png")
        return imgs

    def __len__(self):
        return len(self.samples)



class DatasetSynthImage(Dataset):
    """
    Synthetic dataset class that supports mixing feasible and infeasible synthetic images,
    optionally augmented with real few-shot samples.
    """
    def __init__(
            self,
            synth_train_data_dir,
            transform,
            target_label=None,
            n_img_per_cls=None,
            dataset='imagenet',
            n_shot=16,
            real_train_fewshot_data_dir='',
            is_real_shots=False,
            f_if=False,
            **kwargs
    ):
        self.synth_train_data_dir = synth_train_data_dir
        # print(self.synth_train_data_dir)
        self.transform = transform
        self.is_real_shots = is_real_shots

        self.image_paths = []
        self.image_labels = []

        value_counts = defaultdict(int)

        iter_label = SUBSET_NAMES[dataset]
            
        print('synthetic iter lables', iter_label)
            
        if f_if: # Mix training
            for label, class_name in enumerate(iter_label):
                if target_label is not None and label != target_label:
                    continue

                real_img_paths = os.listdir(
                    os.path.join(real_train_fewshot_data_dir, class_name)
                )
                real_total_number = len(real_img_paths)
                each_choosed_number = real_total_number * (n_img_per_cls // n_shot)

                feasible_dir = os.path.join(synth_train_data_dir, class_name)
                infeasible_dir = os.path.join(
                    Path(synth_train_data_dir).parent, 'infeasible', class_name
                )

                feasible_images = [
                    fname for fname in os.listdir(feasible_dir)
                    if fname.endswith(('.jpg', '.png'))  
                ]
                infeasible_images = [
                    fname for fname in os.listdir(infeasible_dir)
                    if fname.endswith(('.jpg', '.png'))
                ]

                random.shuffle(feasible_images)
                random.shuffle(infeasible_images)

                # Ensure total number is exactly each_choosed_number
                feasible_count = each_choosed_number // 2
                infeasible_count = each_choosed_number - feasible_count  # Fill the remainder

                feasible_selected = feasible_images[:feasible_count]
                infeasible_selected = infeasible_images[:infeasible_count]

                for fname in feasible_selected:
                    self.image_paths.append(os.path.join(feasible_dir, fname))
                    self.image_labels.append(label)

                for fname in infeasible_selected:
                    self.image_paths.append(os.path.join(infeasible_dir, fname))
                    self.image_labels.append(label)

        else:
            for label, class_name in enumerate(iter_label):
                if target_label is not None and label != target_label:
                    continue

                for fname in os.listdir(ospj(synth_train_data_dir, class_name)):
                    if fname.endswith(".txt") or fname.endswith(".json"):
                        continue

                    suffix = fname.split('_')[-1].split('.')[0]
                    
                    basename = fname.split('_')[0]

                    # based on the value of to generate number dynamically
                    max_suffix = n_img_per_cls // n_shot 
                    valid_suffixes = [str(i) for i in range(0, max_suffix)]  

                    if suffix in valid_suffixes:
                        if n_img_per_cls is not None:
                                if value_counts[label] < n_img_per_cls:
                                    value_counts[label] += 1
                        self.image_paths.append(ospj(synth_train_data_dir, class_name, fname))
                        self.image_labels.append(label)
                        
                    else:
                        continue 

        if self.is_real_shots: 
            print('get the real shot images')
            if n_shot == 0:
                n_shot = 16
            reps = round(n_img_per_cls // n_shot)
            for label, class_name in enumerate(iter_label):
                real_img_paths = os.listdir(
                    ospj(real_train_fewshot_data_dir, class_name))
                real_total_number = len(real_img_paths)
                # print('current real images:',real_total_number)
                mut = min(n_shot, real_total_number)
                real_subset = [
                    ospj(
                        real_train_fewshot_data_dir,
                        class_name,
                        real_img_paths[i]
                    ) for i in range(mut)
                ]
                for i in range(reps):
                    self.image_paths.extend(real_subset)
                    self.image_labels.extend([label] * mut)
                    
    def open_and_convert_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            return image

        except (IOError, SyntaxError, ValueError) as e:
            print(f"Failed to load or convert image: {image_path}. Error: {e}")
            return None

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # if idx == 0:
        #     print('image_path0', self.image_paths)
        #     print(len(self.image_paths))
        image_label = self.image_labels[idx]
        
        # image = Image.open(image_path)
        # image = image.convert('RGB')
        image = self.open_and_convert_image(image_path)
        # print(f"Image mode: {image.mode}, Type: {type(image)}")
        # image = image.resize((64, 64), Image.Resampling.LANCZOS)  # Resize the image to 64x64
        image = self.transform(image)

        is_real = "real_train" in image_path  # 我们选择的真实图片路径要有real_train

        if self.is_real_shots:
            return image, image_label, is_real
        else:
            return image, image_label

    def __len__(self):
        return len(self.image_paths)


def filter_dset(dataset, n_img_per_cls, dataset_name, OODGEN=False):
    """
    Filter a dataset to balance the number of images per class.

    Args:
        dataset (Dataset): The dataset to filter.
        n_img_per_cls (int): Max images per class.
        dataset_name (str): Name of the dataset.
        OODGEN (bool): If OOD generation context is applied.

    Returns:
        Tuple[Dataset, List[int]]: Filtered dataset and selected class indices.
    """
    import random
    random.seed(6)
    if dataset_name == 'oxford_pets':
        _images = dataset._images
        _labels = dataset._labels
    elif dataset_name == 'fgvc_aircraft':
        _images = dataset._image_files
        _labels = dataset._labels
    elif dataset_name == 'cars':
        _images = [sample[0] for sample in dataset._samples]
        _labels = [sample[1] for sample in dataset._samples]
    else:
        raise ValueError("Please specify valid dataset.")
    new_images = []
    new_labels = []
    
    index = sorted(list(set(_labels)))
        
    for new_label, i in enumerate(index):
        candidates = [j for j in range(len(_labels)) if _labels[j] == i]
        img_per_cls = min(n_img_per_cls, len(candidates))  # allow for less if not enough
        idx = random.sample(range(0, len(candidates)), img_per_cls)
        new_images.extend([_images[candidates[j]] for j in idx])
        new_labels.extend([_labels[candidates[j]] for j in idx])
    if dataset_name == 'oxford_pets':
        dataset._images = new_images
        dataset._labels = new_labels
    elif dataset_name == 'fgvc_aircraft':
        dataset._image_files = new_images
        dataset._labels = new_labels
    elif dataset_name == 'cars':
        dataset._samples = [(im, lab) for im, lab in zip(new_images, new_labels)]
    else:
        raise ValueError("Please specify valid dataset.")
    return dataset, index


def split_pets(real_train_data_dir, train_transform, split, csv_path, DOWNLOAD):
    """
    Load Oxford Pets dataset and split into train/test according to provided CSV.

    Args:
        real_train_data_dir (str): Root directory for the dataset.
        train_transform (Transform): Transformations to apply.
        split (str): 'train' or 'test'.
        csv_path (str): Path to the split definition CSV.
        DOWNLOAD (bool): Whether to download the dataset if not present.

    Returns:
        Dataset: Subset of the Oxford Pets dataset.
    """
    import csv
    pets_path_train = os.path.join(real_train_data_dir, 'train')
    train_dataset = tv.datasets.OxfordIIITPet(
        root=pets_path_train,
        split='trainval',
        target_types='category',
        download=DOWNLOAD,
        transform=train_transform,
    )
    test_dataset = tv.datasets.OxfordIIITPet(
        root=pets_path_train,
        split='test',
        target_types='category',
        download=DOWNLOAD,
        transform=train_transform,
    )
    train_dataset._images = train_dataset._images + test_dataset._images
    train_dataset._labels = train_dataset._labels + test_dataset._labels

    # split taken from DISEF paper at
    # https://github.com/vturrisi/disef/blob/main/fine-tune/artifacts/caltech101/split_coop.csv
    split_file_path = os.path.join(csv_path, 'split_coop.csv')
    split_files = []
    with open(split_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split'] == split:
                split_files.append(row['filename'].split('/')[-1])
    file_path_full = os.path.join(pets_path_train, 'oxford-iiit-pet', 'images') + '/'
    ind_to_keep = [i for i, file in enumerate(train_dataset._images)
                   if str(file).replace(file_path_full, '') in split_files]
    train_dataset._images = [l for i, l in enumerate(train_dataset._images) if i in ind_to_keep]
    train_dataset._labels = [l for i, l in enumerate(train_dataset._labels) if i in ind_to_keep]
    return train_dataset


def get_data_loader(
        real_train_data_dir="",
        real_test_data_dir="",
        dataset="imagenet",
        bs=32,
        eval_bs=32,
        is_rand_aug=True,
        target_label=None,
        n_img_per_cls=None,
        is_synth_train=False,
        n_shot=0,
        real_train_fewshot_data_dir='',
        is_real_shots=False,
        model_type=None,
        csv_path=None,
        return_root=False,
        DOWNLOAD=False,
        subset=False,
        rank=None,
        world_size=None,
        OODGEN=False,
        num_workers=16,
        transformation=None,
        multithread=False,
        save_few_shots=False,
):
    """
    Get data loaders for real datasets with configurable options.

    Returns:
        train_loader, test_loader (and optionally image paths or labels).
    """
    if not multithread:
        train_transform, test_transform = get_transforms(model_type)
    else:
        train_transform, test_transform = transformation[0], transformation[1]
    csv_path = ospj(csv_path, dataset)

    # if is_synth_train:
    #     train_loader = None
    # # else:

    if dataset == 'oxford_pets':
        train_dataset = split_pets(real_train_data_dir, train_transform, 'train', csv_path, DOWNLOAD)

        train_dataset, selected_index = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset, OODGEN=OODGEN)

    elif dataset == 'fgvc_aircraft':
        aircraft_path_train = os.path.join(real_train_data_dir, 'train')
        train_dataset = tv.datasets.FGVCAircraft(
            root=aircraft_path_train,
            split='trainval',
            annotation_level='variant',
            transform=train_transform,
            download=DOWNLOAD,
        )
        train_dataset, selected_index = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset, OODGEN=OODGEN)

    elif dataset == 'cars':
        cars_path_train = os.path.join(real_train_data_dir, 'train')
        train_dataset = tv.datasets.StanfordCars(
            root=cars_path_train,
            split='train',
            transform=train_transform,
            download=DOWNLOAD,
        )
        train_dataset, selected_index = filter_dset(dataset=train_dataset, n_img_per_cls=n_img_per_cls, dataset_name=dataset, OODGEN=OODGEN)

    elif dataset == "waterbirds" or dataset == 'waterbirds_nobias': 
        waterbird_path = os.path.join(real_train_data_dir)
        trainset = Waterbirds(root=waterbird_path, split='train', transform=train_transform)
        
        train_extra_ids = trainset.get_subset(groups=[1,2], num_per_class=1000)
        extra_trainset = Subset(trainset, train_extra_ids)
        
        valset = Waterbirds(root=waterbird_path, split='val', transform=train_transform)
        
        extra_idxs = valset.get_subset(groups=[1,2], num_per_class=1000)
        extra_valset = Subset(valset, extra_idxs)
        extraset = CombinedDataset([extra_valset, extra_trainset])
        train_dataset = extraset
        selected_index = [0, 1]
    
    
    else:
        raise ValueError("Please specify a valid dataset.")

    selected_labels = selected_index

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs,
        sampler=None,
        shuffle=is_rand_aug, prefetch_factor=None if multithread else 4,pin_memory=True,
        num_workers=num_workers)

    #####################################################
    #####################################################
    #####################################################

    if dataset == 'oxford_pets':
        test_dataset = split_pets(real_train_data_dir, test_transform, 'test', csv_path, DOWNLOAD)

    elif dataset == 'fgvc_aircraft':
        aircraft_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.FGVCAircraft(
            root=aircraft_path_test,
            split='test',
            annotation_level='variant',
            transform=test_transform,
            download=True,
        )
        
    elif dataset == 'cars':
        # note: this must be train, because the Cars dataset had to be downloaded by hand and is kept in the same
        # directory as the training data
        cars_path_test = os.path.join(real_train_data_dir, 'train')
        test_dataset = tv.datasets.StanfordCars(
            root=cars_path_test,
            split='test',
            transform=test_transform,
            download=False,
        )
            
    elif dataset == "waterbirds" or dataset == 'waterbirds_nobias': 
        waterbird_path = os.path.join(real_train_data_dir)
        test_dataset = Waterbirds(root=waterbird_path, split='test', transform=test_transform)
        
    else:
        raise ValueError("Please specify a valid dataset.")
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_bs, shuffle=False,
        num_workers=16, pin_memory=True, sampler=None)
    
    print("Total real training number:", len(train_dataset))
    print("Total real test number:", len(test_dataset))
    
    if return_root:
        if dataset == 'oxford_pets':
            return train_loader, test_loader, train_dataset._images
        elif dataset == 'fgvc_aircraft':
            return train_loader, test_loader, train_dataset._image_files
        elif dataset == 'cars':
            return train_loader, test_loader, [sample[0] for sample in train_dataset._samples]
        elif dataset == 'waterbirds':
            return train_loader, test_loader, train_dataset.all_path
        else:
            return train_loader, test_loader, train_dataset._images
        
    elif save_few_shots:
        if dataset == 'oxford_pets':
            return train_loader, test_loader, train_dataset._images, train_dataset._labels
        elif  dataset == 'fgvc_aircraft':
            return train_loader, test_loader, train_dataset._image_files, train_dataset._labels
        elif dataset == 'cars':
            return train_loader, test_loader, [sample[0] for sample in train_dataset._samples], [sample[1] for sample in train_dataset._samples]
        elif dataset == 'waterbirds':
            return train_loader, test_loader, train_dataset.all_path, train_dataset.targets
        else:
            return train_loader, test_loader, train_dataset._images, train_dataset._images, train_dataset._labels
    else:
        return train_loader, test_loader


def get_synth_train_data_loader(
        synth_train_data_dir="data_synth",
        bs=32,
        is_rand_aug=True,
        target_label=None,
        n_img_per_cls=None,
        dataset='imagenet',
        n_shot=0,
        real_train_fewshot_data_dir='',
        is_real_shots=False,
        model_type=None,
        f_if=False,
):
    """
    Get data loader for synthetic datasets with optional few-shot real data augmentation.

    Returns:
        DataLoader: PyTorch DataLoader for synthetic data.
    """
    train_transform, test_transform = get_transforms(model_type)

    train_dataset = DatasetSynthImage(
        synth_train_data_dir=synth_train_data_dir,
        transform=train_transform if is_rand_aug else test_transform,
        target_label=target_label,
        n_img_per_cls=n_img_per_cls,
        dataset=dataset,
        n_shot=n_shot,
        real_train_fewshot_data_dir=real_train_fewshot_data_dir,
        is_real_shots=is_real_shots,
        f_if=f_if,
    )
    

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs,
        sampler=None,
        shuffle=is_rand_aug,
        num_workers=16, pin_memory=True,
    )
    
    print("Total synthetic training number:", len(train_dataset))

    return train_loader




