# Copyright by HQ-SAM team
# All rights reserved.

# data loader
from __future__ import print_function, division

import os
import random
from copy import deepcopy
from skimage import io
import numpy as np

import ijson
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler


# --------------------- Agumentations (no change) ---------------------

class RandomHFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[2])

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class Resize(object):
    def __init__(self, size=[320, 320]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']  # noqa

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), self.size, mode='bilinear'), dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), self.size, mode='bilinear'), dim=0)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size)}


class RandomCrop(object):
    def __init__(self, size=[288, 288]):
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']  # noqa

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top:top + new_h, left:left + new_w]
        label = label[:, top:top + new_h, left:left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image, self.mean, self.std)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()  # noqa

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()

        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image, 0), scaled_size.tolist(), mode='bilinear'), dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label, 0), scaled_size.tolist(), mode='bilinear'), dim=0)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:, crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0, padding_w, 0, padding_h], value=128)
        label = F.pad(scaled_label, [0, padding_w, 0, padding_h], value=0)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(image.shape[-2:])}


# --------------------- dataloader online ---------------------

def load_coco(file_path):
    """Lazy load the MSCOCO json file to create a list of dictionaries with image names and mask IDs."""
    with open(file_path, 'rb') as file:
        # efficiently create a map from image ID to file name
        image_id_to_path = {}
        for item in ijson.items(file, 'images.item'):
            image_id_to_path[item['id']] = item['file_name']
        file.seek(0)  # rewind file position to the start for parsing annotations
        # iterate over each annotation to get the mask ID and corresponding image file name
        masks = []
        for annotation in ijson.items(file, 'annotations.item'):
            image_id = annotation['image_id']
            mask_id = annotation['id']
            if image_id in image_id_to_path:  # ensure the image_id was found in the initial mapping
                image_name = os.path.basename(image_id_to_path[image_id])
                masks.append({
                    'image_name': image_name,
                    'mask_id': mask_id
                })
    return masks


def fetch_segmentation_for_id(file_path, mask_id):
    """Fetch segmentation data for a specific mask ID using ijson."""
    with open(file_path, 'rb') as file:
        annotations = ijson.items(file, 'annotations.item')
        for annotation in annotations:
            if annotation['id'] == mask_id:
                return annotation.get('segmentation', [])
    return []  # return an empty list if no matching mask ID is found


def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for idx, _ in enumerate(datasets):
        print("--->>>", flag, " dataset ", idx, "/", len(datasets), " ", datasets[idx]["name"], "<<<---")
        gt_coco_path = None
        # image_paths = get_paths(datasets[idx]["image_dir"])
        image_dir = datasets[idx]["image_dir"]
        print('-im-', datasets[idx]["name"], datasets[idx]["image_dir"], ': ', len(os.listdir(image_dir)) - 1)

        if not os.path.exists(datasets[idx]["coco_json_path"]):
            print('-gt-', datasets[idx]["name"], datasets[idx]["coco_json_path"], ': ', 'No Ground Truth Found')
        else:
            gt_coco_path = datasets[idx]["coco_json_path"]
            with open(gt_coco_path, 'rb') as file:
                annotation_count = 0
                annotations = ijson.items(file, 'annotations.item')
                for _ in annotations:
                    annotation_count += 1
            print('-gt-', datasets[idx]["name"], datasets[idx]["coco_json_path"], annotation_count)

        name_im_gt_list.append(
            {
                "dataset_name": datasets[idx]["name"],
                "image_dir": image_dir,
                "coco_json_path": gt_coco_path
            })
    return name_im_gt_list


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        # combine different datasets into one
        dataset_names = []
        dt_name_list = []  # dataset name per mask
        dt_coco_list = []  # datatset coco path per mask
        im_name_list = []  # image names
        im_path_list = []  # image paths
        gt_mask_id_list = []  # gt masks

        for _, dataset in enumerate(name_im_gt_list):
            masks = load_coco(dataset["coco_json_path"])  # get the masks info from the MSCOCO json file
            dataset_names.append(dataset["dataset_name"])  # collect all the dataset names
            dt_name_list.extend([dataset["dataset_name"] for x in masks])  # dataset name repeated based on the number of masks in this dataset
            dt_coco_list.extend([dataset["coco_json_path"] for x in masks])  # dataset coco path repeated based on the number of masks in this dataset
            im_name_list.extend([mask['image_name'] for mask in masks])  # get all the image names
            im_path_list.extend([os.path.join(dataset['image_dir'], mask['image_name']) for mask in masks])  # get all the image paths
            gt_mask_id_list.extend([mask['mask_id'] for mask in masks])  # get all the masks in numpy arrays

        self.dataset["dataset_name"] = dt_name_list
        self.dataset["coco_path"] = dt_coco_list
        self.dataset["image_names"] = im_name_list
        self.dataset["image_paths"] = im_path_list
        self.dataset["original_image_paths"] = deepcopy(im_path_list)
        self.dataset["gt_mask_ids"] = gt_mask_id_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["gt_mask_ids"])

    def __getitem__(self, idx):
        coco_path = self.dataset["coco_path"][idx]
        image_path = self.dataset["image_paths"][idx]
        image = io.imread(image_path)
        gt_id = self.dataset["gt_mask_ids"][idx]
        segmentation = fetch_segmentation_for_id(coco_path, gt_id)
        mask_img = Image.new('L', (image.shape[1], image.shape[0]), 0); draw = ImageDraw.Draw(mask_img)  # noqa: create a empty binary mask image
        for segment in segmentation:
            polygon = [(segment[idx], segment[idx + 1]) for idx in range(0, len(segment), 2)]  # convert a flat list of coordinates into pairs
            draw.polygon(polygon, outline=255, fill=255)  # get the mask colored with 255
        gt = np.array(mask_img)  # convert the PIL image to a NumPy array

        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(image.shape) < 3:
            image = image[:, :, np.newaxis]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        image = torch.tensor(image.copy(), dtype=torch.float32)
        image = torch.transpose(torch.transpose(image, 1, 2), 0, 1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32), 0)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),
            "image": image,
            "label": gt,
            "shape": torch.tensor(image.shape[-2:]),
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["image_paths"][idx]
        return sample


def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if len(name_im_gt_list) == 0:
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if batch_size > 1:
        num_workers_ = 2
    elif batch_size > 4:
        num_workers_ = 4
    elif batch_size > 8:
        num_workers_ = 8

    if training:
        for _, dataset in enumerate(name_im_gt_list):
            gos_dataset = OnlineDataset([dataset], transform=transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for _, dataset in enumerate(name_im_gt_list):
            gos_dataset = OnlineDataset([dataset], transform=transforms.Compose(my_transforms), eval_ori_resolution=True)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets
