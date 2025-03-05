import os
import pickle

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import json


def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.args = args
        if args.dataset_name == 'iu_xray':
            self.ann_path = args.ann_path.split('&')[0]
            self.image_dir = args.image_dir.split('&')[0]
        elif args.dataset_name == 'mimic_cxr':
            self.ann_path = args.ann_path.split('&')[1]
            self.image_dir = args.image_dir.split('&')[1]
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        with open(self.ann_path, encoding='utf-8') as f:
            self.ann = json.load(f)
            f.close()
        # self.ann = pd.read_csv(self.ann_path)
        self.examples = self.ann[self.split]
        # self.examples = self.ann[self.ann['Split'] == self.split]
        # reset the index, and each split has one dataframe
        # self.examples.reset_index(drop=True, inplace=True)
        ## create a dict to store the mask
        self.masks = []
        self.reports = []
        self.prompt = 'a picture of '

        for i in range(len(self.examples)):
            if self.split == 'train':
                caption = tokenizer(self.prompt + self.examples[i]['report'])[:self.max_seq_length]
            else:
                caption = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            # caption = tokenizer(self.examples[i]['report'], padding='longest', max_length=250, return_tensors="pt")
            # caption = caption['input_ids'][:self.max_seq_length]
            self.examples[i]['ids'] = caption
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])
            # self.masks.append([1] * len(self.reports[i]))

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        # example = self.examples.loc[idx]
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1 = [
            load_pickle("{}/data/IU_xray_segmentation/{}/0_mask/{}_concat.pkl".format(self.mask_path, image_id, each))
            for each in ["bone", "lung", "heart", "mediastinum"]]
        mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2 = [
            load_pickle("{}/data/IU_xray_segmentation/{}/1_mask/{}_concat.pkl".format(self.mask_path, image_id, each))
            for each in ["bone", "lung", "heart", "mediastinum"]]

        if self.transform is not None:
            image_1, mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1 = \
                self.transform(image_1, mask_bone_1, mask_lung_1, mask_heart_1, mask_mediastinum_1)
            image_2, mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2 = \
                self.transform(image_2, mask_bone_2, mask_lung_2, mask_heart_2, mask_mediastinum_2)
        image = torch.stack((image_1, image_2), 0)
        image_mask_bone = torch.stack((mask_bone_1, mask_bone_2), 0)
        image_mask_lung = torch.stack((mask_lung_1, mask_lung_2), 0)
        image_mask_heart = torch.stack((mask_heart_1, mask_heart_2), 0)
        image_mask_mediastinum = torch.stack((mask_mediastinum_1, mask_mediastinum_2), 0)
        report_ids = example['ids']     #ids?
        report_masks = example['mask']      #mask?
        seq_length = len(report_ids)
        sample = (image_id, image, image_mask_bone, image_mask_lung, image_mask_heart, image_mask_mediastinum, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample




