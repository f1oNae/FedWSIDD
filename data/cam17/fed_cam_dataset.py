import h5py
import torch
from PIL import Image
from torchvision import transforms
import openslide
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Camelyon17Raw(Dataset):
    def __init__(
            self, X_dtype=torch.float32, y_dtype=torch.float32, debug=False, data_path=None, logger=None,
            feature_type='R50_features', n_patients=30, top_k=-1
    ):
        self.data_path = data_path
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.center_slide = {}
        self.slide_label = {}
        self.slide_pt_pth = []
        self.feature_type = feature_type
        self.top_k = top_k
        label_map = {'itc': 0, 'macro': 1, 'micro': 2, 'negative': 3}
        for center in os.listdir(self.data_path):
            # c0, c1,..,c4
            if center not in self.center_slide:
                self.center_slide[center] = []
                for slide_label in os.listdir(os.path.join(self.data_path, center)):
                    slide_pths = f'{self.data_path}/{center}/{slide_label}/{feature_type}/pt_files'
                    for slide_pth in os.listdir(slide_pths):
                        self.slide_pt_pth.append(f'{slide_pths}/{slide_pth}')
                        self.center_slide[center].append(f'{slide_pths}/{slide_pth}')
                        self.slide_label[f'{slide_pths}/{slide_pth}'] = label_map[slide_label]
            else:
                continue


    def __len__(self):
        return len(self.slide_pt_pth)

    def __getitem__(self, idx, path=False):
        slide_pth = self.slide_pt_pth[idx]
        slide_label = self.slide_label[slide_pth]
        slide = torch.load(slide_pth)
        if len(slide.size()) == 1:
            if 'ViT' in self.feature_type:
                slide = slide.view(-1, 384)
            elif 'R50' in self.feature_type:
                slide = slide.view(-1, 1024)
        if self.top_k > 0:
            if slide.size(0) > self.top_k:
                idx = torch.randperm(slide.size(0))[:self.top_k]
                slide = slide[idx]
        if path:
            return slide, slide_label, slide_pth
        return slide, slide_label


class FedCamelyon17(Camelyon17Raw):
    def __init__(
        self,
        center,
        train_ratio=0.8,
        train: bool = True,
        pooled: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.float32,
        debug: bool = False,
        data_path: str = None,
        logger=None,
        require_image: bool = False,
        image_size: int = 256,
        deterministic=False,
        feature_type='R50_features',
        top_k=-1,
    ):
        """
        Cf class docstring
        """
        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            debug=debug,
            data_path=data_path,
            logger=logger,
            feature_type=feature_type,
            top_k=top_k
        )
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor()])
        self.center_slide = self.center_slide[center]  # list of slide pts
        self.slide_pt_pth = self.center_slide

        train_idx, test_idx = self.train_test_split(train_ratio, deterministic)
        if train:
            self.slide_pt_pth = [self.slide_pt_pth[idx] for idx in train_idx]
            self.slide_label = {slide: self.slide_label[slide] for slide in self.slide_pt_pth}
        else:
            self.slide_pt_pth = [self.slide_pt_pth[idx] for idx in test_idx]
            self.slide_label = {slide: self.slide_label[slide] for slide in self.slide_pt_pth}
        if require_image:
            self.indices_classes = self.get_idx_per_class()
    def train_test_split(self, train_ratio, deterministic=False):
        if deterministic:
            random.seed(0)
        n_slides = len(self.slide_pt_pth)
        n_train = int(n_slides * train_ratio)
        indices = list(range(n_slides))
        random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        return train_indices, test_indices

    def get_idx_per_class(self):
        idx_per_class = {}
        for slide_pth in self.slide_label:
            label = self.slide_label[slide_pth]
            if label not in idx_per_class:
                idx_per_class[label] = []
            slide_patches = torch.load(slide_pth)
            idx_per_class[label].append([slide_pth, slide_patches.size(0)])
        return idx_per_class

    def get_image(self, c, n_slide, n_patch=0):
        # Get N-patches from N-slides of class c
        idx_slide_n_patches = random.sample(self.indices_classes[c], n_slide)
        sample_slides = []
        for i in range(len(idx_slide_n_patches)):
            slide_pth, n_patches = idx_slide_n_patches[i]
            slide_name = slide_pth.split('/')[-1].split('.')[0]
            slide_meta = self.metadata.loc[self.metadata['slide_id']==slide_name]
            if len(slide_meta) < n_patch:
                # repeat sample if not enough patches
                idx_to_select = random.choices(range(len(slide_meta)), k=n_patch)
            else:
                idx_to_select = random.sample(range(len(slide_meta)), n_patch)
            slide_patches = []
            for idx in idx_to_select:
                stage_root = slide_meta.iloc[idx]['root']
                pat_id = slide_meta.iloc[idx]['patient_id']
                patch_id = slide_meta.iloc[idx]['patch_id']
                image_root = f'{self.data_path}/{stage_root}/{pat_id}/{patch_id}'
                image = Image.open(image_root)
                slide_patches.append(self.transform(image).unsqueeze(0))
            sample_slides.append(torch.cat(slide_patches, dim=0).unsqueeze(0))
        return torch.cat(sample_slides, dim=0)

if __name__=='__main__':
    train_dict = {'center_0': 0, 'center_1': 0, 'center_2': 0, 'center_3': 0, 'center_4': 0}
    test_dict = {'center_0': 0, 'center_1': 0, 'center_2': 0, 'center_3': 0, 'center_4': 0}
    print('====================')
    for center in train_dict:
        center_train = FedCamelyon17(center=center, train=True, data_path='/g/data/iq24/CAMELYON17_patches/centers/')
        train_dict[center] = len(center_train)
        center_test = FedCamelyon17(center=center, train=False, data_path='/g/data/iq24/CAMELYON17_patches/centers/')
        test_dict[center] = len(center_test)
    print(train_dict)
    print(test_dict)

    c0_train_dataset = FedCamelyon17(center='center_0', train=True, data_path='/g/data/iq24/CAMELYON17_patches/centers/')
    X, y = iter(DataLoader(c0_train_dataset, batch_size=1, shuffle=True, num_workers=0)).next()
    print(X.size(), y.size())
