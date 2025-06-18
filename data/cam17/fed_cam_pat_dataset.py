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


class Camelyon17RawPat(Dataset):
    def __init__(
            self, X_dtype=torch.float32, y_dtype=torch.float32, debug=False, data_path=None, logger=None,
            feature_type='R50_features', n_patients=30, top_k=-1
    ):
        self.data_path = data_path
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.center_slide = {}
        self.center_pat_slide = {}
        self.slide_label = {}
        self.slide_pt_pth = []
        self.feature_type = feature_type
        self.top_k = top_k
        label_map = {'itc': 0, 'macro': 1, 'micro': 2, 'negative': 3}
        for center in os.listdir(self.data_path):
            # c0, c1,..,c4
            if center not in self.center_slide:
                self.center_slide[center] = []
                self.center_pat_slide[center] = {}
                for slide_label in os.listdir(os.path.join(self.data_path, center)):
                    slide_pths = f'{self.data_path}/{center}/{slide_label}/{feature_type}/pt_files'
                    for slide_pth in os.listdir(slide_pths):
                        pat_name = '_'.join(slide_pth.split('_')[:2])
                        if pat_name not in self.center_pat_slide[center]:
                            self.center_pat_slide[center][pat_name] = []
                        self.slide_pt_pth.append(f'{slide_pths}/{slide_pth}')
                        self.center_slide[center].append(f'{slide_pths}/{slide_pth}')
                        self.slide_label[f'{slide_pths}/{slide_pth}'] = label_map[slide_label]
                        self.center_pat_slide[center][pat_name].append(f'{slide_pths}/{slide_pth}')
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


class FedCamelyon17Pat(Camelyon17RawPat):
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
        deterministic=True,
        feature_type='R50_features',
        top_k=-1
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
        self.center_pat_slide = self.center_pat_slide[center]
        self.slide_pt_pth = []
        train_idx, test_idx = self.train_test_split(train_ratio, deterministic)
        all_idx = train_idx if train else test_idx
        all_patients = list(self.center_pat_slide.keys())
        self.selected_patients = []
        for idx in all_idx:
            pat_name = all_patients[idx]
            pat_slides = self.center_pat_slide[pat_name]
            self.slide_pt_pth.extend(pat_slides)
            self.selected_patients.append(pat_name)

        self.slide_label = {slide: self.slide_label[slide] for slide in self.slide_pt_pth}
        print(f'Center {center}[train: {train}]')
        print(f'Number of patients: {len(self.selected_patients)}/{len(all_patients)}')
        print('Number of slides:', len(self.slide_pt_pth))
        idx_per_class = self.get_idx_per_class()
        for c in idx_per_class:
            print(f'Class {c}: {len(idx_per_class[c])} slides')

        if require_image:
            self.indices_classes = self.get_idx_per_class()
    def train_test_split(self, train_ratio, deterministic):
        if deterministic:
            random.seed(0)
        n_pat = len(self.center_pat_slide)
        n_train = int(n_pat * train_ratio)
        indices = list(range(n_pat))
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
            slide_pt_pth, n_patches = idx_slide_n_patches[i]
            slide_h5 = slide_pt_pth.replace("pt_files", "h5_files").replace(".pt", ".h5")
            slide_pth = slide_pt_pth.replace("CAMELYON17_patches", "CAMELYON17").replace(
                "R50_features/pt_files/", "").replace("pt", "tif")
            slide_patches = []

            with h5py.File(slide_h5, 'r') as hdf5_file:
                number_of_patches = len(hdf5_file["coords"])
                sampled_patch_idx = random.sample(range(number_of_patches), n_patch)
                for patch_idx in sampled_patch_idx:
                    coord = hdf5_file['coords'][patch_idx]
                    wsi = openslide.open_slide(slide_pth)
                    img = wsi.read_region(coord, 0, (256, 256)).convert('RGB')
                    img = self.transform(img)
                    slide_patches.append(img.unsqueeze(0))
                del wsi
            sample_slides.append(torch.cat(slide_patches, dim=0).unsqueeze(0))
        return torch.cat(sample_slides, dim=0)

if __name__=='__main__':
    train_dict = {'center_0': 0, 'center_1': 0, 'center_2': 0, 'center_3': 0, 'center_4': 0}
    test_dict = {'center_0': 0, 'center_1': 0, 'center_2': 0, 'center_3': 0, 'center_4': 0}
    print('====================')
    for center in train_dict:
        center_train = FedCamelyon17Pat(center=center, train=True, data_path='/g/data/iq24/CAMELYON17_patches/centers/')
        train_dict[center] = len(center_train)
        center_test = FedCamelyon17Pat(center=center, train=False, data_path='/g/data/iq24/CAMELYON17_patches/centers/')
        test_dict[center] = len(center_test)
    print(train_dict)
    print(test_dict)

    c0_train_dataset = FedCamelyon17Pat(center='center_0', train=True, data_path='/g/data/iq24/CAMELYON17_patches/centers/')
    X, y = iter(DataLoader(c0_train_dataset, batch_size=1, shuffle=True, num_workers=0)).next()
    print(X.size(), y.size())
