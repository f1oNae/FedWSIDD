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

def add_root(df, fpth):
    n_rows = df.shape[0]
    root_data_path = [fpth] * n_rows
    # add extra coloumn to the dataframe
    df.insert(0, 'root', root_data_path)
    return df

class TCGAIDHRaw(Dataset):
    def __init__(
        self, X_dtype=torch.float32, y_dtype=torch.float32, debug=False, data_path=None, logger=None, feature_type='R50_features', n_patients=30, top_k=-1
    ):
        self.data_path = data_path
        self.metadata_pat = pd.read_excel('/scratch/iq24/cc0395/FedDDHist/data/tcga_idh/TCGA_Patient_List.xlsx', engine='openpyxl')
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.center_pat = {}
        self.center_slide = {}
        self.slide_label = {}
        self.slide_pt_pth = []
        self.top_k = top_k
        label_map = {'WT': 0, 'MU': 1}
        # get metadata from csv
        train_csv = add_root(pd.read_csv(os.path.join(self.data_path,'train_patch.csv')), '_Train_All')
        test_csv = add_root(pd.read_csv(os.path.join(self.data_path,'test_patch.csv')), '_Test_All')
        valid_csv = add_root(pd.read_csv(os.path.join(self.data_path,'valid_patch.csv')), '_Valid_All')
        self.metadata = pd.concat([train_csv, test_csv, valid_csv])
        # get slide_pth from data_path
        train_data = [os.path.join(self.data_path, f'Train/{feature_type}/pt_files', pat_pth) for pat_pth in os.listdir(os.path.join(self.data_path, f'Train/{feature_type}/pt_files'))]
        test_data = [os.path.join(self.data_path, f'Test/{feature_type}/pt_files', pat_pth) for pat_pth in os.listdir(os.path.join(self.data_path, f'Test/{feature_type}/pt_files'))]
        valid_data = [os.path.join(self.data_path, f'Valid/{feature_type}/pt_files', pat_pth) for pat_pth in os.listdir(os.path.join(self.data_path, f'Valid/{feature_type}/pt_files'))]
        all_data = train_data + test_data + valid_data
        # map slide to centers
        for center in self.metadata_pat['Unnamed: 2'].unique():
            if center != 'Tissue source site':
                pat_per_center = self.metadata_pat.loc[self.metadata_pat['Unnamed: 2']==center]['List of Cases'].tolist()
                if len(pat_per_center) >= n_patients:
                    self.center_pat[center] = pat_per_center
        # map patients to slides
        for center in self.center_pat:
            for pat in self.center_pat[center]:
                pat_slides = [slide for slide in all_data if pat in slide]
                if len(pat_slides) == 0:
                    # print(f'{pat} not found')
                    # logger.info(f'{pat} not found')
                    continue
                if center not in self.center_slide:
                    self.center_slide[center] = pat_slides
                else:
                    self.center_slide[center] += pat_slides
            # print(f'{center}: {len(self.center_slide[center])}')
        # map slides to labels
        for center in self.center_slide:
            for slide in self.center_slide[center]:
                self.slide_pt_pth.append(slide)
                slide_name = slide.split('/')[-1].split('.')[0]
                self.slide_label[slide] = label_map[self.metadata.loc[self.metadata['slide_id']==slide_name]['label'].tolist()[0]]

    def __len__(self):
        return len(self.slide_pt_pth)

    def __getitem__(self, idx, path=False):
        slide_pth = self.slide_pt_pth[idx]
        slide_label = self.slide_label[slide_pth]
        slide = torch.load(slide_pth)
        if self.top_k > 0:
            if slide.size(0) > self.top_k:
                idx = torch.randperm(slide.size(0))[:self.top_k]
                slide = slide[idx]
        if path:
            return slide, slide_label, slide_pth
        return slide, slide_label

class FedTCGAIDH(TCGAIDHRaw):
    def __init__(self,
                 center,
                 train_ratio=0.8,
                 train=True,
                 X_dtype=torch.float32,
                 y_dtype=torch.float32,
                 debug=False,
                 data_path=None,
                 logger=None,
                 n_patients=30,
                 require_image: bool = False,
                 image_size: int = 256,
                 deterministic=False,
                 feature_type='R50_features',
                 top_k=-1
                 ):
        super(FedTCGAIDH, self).__init__(X_dtype, y_dtype, debug, data_path, logger, feature_type, n_patients, top_k=top_k)
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor()])
        self.center_slide = self.center_slide[center] #list of slide pts
        self.slide_label = {slide: self.slide_label[slide] for slide in self.center_slide}
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

    def get_numer_instances_per_class(self):
        n_instances = [0, 0]
        for slide_pth in self.slide_label:
            label = self.slide_label[slide_pth]
            n_instances[label] += 1
        return n_instances

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
    dataset = TCGAIDHRaw(data_path='/g/data/iq24/IDH')
    slide, slide_label = dataset[0]
    print(slide.size(), slide_label)
    center = 'Cedars Sinai'
    # # sample_real = dataset.get_image(0, 1, 5)
    # # print(sample_real.size())
    train_dataset = FedTCGAIDH(center=center, train=True, data_path='/g/data/iq24/IDH', deterministic=True)
    test_dataset = FedTCGAIDH(center=center, train=False, data_path='/g/data/iq24/IDH', deterministic=True)
    print(len(train_dataset), len(test_dataset))

    indices_class = [[] for c in range(2)]
    labels_all = [train_dataset[i][1] for i in range(len(train_dataset))]
    for idx, lab in enumerate(labels_all):
        lab_item = int(lab.item()) if isinstance(lab, torch.Tensor) else int(lab)
        indices_class[lab_item].append(idx)
    print(len(indices_class[0]), len(indices_class[1]))

    indices_class = [[] for c in range(2)]
    labels_all = [test_dataset[i][1] for i in range(len(test_dataset))]
    for idx, lab in enumerate(labels_all):
        lab_item = int(lab.item()) if isinstance(lab, torch.Tensor) else int(lab)
        indices_class[lab_item].append(idx)
    print(len(indices_class[0]), len(indices_class[1]))
