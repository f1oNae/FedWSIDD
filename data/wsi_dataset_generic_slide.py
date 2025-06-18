import os
import h5py
import torch
import numpy as np
import pandas as pd
import openslide
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def preprocess_df(df, label, target='grade'):
    if label == 'norm':
        df.loc[df.grade == 0, 'grade'] = -1
        df.loc[df.type == 'norm', 'grade'] = 0

    df = df[df[target] >= 0].copy()

    if label != 'both' and label != 'norm':
        df = df[df.type == label].copy()
    return df


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 pretrained=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        # self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 label_col=None,
                 dataset_name='IDH',
                 test_label='label',
                 format='svs',
                 ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            patient_strat (boolean): Whether the dataset contains patient info
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.dataset_name = dataset_name
        self.test_label = test_label
        self.format = format
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        self.slide_data = slide_data
        self.slide_cls_prep()

    def csv_preprocess(self):
        pass

    def slide_pth_preprocess(self):
        pass

    def computeposweight(self):
        pass

    def get_bag_sizes(self):
        pass

    def slide_cls_prep(self):
        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def return_splits(self,
                      csv_path=None,
                      test_folder=False,
                      preprocess=False,
                      num_class=2,
                      print_inf=False,
                      test_label='label',
                      transform=None,
                      balance=False,
                      format='svs',
                      top_k=10,
                      ):
        assert csv_path
        train_split = pd.read_csv(csv_path['train'], dtype=self.slide_data['slide_id'].dtype)
        val_split = pd.read_csv(csv_path['test'], dtype=self.slide_data['slide_id'].dtype)
        if test_label != 'label':
            label_dict = {'0': 0, '1': 1}
            train_split[test_label] = train_split[test_label].map(label_dict, na_action=None)
            val_split[test_label] = val_split[test_label].map(label_dict, na_action=None)
            train_split = preprocess_df(train_split, 'both', test_label)
            val_split = preprocess_df(val_split, 'both', test_label)
            print(train_split.groupby(test_label).count())
            if 'slide' in train_split.columns:
                print(len(train_split.slide.unique()), 'WSIs')
            print(val_split.groupby(test_label).count())
            if 'slide' in val_split.columns:
                print(len(val_split.slide.unique()), 'WSIs')
        train_split = Generic_ImageSplit(train_split,
                                         coord_root_dir=self.coord_root_dir,
                                         data_dir=self.data_dir,
                                         num_classes=num_class,
                                         label_dict=self.label_dict,
                                         dataset_name=self.dataset_name,
                                         test_label=test_label,
                                         transform=transform,
                                         balance=balance,
                                         format=format,
                                         top_k=top_k)
        val_split = Generic_ImageSplit(val_split,
                                       coord_root_dir=self.coord_root_dir,
                                       data_dir=self.data_dir,
                                       num_classes=num_class,
                                       label_dict=self.label_dict,
                                       dataset_name=self.dataset_name,
                                       test_label=test_label,
                                       transform=transform,
                                       balance=balance,
                                       format=format,
                                       top_k=top_k)
        if test_folder:
            assert isinstance(test_folder, str)
            csv_test = test_folder + '/test.csv'
            test_data = pd.read_csv(csv_test)
            test_split = Generic_ImageSplit(test_data,
                                            coord_root_dir=self.coord_root_dir,
                                            data_dir=test_folder,
                                            num_classes=num_class,
                                            label_dict=self.label_dict,
                                            dataset_name=self.dataset_name,
                                            test_label=test_label,
                                            transform=transform,
                                            balance=balance,
                                            format=format,
                                            top_k=top_k)
        else:
            test_split = None
        if preprocess:
            train_split.slide_pth_preprocess(stage='train')
            train_split.csv_preprocess()
            train_split.bag_mu, train_split.bag_max, train_split.bag_min = train_split.get_bag_sizes()
            print(f'Train mu {train_split.bag_mu} | min {train_split.bag_min} | max {train_split.bag_max}\n')
            val_split.slide_pth_preprocess(stage='val')
            val_split.csv_preprocess()
            val_split.bag_mu, val_split.bag_max, val_split.bag_min = val_split.get_bag_sizes()
            print(f'Val mu {val_split.bag_mu} | min {val_split.bag_min} | max {val_split.bag_max}\n')
            if test_split is not None:
                test_split.slide_pth_preprocess(stage='test')
                test_split.csv_preprocess()
                test_split.bag_mu, test_split.bag_max, test_split.bag_min = test_split.get_bag_sizes()
                print(f'Test mu {test_split.bag_mu} | min {test_split.bag_min} | max {test_split.bag_max}\n')
        if print_inf:
            print("Training on %d samples" % (len(train_split)))
            print("Validating on {} samples".format(len(val_split)))

            if test_split is not None:
                print("Testing on {} samples".format(len(test_split)))
        if test_split is None:
            return train_split, val_split, []
        return train_split, val_split, test_split


class Generic_MIL_ImageDataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 coord_root_dir,
                 data_dir,
                 **kwargs):
        super(Generic_MIL_ImageDataset, self).__init__(**kwargs)
        self.coord_root_dir = coord_root_dir
        self.data_dir = data_dir
        self.slide_lookup_dic = None

    def slide_pth_preprocess(self, stage='train'):
        lookup_dict = {}
        for i in range(len(self.slide_data)):
            slide_pth = self.slide_data.iloc[i][0]
            slide_pth_list = slide_pth.split('/')
            slide_id = slide_pth_list[-1]
            slide_pth_new = self.data_dir + f'/{stage}/pt_files/' + '{}.pt'.format(slide_id)
            lookup_dict[slide_id] = slide_pth_new
        self.slide_lookup_dic = lookup_dict

    def csv_preprocess(self, fix_number=3):
        for i in range(len(self.slide_data)):
            slide_pth = self.slide_data.iloc[i, 0]
            slide_pth_list = slide_pth.split('/')
            slide_id = slide_pth_list[-1]
            if self.slide_lookup_dic is not None:
                self.slide_data.iloc[i, 0] = self.slide_lookup_dic[slide_id]
            else:
                # TODO
                continue

    def computeposweight(self):
        pos_count = 0
        count_dict = {x: 0 for x in range(len(self.label_dict))}
        labels = []
        for item in range(len(self.bags_list)):
            slide_id = self.bags_list[item]
            bag_rows = self.slide_data[self.slide_data['slide_id'] == slide_id]
            cls_id = bag_rows.iloc[0]['label']  # all patches in a slide share the same label
            if isinstance(cls_id, str):
                cls_id = self.label_dict[cls_id]
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.slide_data) - pos_count) / pos_count), count_dict, labels

    def get_bag_sizes(self):
        bags = []
        for item in range(len(self.bags_list)):
            slide_id = self.bags_list[item]
            bag_rows = self.slide_data[self.slide_data['slide_id'] == slide_id]
            bags.append(len(bag_rows))
        return np.mean(bags), np.max(bags), np.min(bags)

    def __getitem__(self, idx):
        slide_id = self.bags_list[idx]
        bag_rows = self.slide_data[self.slide_data['slide_id'] == slide_id]
        label = bag_rows.iloc[0]['label']  # all patches in a slide share the same label
        if label in list(self.label_dict.keys()):
            label = self.label_dict[label]
        tiles = []
        # need .pt from patches folder
        slide_name = slide_id.split('/')[-1].split('.')[0]
        _slide_id_ = slide_id.replace('R50_features', 'patches')
        _slide_id_ = _slide_id_.replace('pt_files', '')
        coord_path = _slide_id_.replace(f'{slide_name}.pt', f'{slide_name}.h5')
        stage_label = '/'.join(slide_id.split('/')[5:7])

        slide_file_path = f'{self.data_dir}/{stage_label}/{slide_name}.{self.format}'
        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=coord_path, wsi=wsi,
                                     pretrained=True,
                                     custom_downsample=1,
                                     target_patch_size=-1)

        if len(dataset) > self.top_k:
            n_select = torch.randperm(len(dataset))[:self.top_k]
        else:
            n_select = range(len(dataset))
        for i in n_select:
            img_tensor, coord = dataset[i]
            tiles.append(img_tensor)
        tiles = torch.cat(tiles)
        if self.balance:  # TODO: need check
            sizes = tiles.size()
            n, c, h, w = sizes
            bag_feats = torch.zeros((self.bag_max, c, h, w), dtype=torch.float)
            bag_feats[:n, :] = tiles
            tiles = bag_feats
        del dataset
        return tiles, int(label)


class Generic_ImageSplit(Generic_MIL_ImageDataset):
    def __init__(self, slide_data, coord_root_dir,  data_dir=None, num_classes=2, label_dict={}, dataset_name=None,
                 test_label='label', transform=None, balance=False, format='svs', top_k=10):
        self.coord_root_dir = coord_root_dir
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.label_dict = label_dict
        self.dataset_name = dataset_name
        self.test_label = test_label
        self.bags_list = self.slide_data['slide_id'].unique().tolist()
        self.balance = balance
        self.format = format
        self.top_k = top_k
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        self.transform = transform
        print('total ', len(self.bags_list))
        self.pos_weight, self.count_dict, self.labels = self.computeposweight()

    def __len__(self):
        return len(self.bags_list)
