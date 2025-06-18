import os
import random
import time
from copy import deepcopy
# import PIL.Image
# import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
from PIL import Image
from skimage.color import rgb2hed, hed2rgb
import torch.nn.functional as F
import math
from itertools import islice
import collections
from torchvision.utils import save_image
from tqdm import tqdm
import h5py
# PIL.Image.MAX_IMAGE_PIXELS = 933120000


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path
def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)
    return "_".join(aug_list)

class Normalize():
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
def diffaug(args, mean, std, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type

    normalize = Normalize(mean=mean, std=std, device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup_net == 'cut':
        aug_type = remove_aug(aug_type, 'cutout')
    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand

def img_denormlaize(img, dataname='imagenet'):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = [0.485, 0.456, 0.406]# MEANS[dataname]
    std = [0.229, 0.224, 0.225]#STDS[dataname]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean

def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataname=dataname)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    print('Image saved @ ', save_dir)
    save_image(img.cpu(), save_dir, nrow=nrow)

def preprocess_df(df, label, target='grade'):
    if label == 'norm':
        df.loc[df.grade == 0, 'grade'] = -1
        df.loc[df.type == 'norm', 'grade'] = 0

    df = df[df[target] >= 0].copy()

    if label != 'both' and label != 'norm':
        df = df[df.type == label].copy()
    cols = df.columns.tolist()
    cols_target = cols.index(target)
    if cols_target != 1:
        old = cols[1]
        cols[1] = target
        cols[cols_target] = old
        df = df[cols]
    return df

def preprocess_df_topatch(df, img_root, read_h5=False):
    patch_df = {'image_id':[], 'grade':[]}
    for i in range(len(df)):
        slide_path = df.iloc[i]['slide_id']
        slide_id = slide_path.split('/')[-1]
        slide_label = df.iloc[i]['label']
        slide_path = os.path.join(img_root, slide_id)
        if read_h5:
            with h5py.File(slide_path+'.h5', 'r') as hdf5_file:
                print(hdf5_file.keys() )
                for i in range(len(hdf5_file['imgs'])):
                    print('progress: {}/{} of label [{}]'.format(i, len(hdf5_file['imgs']), slide_id))
                    patch_df['image_id'].append(hdf5_file['imgs'][i])
                    patch_df['grade'].append(slide_label)

        else:
            patch_dir = os.listdir(slide_path)
            for patch_id in patch_dir:
                if patch_id.split('.')[-1] != 'jpg':
                    continue
                patch_df['image_id'].append(os.path.join(slide_path, patch_id))
                patch_df['grade'].append(slide_label)
    patch_df = pd.DataFrame.from_dict(patch_df)
    return patch_df

class TimeStamp():
    def __init__(self, print_log=True):
        self.prev = time.time()
        self.print_log = print_log
        self.times = {}

    def set(self):
        self.prev = time.time()

    def flush(self):
        if self.print_log:
            print("\n=========Summary=========")
            for key in self.times.keys():
                times = np.array(self.times[key])
                print(
                    f"{key}: {times.sum():.4f}s (avg {times.mean():.4f}s, std {times.std():.4f}, count {len(times)})"
                )
                self.times[key] = []

    def stamp(self, name=''):
        if self.print_log:
            spent = time.time() - self.prev
            # print(f"{name}: {spent:.4f}s")
            if name in self.times.keys():
                self.times[name].append(spent)
            else:
                self.times[name] = [spent]
            self.set()


class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return len(self.sampler)

class ClassBatchSampler(object):
    """Intra-class batch sampler
    """
    def __init__(self, cls_idx, batch_size, drop_last=True):
        self.samplers = []
        # per class indices
        for indices in cls_idx:
            n_ex = len(indices)
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                          batch_size=min(n_ex, batch_size),
                                                          drop_last=drop_last)
            self.samplers.append(iter(_RepeatSampler(batch_sampler)))

    def __iter__(self):
        while True:
            for sampler in self.samplers:
                yield next(sampler)

    def __len__(self):
        return len(self.samplers)

def sample_with_repetition(lst, n):
    if n >= len(lst):
        return np.random.choice(lst * (n // len(lst) + 1), n)
    else:
        return np.random.choice(lst, n)

def collate_MIL(batch):
    if len(batch[0]) == 2: # load feature, label
        img = torch.cat([item[0] for item in batch], dim=0)
        label = []
        for item in batch:
            if isinstance(item[1], int):
                label.append(item[1])
            else:
                label.append(item[1].tolist())
        label = torch.LongTensor(label)
        # label = torch.LongTensor([item[1].tolist() for item in batch])
        return [img, label]
    elif len(batch[0]) == 3: # feature, label, slide_id
        ft1 = torch.cat([item[0] for item in batch], dim=0)
        label = torch.LongTensor([item[1] for item in batch])
        slide_pth = [item[2] for item in batch]
        return [ft1, label, slide_pth]
    elif len(batch[0]) == 4: # feature, colour label, slide_id
        ft1 = torch.cat([item[0] for item in batch], dim=0)
        ft2 = torch.cat([item[1] for item in batch], dim=0)
        label = torch.LongTensor([item[2] for item in batch])
        return [ft1, ft2, label]

def collate_WSI(batch):
    img = [item[0] for item in batch]
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_MIL_global(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    global_ft = torch.cat([item[3] for item in batch], dim=0)
    return [img, label, global_ft]


def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]


def get_simple_loader(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                        collate_fn=collate_MIL, **kwargs)
    return loader


def get_split_loader(split_dataset, training=False, testing=False, weighted=False, ddp=False, load_wsi=False):
    """
        return either the validation loader or training loader
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if ddp:
                train_sampler = torch.utils.data.distributed.DistributedSampler(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=train_sampler,
                                    collate_fn=collate_MIL, **kwargs)
            elif weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_MIL, **kwargs)
            elif load_wsi:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_WSI, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_MIL, **kwargs)
        else:
            if load_wsi:
                loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset),
                                    collate_fn=collate_WSI, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset),
                                    collate_fn=collate_MIL, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids), collate_fn=collate_MIL,
                            **kwargs)

    return loader

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """Multi epochs data loader
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()  # Init iterator and sampler once

        self.convert = None
        if self.dataset[0][0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

        if self.dataset[0][0].device == torch.device('cpu'):
            self.device = 'cpu'
        else:
            self.device = 'cuda'

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for i in range(len(self)):
            data, target = next(self.iterator)
            if self.convert != None:
                data = self.convert(data)
            yield data, target

class ClassDataLoader(MultiEpochsDataLoader):
    """Basic class loader (might be slow for processing data)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nclass = self.dataset.nclass
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(self.dataset)):
            self.cls_idx[self.dataset.df.iloc[i,1]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)

        self.cls_targets = torch.tensor([np.ones(self.batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device='cuda')
    def class_sample(self, c, ipc=-1, preset_idx=None):
        if ipc > 0:
            if preset_idx is not None:
                indices = preset_idx[:ipc]
            else:
                if len(self.cls_idx[c]) < ipc:
                    patch_idx = self.cls_idx[c]
                    indices = sample_with_repetition(patch_idx, ipc)
                else:
                    indices = self.cls_idx[c][:ipc]
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.dataset[i][0] for i in indices])
        target = torch.tensor([self.dataset[i][1] for i in indices])
        # target = torch.tensor([self.dataset.targets[i] for i in indices])
        return data.cuda(), target.cuda()

    def sample(self):
        data, target = next(self.iterator)
        if self.convert != None:
            data = self.convert(data)

        return data.cuda(), target.cuda()

class ClassDataset():
    """Class loader with data
    """
    def __init__(self, dataset, batch_size, drop_last=False, device='cuda'):
        self.device = device
        self.batch_size = batch_size
        self.dataset = dataset
        start = time.time()
        self.targets = [torch.tensor(dataset[i][1]).to(device) for i in range(len(dataset))] # [n_slides, 1]
        self.nclass = dataset.num_classes
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.cls_targets = torch.tensor([[c] * batch_size for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device=self.device)
    def class_sample(self, c, ppc=-1, spc=-1, preset_idx=None):
        if spc > 0:
            # if spc > 0, return :spc slides idx from each class
            indices = self.cls_idx[c][:spc]
        else:
            # return all slides idx from each class
            indices = self.cls_idx[c]
        if preset_idx is not None:
            indices = preset_idx
        if ppc > 0:
            data_list = []
            for i in indices:
                slide_data = self.dataset[i][0].to(self.device)
                patch_idx = [_ for _ in range(slide_data.size()[0])]
                patch_idx = sample_with_repetition(patch_idx, ppc)
                data_list.append(slide_data[patch_idx])
            data = torch.stack(data_list)
        else:
            data = torch.stack([self.data[i] for i in indices]).to(self.device)
        if self.cls_targets[c].size()[0] < data.size()[0]:
            cls_targets = self.cls_targets[c].repeat(data.size()[0])
        else:
            cls_targets = self.cls_targets[c]
        return data, cls_targets
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.dataset[idx][0]
        label = self.targets[idx]
        return features, int(label)

def patch_select(data, ipc, nclass=2, method='herd'):
    model = models.resnet18()
    model.load_state_dict(torch.load('resnet18-f37072fd.pth'))
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=nclass, bias=True)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                                  padding=(3, 3),
                                  bias=False)
    model.load_state_dict(torch.load('results/IDH_patch_r18_base/model_4.pt')['model'])
    new_classifier = nn.Sequential(*list(model.children())[:-1])
    new_classifier = new_classifier.cuda()
    features = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(range(len(data))):
            input = data[i].cuda()
            feat = new_classifier(input.unsqueeze(0))
            feat = feat.reshape(feat.size(0), -1)
            features.append(feat)

    features = torch.cat(features).squeeze()
    print("Feature shape: ", features.shape)
    indices_slct = []
    indices_full = torch.arange(len(features))
    # for c in range(nclass):
    feature_c = features
    indices_c = indices_full

    feature_mean = feature_c.mean(0, keepdim=True)
    current_sum = torch.zeros_like(feature_mean)

    cur_indices = []
    for k in range(ipc):
        target = (k + 1) * feature_mean - current_sum
        dist = torch.norm(target - feature_c, dim=1)
        indices_sorted = torch.argsort(dist, descending=False)

        # We can make this faster by reducing feature matrix
        for idx in indices_sorted:
            idx = idx.item()
            if idx not in cur_indices:
                cur_indices.append(idx)
                break
        current_sum += feature_c[idx]

    indices_slct.append(indices_c[cur_indices])
    return indices_slct

class ClassMemDataLoader():
    """Class loader with data on GPUs
    """
    def __init__(self, dataset, batch_size, drop_last=False, device='cuda'):
        self.device = device
        self.batch_size = batch_size

        self.dataset = dataset
        self.data = [d[0].to(device) for d in dataset]  # uint8 data
        self.targets = torch.tensor(dataset.df.iloc[:, 1].to_list(), dtype=torch.long, device=device)

        sampler = torch.utils.data.SubsetRandomSampler([i for i in range(len(dataset))])
        self.batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                           batch_size=batch_size,
                                                           drop_last=drop_last)
        self.iterator = iter(_RepeatSampler(self.batch_sampler))

        self.nclass = dataset.nclass
        if self.nclass < 2:
            lab = self.targets[0]
            self.cls_idx = [[i for i in range(len(dataset))]]
        else:
            self.cls_idx = [[] for _ in range(self.nclass)]
            for i in range(len(dataset)):
                self.cls_idx[self.targets[i]].append(i)
        self.class_sampler = ClassBatchSampler(self.cls_idx, self.batch_size, drop_last=True)
        self.cls_targets = torch.tensor([np.ones(batch_size) * c for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device=self.device)

        self.convert = None
        if self.data[0].dtype == torch.uint8:
            self.convert = transforms.ConvertImageDtype(torch.float)

    def combine_new_data(self, new_dataset):
        self.data += [new_dataset[0].to(self.device)]
        self.targets = torch.cat([self.targets, new_dataset[1]])


    def class_sample(self, c, ipc=-1, preset_idx=None):
        # print(self.cls_idx[c][:ipc])
        if ipc > 0:
            if preset_idx is not None:
                indices = preset_idx[:ipc]
            else:
                if len(self.cls_idx[c])>0 and len(self.cls_idx[c]) < ipc:
                    patch_idx = self.cls_idx[c]
                    indices = sample_with_repetition(patch_idx, ipc)
                else:
                    indices = self.cls_idx[c][:ipc]
            # data_c = torch.stack([self.data[i] for i in self.cls_idx[c]])
            # indices = patch_select(data_c, ipc)
            # indices = torch.cat(indices).tolist()
        else:
            indices = next(self.class_sampler.samplers[c])

        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)
        # print('Selected index ', indices)
        # print(self.targets[indices])
        return data, self.cls_targets[c]

    def sample(self):
        indices = next(self.iterator)
        data = torch.stack([self.data[i] for i in indices])
        if self.convert != None:
            data = self.convert(data)
        target = self.targets[indices]

        return data, target

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            data, target = self.sample()
            yield data, target

class ClassMemDataset():
    """Class loader with data on memory
    """
    def __init__(self, dataset, batch_size, drop_last=False, device='cuda'):
        self.device = device
        self.batch_size = batch_size
        self.dataset = dataset
        self.data = [dataset[i][0] for i in range(len(dataset))] # [n_slides, n_patch, ft_size]
        self.targets = [torch.tensor(dataset[i][1]) for i in range(len(dataset))] # [n_slides, 1]
        self.nclass = dataset.num_classes
        self.cls_idx = [[] for _ in range(self.nclass)]
        for i in range(len(dataset)):
            self.cls_idx[self.targets[i]].append(i)
        self.cls_targets = torch.tensor([[c] * batch_size for c in range(self.nclass)],
                                        dtype=torch.long,
                                        requires_grad=False,
                                        device=self.device)
    def class_sample(self, c, ppc=-1, spc=-1):
        if spc > 0:
            cls_idxlist = self.cls_idx[c]
            if len(cls_idxlist) < spc:
                indices = sample_with_repetition(cls_idxlist, spc)
            else:
                indices = cls_idxlist[:spc]
        else:
            indices = self.cls_idx[c]
        if ppc > 0:
            data_list = []
            for i in indices:
                # patch_idx = torch.randperm(self.data[i].size()[0])[:ppc]
                # patch_idx = [_ for _ in range(self.data[i].size()[0])]
                # patch_idx = sample_with_repetition(patch_idx, ppc)
                # unif = torch.ones(self.data[i].size()[0])
                # patch_idx = unif.multinomial(ppc, replacement=True)
                patch_idx = np.random.choice(self.data[i].size()[0],ppc, replace=True)
                data_list.append(self.data[i][patch_idx])
            data = torch.stack(data_list)
        else:
            data = torch.stack([self.data[i] for i in indices])
        if self.cls_targets[c].size()[0] < data.size()[0]:
            cls_targets = self.cls_targets[c].repeat(data.size()[0])
        else:
            cls_targets = self.cls_targets[c]
        return data.to(self.device), cls_targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.targets[idx]
        return features, int(label)

# class HistoPatchData(torch.utils.data.Dataset):
#     def __init__(self, df, T, path, target, data_name, subsample=-1, gray=False, mock=False, unit='image_id', nclass=-1):
#         self.path = path
#         self.df = df
#         self.T = T
#         self.target = target
#         self.targets = self.df[self.target].to_list()
#         self.subsample = subsample
#         self.mock = mock
#         self.gray = gray
#         self.unit = unit
#         self.data_name = data_name
#         self.label_dict = {'IDH':{'MU':0, 'WT':1}}
#         self.nclass = len(self.df[self.target].unique()) if nclass < 0 else nclass
#         print('=> Loading dataset..', self.data_name)
#
#         print(f'Loaded {len(self.df)} images')
#
#         im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#         self.transform = transforms.Compose(
#                 [
#                     transforms.ToTensor(),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.RandomVerticalFlip(p=0.5),
#                     transforms.ColorJitter(),
#                     transforms.Normalize(mean=im_mean, std=im_std),
#                     transforms.Resize((224, 224)),  # (256, 256)
#                 ]
#             )
#
#     def process_image(self, image_path):
#         img_name = image_path.split('/')[-1]
#         return img_name
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, index):
#         # print('index is ', index)
#         entry = self.df.iloc[index]
#         image_id = entry[0]
#         if self.data_name == 'utp':
#             image_id = os.path.join(self.path, self.unit, image_id)#entry.top_label_name
#         elif self.data_name == 'CAM16':
#             image_id = os.path.join(self.path, entry[2], image_id)
#         else:
#             image_id = os.path.join(self.path, image_id)
#
#         if self.mock:
#             C = 1 if self.gray else 3
#             img = np.random.randint(0, 255, (224, 224, C)).astype(np.uint8)
#
#         else:
#             img = cv2.imread(image_id)
#             # print('Original image shape ', img.shape)
#             if self.subsample != -1:
#                 w = img.shape[0]
#                 while w//2 > self.subsample:
#                     img = cv2.resize(img, (w//2, w//2))
#                     w = w//2
#                 img = cv2.resize(img, (self.subsample, self.subsample))
#
#             if self.gray:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                 img = np.expand_dims(img, axis=2)
#             else:
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         if self.T is not None:
#             img = self.T(img)
#         # img_pil = PIL.Image.open(image_id)
#         # img = self.transform(img_pil)
#
#         return img, entry[self.target]#self.label_dict[self.data_name][entry[self.target]]

def get_simple_loader_global(dataset, batch_size=1, num_workers=1):
    kwargs = {'num_workers': 4, 'pin_memory': False, 'num_workers': num_workers} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler.SequentialSampler(dataset),
                        collate_fn=collate_MIL_global, **kwargs)
    return loader


def get_split_loader_global(split_dataset, training=False, testing=False, weighted=False):
    """
        return either the validation loader or training loader
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler=WeightedRandomSampler(weights, len(weights)),
                                    collate_fn=collate_MIL_global, **kwargs)
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler=RandomSampler(split_dataset),
                                    collate_fn=collate_MIL_global, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler=SequentialSampler(split_dataset),
                                collate_fn=collate_MIL_global, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset) * 0.1)), replace=False)
        loader = DataLoader(split_dataset, batch_size=1, sampler=SubsetSequentialSampler(ids),
                            collate_fn=collate_MIL_global,
                            **kwargs)

    return loader


def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9,
                              weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits=5,
                   seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)

    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []

        if custom_test_ids is not None:  # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)  # all indices of this class
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)  # validation ids

            remaining_ids = np.setdiff1d(possible_indices, val_ids)  # indices of this class left after validation
            all_val_ids.extend(val_ids)

            if custom_test_ids is None:  # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)

            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)


def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    weight_per_class = [N / len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.getlabel(idx)
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Logger():
    def __init__(self, path, ddp=False):
        self.logger = open(os.path.join(path, 'log.txt'), 'w')
        self.ddp = ddp

    def __call__(self, string, end='\n', print_=True):
        if print_:
            if not self.ddp:
                print("{}".format(string), end=end)
            elif os.environ['LOCAL_RANK'] == 0:
                print("{}".format(string), end=end)
        if end == '\n':
            self.logger.write('{}\n'.format(string))
        else:
            self.logger.write('{} '.format(string))
        self.logger.flush()

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        if torch.is_tensor(images):
            self.images = images.detach().cpu().float()
        else:
            self.images = images
        self.targets = labels.detach().cpu()
        unique_labels =  torch.unique(self.targets)
        self.nclass = unique_labels.size()[0]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]

def Hed_Aug(img):
    img = np.array(img)
    Hed = rgb2hed(img)
    H = Hed[..., [0]]
    E = Hed[..., [1]]
    D = Hed[..., [2]]

    alpha1 = np.clip(random.random(), a_min=0.9, a_max=1)
    beta1 = np.clip(random.random(), a_min=0, a_max=0.01)

    alpha2 = np.clip(random.random(), a_min=0.9, a_max=1)
    beta2 = np.clip(random.random(), a_min=0, a_max=0.01)

    alpha3 = np.clip(random.random(), a_min=0.9, a_max=1)
    beta3 = np.clip(random.random(), a_min=0, a_max=0.01)

    H = H * alpha1 + beta1
    E = E * alpha2 + beta2
    D = D * alpha3 + beta3

    Hed_cat = np.concatenate((H, E, D), axis=-1)
    Hed_cat = hed2rgb(Hed_cat)
    Hed_cat = np.clip(Hed_cat, a_min=0, a_max=1)
    Hed_cat = np.uint8(Hed_cat * 255)
    return Hed_cat

class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indices, e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

def save_syn_img(image_syns, results_dir, iter=0, client_idx=None):
    STDS = [0.229, 0.224, 0.225]
    MEANS = [0.485, 0.456, 0.406]
    if client_idx is None:
        # iterate over clients
        for i, local_syn_images in enumerate(image_syns):
            save_client = os.path.join(results_dir, f'client{i}/{iter}')
            if not os.path.exists(save_client):
                os.makedirs(save_client)
            for slide_i in range(len(local_syn_images)):
                save_name = os.path.join(save_client, f'CLAM_DM_slide_{slide_i}.png')
                image_syn_vis = deepcopy(local_syn_images[slide_i].detach().cpu())
                # for i in range(image_syn_vis.size(1)):  # Iterate over the second dimension (channels)
                #     image_syn_vis[:, i, :, :] = image_syn_vis[:, i, :, :] * STDS[i] + MEANS[i]
                # image_syn_vis[image_syn_vis < 0] = 0.0
                # image_syn_vis[image_syn_vis > 1] = 1.0
                save_image(image_syn_vis, save_name, nrow=5)
    else:
        assert client_idx is not None
        save_client = os.path.join(results_dir, f'client{client_idx}/{iter}')
        if not os.path.exists(save_client):
            os.makedirs(save_client)
        for slide_i in range(len(image_syns)):
            save_name = os.path.join(save_client, f'CLAM_DM_slide_{slide_i}.png')
            image_syn_vis = deepcopy(image_syns[slide_i].detach().cpu())
            for i in range(image_syn_vis.size(1)):  # Iterate over the second dimension (channels)
                image_syn_vis[:, i, :, :] = image_syn_vis[:, i, :, :] * STDS[i] + MEANS[i]
            image_syn_vis[image_syn_vis < 0] = 0.0
            image_syn_vis[image_syn_vis > 1] = 1.0
            save_image(image_syn_vis, save_name, nrow=5)