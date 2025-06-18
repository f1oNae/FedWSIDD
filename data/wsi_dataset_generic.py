import copy
import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.utils.data import Dataset
from scipy import stats


def preprocess_df(df, label, target='grade'):
    if label == 'norm':
        df.loc[df.grade == 0, 'grade'] = -1
        df.loc[df.type == 'norm', 'grade'] = 0

    df = df[df[target] >= 0].copy()

    if label != 'both' and label != 'norm':
        df = df[df.type == label].copy()
    return df
class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/ccrcc_clean.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict={},
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max',
                 dataset_name='IDH',
                 use_h5=False,
                 test_label='label',
                 top_k = 0
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
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.dataset_name = dataset_name
        self.use_f5 = use_h5
        self.test_label = test_label
        self.top_k = top_k
        if not label_col:
            label_col = 'label'
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        self.slide_data = slide_data
        self.slide_cls_prep()
        if self.patient_strat:
            self.patient_data_prep(patient_voting)
        if print_info:
            self.summarize()

    def csv_preprocess(self):
        pass

    def slide_pth_preprocess(self):
        pass

    def computeposweight(self):
        pass

    def get_bag_sizes(self):
        pass

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id']))  # get unique patients
        patient_labels = []

        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()  # get patient label (MIL convention)
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)

        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}
        # store ids corresponding each class at the patient or case level
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

    def slide_cls_prep(self):
        # store ids corresponding each class at the slide level
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            if self.patient_strat:
                print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def return_splits(self, from_id=True,
                      csv_path=None,
                      test_folder=False,
                      preprocess=False,
                      num_class=2,
                      print_inf=False,
                      test_label='label',
                      balance=False,
                      use_stain=False,
                      backbone='ResNet50',
                      top_k=20000):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir,
                                            num_classes=self.num_classes,
                                            label_dict=self.label_dict,
                                            dataset_name=self.dataset_name,
                                            use_h5=self.use_f5,
                                            test_label=test_label,
                                            balance=balance,
                                            use_stain=use_stain,
                                            backbone=backbone,
                                            top_k=top_k)
            else:
                train_split = None
            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir,
                                          num_classes=self.num_classes,
                                          label_dict=self.label_dict,
                                          dataset_name=self.dataset_name,
                                          use_h5=self.use_f5,
                                          test_label=test_label,
                                          balance=balance,
                                          use_stain=use_stain,
                                          backbone=backbone,
                                          top_k=top_k)
            else:
                val_split = None
            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir,
                                           num_classes=self.num_classes,
                                           label_dict=self.label_dict,
                                            dataset_name=self.dataset_name,
                                            use_h5=self.use_f5,
                                            test_label=test_label,
                                            balance=balance,
                                            use_stain=use_stain,
                                            backbone=backbone,
                                            top_k=top_k)
            else:
                test_split = None
        else:
            assert csv_path
            train_split = pd.read_csv(csv_path['train'], dtype=self.slide_data['slide_id'].dtype)
            val_split = pd.read_csv(csv_path['test'], dtype=self.slide_data['slide_id'].dtype)

            if test_label != 'label':
                label_dict = {'0': 0, '1': 1}
                train_split[test_label] = train_split[test_label].map(label_dict, na_action=None)
                val_split[test_label] = val_split[test_label].map(label_dict, na_action=None)
                train_split = preprocess_df(train_split, 'both', test_label)
                val_split = preprocess_df(val_split, 'both', test_label)
                if 'slide' in train_split.columns:
                    print(len(train_split.slide.unique()), 'WSIs')
                print(val_split.groupby(test_label).count())
                if 'slide' in val_split.columns:
                    print(len(val_split.slide.unique()), 'WSIs')
            train_split = Generic_Split(train_split, data_dir=self.data_dir,
                                            num_classes=num_class,
                                            label_dict=self.label_dict,
                                            dataset_name=self.dataset_name,
                                            use_h5=self.use_f5,
                                            test_label=test_label,
                                            balance=balance,
                                            use_stain=use_stain,
                                            backbone=backbone,
                                            top_k=top_k)
            val_split = Generic_Split(val_split, data_dir=self.data_dir,
                                            num_classes=num_class,
                                            label_dict=self.label_dict,
                                            dataset_name=self.dataset_name,
                                            use_h5=self.use_f5,
                                            test_label=test_label,
                                            balance=balance,
                                            use_stain=use_stain,
                                            backbone=backbone,
                                            top_k=top_k)
        if test_folder:
            assert isinstance(test_folder, str)
            csv_test = test_folder + '/test.csv'
            test_data = pd.read_csv(csv_test)
            test_split = Generic_Split(test_data, data_dir=test_folder,
                                       num_classes=num_class,
                                       label_dict=self.label_dict,
                                       dataset_name=self.dataset_name,
                                       use_h5=self.use_f5,
                                       test_label=test_label,
                                       balance=balance,
                                       use_stain=use_stain,
                                       backbone=backbone,
                                       top_k=top_k)
        else:
            test_split = None
        if preprocess:
            train_split.slide_pth_preprocess(stage='train')
            train_split.csv_preprocess()
            val_split.slide_pth_preprocess(stage='val')
            val_split.csv_preprocess()
            if test_split is not None:
                test_split.slide_pth_preprocess(stage='test')
                test_split.csv_preprocess()
        if print_inf:
            print("Training on %d samples"%(len(train_split)))
            print("Validating on {} samples".format(len(val_split)))

            if test_split is not None:
                print("Testing on {} samples".format(len(test_split)))
        train_split.bag_mu, train_split.bag_max, train_split.bag_min = train_split.get_bag_sizes()
        print(f'Train mu {train_split.bag_mu} | min {train_split.bag_min} | max {train_split.bag_max}\n')
        val_split.bag_mu, val_split.bag_max, val_split.bag_min = val_split.get_bag_sizes()
        print(f'Val mu {val_split.bag_mu} | min {val_split.bag_min} | max {val_split.bag_max}\n')
        if test_split is not None:
            test_split.bag_mu, test_split.bag_max, test_split.bag_min = test_split.get_bag_sizes()
            print(f'Test mu {test_split.bag_mu} | min {test_split.bag_min} | max {test_split.bag_max}\n')
        if test_split is None:
            return train_split, val_split, []
        return train_split, val_split, test_split


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    def __init__(self,
                 data_dir,
                 **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.slide_lookup_dic = None

    def slide_pth_preprocess(self, stage='train'):
        lookup_dict = {}
        for i in range(len(self.slide_data)):
            slide_pth = self.slide_data.iloc[i][0]
            slide_pth_list = slide_pth.split('/')
            slide_id = slide_pth_list[-1]
            # if stage == 'train':
            #     if self.use_f5:
            #         slide_pth_new = self.data_dir + '/Train/h5_files/' + '{}.h5'.format(slide_id)
            #     else:
            #         slide_pth_new = self.data_dir + '/Train/pt_files/' + '{}.pt'.format(slide_id)
            # else:
            #     if self.use_f5:
            #         slide_pth_new = self.data_dir + '/Test/h5_files/' + '{}.h5'.format(slide_id)
            #     else:
            #         slide_pth_new = self.data_dir + '/Test/pt_files/' + '{}.pt'.format(slide_id)
            lookup_dict[slide_id] = slide_pth
        self.slide_lookup_dic = lookup_dict

    def csv_preprocess(self, fix_number=3):
        '''
        :param fix_number: number of remained prefix
        :param pt_pth: prefix pth to pt files of each slide
        '''
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
        pos_count  = 0
        count_dict = {x: 0 for x in range(len(self.label_dict))}
        labels     = []
        label_idx = self.slide_data.columns.tolist()
        for item in range(len(self.slide_data)):
            cls_id    =  self.slide_data.iloc[item, label_idx.index(self.test_label)]
            cls_id = self.label_dict[cls_id]
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.slide_data)-pos_count)/pos_count), count_dict, labels

    def get_bag_sizes(self):
        bags = []
        for item in range(len(self.slide_data)):
            slide_pth = self.slide_data.iloc[item, 0]
            if self.use_f5:
                with h5py.File(slide_pth, 'r') as hdf5_file:
                    features = hdf5_file['features'][:]
                features = torch.from_numpy(features)
            else:
                if not os.path.exists(slide_pth):
                    print('%s not exist!!!!' % slide_pth)
                # print('Loading slide:', slide_pth)
                features = torch.load(slide_pth)
            if len(features.size()) < 2:
                ft_len = 1024 if 'R50' in slide_pth else 384
                features = features.view(-1, ft_len)
            num_insts = np.asarray(features).shape[0]
            bags.append(num_insts)
        return np.mean(bags),np.max(bags), np.min(bags)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        slide_pth = self.slide_data.iloc[idx, 0]
        label_idx = self.slide_data.columns.tolist()
        label = self.slide_data.iloc[idx, label_idx.index(self.test_label)]
        if label in list(self.label_dict.keys()):
            label = self.label_dict[label]
        if self.use_f5:
            with h5py.File(slide_pth, 'r') as hdf5_file:
                features = hdf5_file['features'][:]
            features = torch.from_numpy(features)
        else:
            if not os.path.exists(slide_pth):
                print('%s not exist!!!!'%slide_pth)
            features = torch.load(slide_pth)
            # print('Loading slide:', slide_pth)
            if len(features.size()) < 2:
                # print('slide_pth:', slide_pth)
                ft_len = 1024 if 'R50' in slide_pth else 384
                features = features.view(-1, ft_len)
        # print('Feature size ', features.size())
        if self.balance:
            num_inst  = features.size()[0]
            ndims     = features.size()[1]
            bag_feats = torch.zeros((self.bag_max, ndims), dtype=torch.float)
            bag_feats[:num_inst,:] = features
            features = bag_feats

        if self.use_stain:
            if self.dataset_name in ['CAM16','BRIGHT', 'IDH']:
                if 'R50' in slide_pth:
                    stain_feature_pth = slide_pth.replace('R50_features', 'pre_extracted_color_label')
                elif 'VIT' in slide_pth:
                    stain_feature_pth = slide_pth.replace('VIT_features', 'pre_extracted_color_label')
                else:
                    stain_feature_pth = slide_pth.replace('pre_extracted_feature', 'pre_extracted_color_label')

                # stain_feature_pth = stain_feature_pth.replace('pt', 'h5')
                # with h5py.File(stain_feature_pth, 'r') as hdf5_file:
                #     st_features = hdf5_file['features'][:]
                # stain_features = torch.from_numpy(st_features)
                # print('Loading stain features...', stain_feature_pth)
                stain_features = torch.load(stain_feature_pth)
            else:
                stain_feature_pth = slide_pth.replace('pre_extracted_feature', 'pre_extracted_color_label')
                stain_feature_pth = stain_feature_pth.replace(self.backbone, '')
                if os.path.exists(stain_feature_pth):
                    stain_feature_pth = stain_feature_pth.replace(self.backbone, '')
                    stain_features = torch.load(stain_feature_pth)
                else:
                    stain_feature_pth = slide_pth.replace('pre_extracted_feature', 'pre_extracted_color_feature')
                    stain_feature_pth = stain_feature_pth.replace(self.backbone, '')
                    stain_features = torch.load(stain_feature_pth)

            if self.balance:
                ######** stain features = stain labels [n_patch] **######
                num_inst  = stain_features.size()[0]
                bag_color_feats = torch.zeros((self.bag_max), dtype=torch.float)
                bag_color_feats[:num_inst] = stain_features
                stain_features = bag_color_feats
            if self.top_k > 0:
                if stain_features.size(0) > self.top_k:
                    idx = torch.randperm(stain_features.size(0))[:self.top_k]
                    stain_features = stain_features[idx]
                    features = features[idx]
            # print('Feature size ', features.size(), stain_features.size())
            feature_color = torch.zeros((features.size()[0], features.size()[1] + 1))
            feature_color[:, :-1] = features
            feature_color[:, -1] = stain_features
            # print('Feature color size ', feature_color.size())
            return feature_color, int(label)
        if self.top_k > 0:
            if features.size(0) > self.top_k:
                idx = torch.randperm(features.size(0))[:self.top_k]
                features = features[idx]
        return features, int(label)

class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data,
                 data_dir=None,
                 num_classes=2,
                 label_dict={},
                 dataset_name=[],
                 use_h5=False,
                 test_label='label',
                 balance=False,
                 use_stain=False,
                 backbone='ResNet50',
                 top_k=0):
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.label_dict = label_dict
        self.dataset_name = dataset_name
        self.use_f5 = use_h5
        self.test_label = test_label
        self.balance = balance
        self.use_stain = use_stain
        self.backbone = backbone
        self.top_k = top_k
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        print('total ', len(self.slide_data))
        self.pos_weight, self.count_dict, self.labels = self.computeposweight()

    def __len__(self):
        return len(self.slide_data)
