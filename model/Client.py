import torch
import random
from copy import deepcopy
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils.Get_model import define_model
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner, acmil_runner
from utils.data_utils import get_split_loader, CategoriesSampler
from utils.trainer_util import get_loss
from model.FedGen.Generator import DiversityLoss
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentBase:
    def __init__(self, args, global_model, logger, MIL_pool=None):
        self.args = deepcopy(args)
        if self.args.heter_model:
            assert MIL_pool is not None
            seed = 33 + self.args.rep + len(MIL_pool)
            random.seed(seed)
            self.args.mil_method = random.choice(MIL_pool)
        self.local_model = deepcopy(global_model) if global_model is not None else define_model(self.args)
        self.local_model_name = self.args.mil_method
        self.logger = logger
        self.device = device
        self.init_loss_fn()

    def init_dataset(self, local_train_ds, local_test_ds):
        self.train_dataset = local_train_ds
        self.test_dataset = local_test_ds
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()

    def label_counts(self, cls_idx=None):
        label_count_dct = {}
        for idx in range(len(self.train_dataset)):
            label = self.train_dataset.features_labels[idx]
            if label not in label_count_dct:
                label_count_dct[label] = 1
            else:
                label_count_dct[label] += 1
        if cls_idx is not None:
            return label_count_dct[cls_idx]
        return label_count_dct

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduce=False)
        self.diversity_loss = DiversityLoss(metric='l1')
        self.dist_loss = nn.MSELoss()
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.CE_loss = nn.CrossEntropyLoss()
        self.mil_loss = get_loss(self.args)
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")

    def get_train_loader(self):
        if 'frmil' in self.args.mil_method:
            train_sampler = CategoriesSampler(self.train_dataset.labels,
                                              n_batch=len(self.train_dataset.slide_data),
                                              n_cls=self.args.n_classes,
                                              n_per=1)
            train_loader = DataLoader(dataset=self.train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
        else:
            if 'IMAGE' in self.args.task:
                train_loader = get_split_loader(self.train_dataset, training=True, weighted=self.args.weighted_sample, load_wsi=True)
            else:
                train_loader = get_split_loader(self.train_dataset, training=True, weighted=self.args.weighted_sample)
        return train_loader
    def get_test_loader(self):
        if 'frmil' in self.args.mil_method:
            test_loader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            if 'IMAGE' in self.args.task:
                test_loader = get_split_loader(self.test_dataset,  load_wsi=True)
            else:
                test_loader = get_split_loader(self.test_dataset)
        return test_loader

    def get_local_model_weights(self):
        return self.local_model.state_dict()

    def get_local_data_weights(self):
        return len(self.train_dataset)

    def mil_run(self, model,
                data,
                label,
                loss_fn,
                return_lgt=False,
                return_feature=False,
                raw_image=False,
                aug_feature=None):
        if 'CLAM' in self.args.mil_method:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = clam_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image, aug_feature=aug_feature)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = clam_runner(self.args, model, data, label, loss_fn,
                                                          return_lgt=True, raw_image=raw_image, aug_feature=aug_feature)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = clam_runner(self.args, model, data, label, loss_fn,
                                                              return_feature=True, raw_image=raw_image, aug_feature=aug_feature)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = clam_runner(self.args, model, data, label, loss_fn, raw_image=raw_image, aug_feature=aug_feature)
                return loss, error, pred_prob
        elif 'TransMIL' in self.args.mil_method:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = transmil_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = transmil_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = transmil_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = transmil_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob
        elif 'ABMIL' in self.args.mil_method:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = abmil_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = abmil_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = abmil_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = abmil_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob
        elif 'ACMIL' in self.args.mil_method:
            loss, error, pred_prob = acmil_runner(model, data, label, loss_fn)
            return loss, error, pred_prob
        elif 'HIPT' in self.args.mil_method:
            loss, error, pred_prob = hipt_runner(model, data, label, loss_fn)
            return loss, error, pred_prob


    def local_test(self, model=None):
        if model is not None:
            model = self.turn_off_training(model)
        else:
            self.turn_off_training()
            model = self.local_model
        total_loss = 0.
        total_error = 0.
        all_probs = np.zeros((len(self.test_loader), self.args.n_classes))
        all_labels = np.zeros(len(self.test_loader))

        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            loss, error, Y_prob = self.mil_run(model, images, labels, self.mil_loss)

            total_loss += loss.item()
            total_error += error
            probs = Y_prob.detach().cpu().numpy()

            all_probs[batch_idx] = probs
            all_labels[batch_idx] = labels.item()

        total_loss /= len(self.test_loader)
        total_error /= len(self.test_loader)
        if self.args.n_classes == 2:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
        else:
            fpr = dict()
            tpr = dict()
            y_true_bin = label_binarize(all_labels, classes=list(range(self.args.n_classes)))
            for i in range(y_true_bin.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        return total_loss, total_error, fpr, tpr

    def local_test_proto(self, global_protos):
        self.turn_off_training()
        model = self.local_model
        total_correct = 0.
        total_loss = 0.
        for batch_idx, (images, labels) in enumerate(self.test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            if 'CLAM' in self.args.mil_method:
                loss, error, bag_feature = clam_runner(self.args,
                                          model,
                                          images,
                                          labels,
                                          self.mil_loss,
                                          return_feature=True)
            else:
                self.logger.error(f'{self.args.mil_method} not implemented')
                raise NotImplementedError

            # compute the dist between protos and global_protos
            a_large_num = 100
            dist = a_large_num * torch.ones(size=(images.shape[0], self.args.n_classes)).to(
                self.device)  # initialize a distance matrix
            for i in range(images.shape[0]):
                for j in range(self.args.n_classes):
                    if j in global_protos.keys():
                        d = self.dist_loss(bag_feature[i, :], global_protos[j][0])
                        dist[i, j] = d

            # prediction
            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)
            total_correct += torch.sum(torch.eq(pred_labels, labels)).item()

            total_loss += loss.item()
        total_loss /= len(self.test_loader)
        total_correct /= len(self.test_loader)
        return total_loss, total_correct

    def turn_on_training(self, model=None):
        if model is not None:
            model.to(self.device)
            model.train()
            return model
        else:
            self.local_model.to(self.device)
            self.local_model.train()

    def turn_off_training(self, model=None):
        if model is not None:
            model.to(self.device)
            model.eval()
            return model
        else:
            self.local_model.to(self.device)
            self.local_model.eval()

    def update_global_model(self, global_model):
        self.global_model_local = deepcopy(global_model)

    def update_local_model(self, local_model):
        self.local_model = deepcopy(local_model)

    def load_local_model(self, model):
        self.local_model.load_state_dict(model)

