import sys
from copy import deepcopy
from torch import nn
sys.path.insert(1, '/scratch/sz65/cc0395/WSI_prompt/')
import os
import numpy as np
import torch
import random
import torch.optim as optim
import math
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps
def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error
def compute_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).type(torch.float).mean().item() * 100.

def compute_accuracy_bce(logits, labels, thr=0.5):
    pred = torch.ge(logits, thr).float()
    return (pred == labels).type(torch.float).mean().item() * 100.
def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal
def eval_metric(oprob, label):

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, specificity, F1, auc
def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx  = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions, op_thres=None):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    if op_thres is not None:
        this_class_label[this_class_label>=op_thres] = 1
        this_class_label[this_class_label<op_thres] = 0
    else:
        this_class_label[this_class_label>=threshold_optimal] = 1
        this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='macro')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore, threshold_optimal

class Meter(object):

    def __init__(self):
        self.list = []
        self.labels = []
        self.preds = []

    def update(self, item):
        self.list.append(item)

    def update_gt(self, label, pred):
        self.labels.append(np.clip(label, 0, 1))
        self.preds.append(pred)

    def avg_test(self):
        return torch.tensor(np.array(five_scores(self.labels, self.preds)[0])) * 100.0

    def acc_auc(self, thres=None):
        accuracy, auc_value, precision, recall, fscore, thres_op = five_scores(self.labels, self.preds, op_thres=thres)
        acc = torch.tensor(np.array(accuracy)) * 100.0
        auc = torch.tensor(np.array(auc_value)) * 100.0
        return acc, auc, fscore, thres_op

    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

    def std(self):
        return torch.tensor(self.list).std() if len(self.list) else None

    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci

    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin or radius
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)

    def forward(self, instance, instance_bank, instance_label):
        # given a batch of instance, minimize the distance between the instance and the mean instance in the bank which has the same label
        # and maximize the distance between the instance and the mean instance in the bank which has different label
        # instance: [N, D]
        # instance_bank: [C, M, D]
        # instance_label: [N]
        # C: number of classes
        # M: number of instances per class in the memory bank
        # D: feature dimension
        # N: batch size
        # print(instance.size(), instance_bank.size(), instance_label.size())
        mean_instance_per_cls = torch.mean(instance_bank, dim=1).detach()  # [C, D]
        pos_mask = instance_label.unsqueeze(1) == torch.arange(mean_instance_per_cls.size(0)).to(mean_instance_per_cls.device).unsqueeze(0)  # [N, C]
        neg_mask = ~pos_mask
        pos_mean_instance = torch.matmul(pos_mask.float(), mean_instance_per_cls)  # [N, D]
        neg_mean_instance = torch.matmul(neg_mask.float(), mean_instance_per_cls)  # [N, D]
        pos_dist = torch.nn.functional.pairwise_distance(instance, pos_mean_instance)
        neg_dist = torch.nn.functional.pairwise_distance(instance, neg_mean_instance)
        # anchor_dot_contrast = torch.div(
        #     torch.matmul(instance, pos_mean_instance.T),
        #     0.7)
        # anchor_dot_contrast_neg = torch.div(
        #     torch.matmul(instance, neg_mean_instance.T),
        #     0.7)
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        # log_prob = logits - torch.log(torch.exp(anchor_dot_contrast+ anchor_dot_contrast_neg) + 1e-12)
        # loss = - log_prob

        # loss = torch.pow(pos_dist, 2) + torch.pow(torch.clamp(self.m - neg_dist, min=0.0), 2)
        # loss = loss.mean()

        loss = self.triplet(instance, pos_mean_instance, neg_mean_instance)
        return {'loss': loss, 'pos_dist': pos_dist.mean(), 'neg_dist': neg_dist.mean()}

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=512, device='cpu'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim    = feat_dim
        self.device      = device

        center_init = torch.zeros(self.num_classes, self.feat_dim).to(self.device)

        nn.init.xavier_uniform_(center_init)
        self.centers = nn.Parameter(center_init)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long() # should be long()
        classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask   = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

    def get_assignment(self, batch):
        alpha = 1.0
        norm_squared = torch.sum((batch.unsqueeze(1) - self.centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / alpha))
        power = float(alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self, batch):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.
        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

class FeatMag(nn.Module):

    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, feat_pos, feat_neg, w_scale=1.0):
        loss_act = self.margin - torch.norm(torch.mean(feat_pos, dim=1), p=2, dim=1)
        loss_act[loss_act < 0] = 0
        loss_bkg = torch.norm(torch.mean(feat_neg, dim=1), p=2, dim=1)

        loss_um = torch.mean((loss_act + loss_bkg) ** 2)
        return loss_um / w_scale
class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0 and group['weight_decay'] is not None:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def set_client_from_params(mdl, params):
    dict_param = deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length
    # print(mdl.state_dict().keys())
    # print('====================')
    # print(dict_param.keys())
    mdl.load_state_dict(dict_param, strict=False)
    return mdl

def get_loss(args):
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes=args.n_classes)
        loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'mag':
        loss_fn = FeatMag(margin=args.mag).cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def get_optim(args, model, alpha=None):
    params_gather = []
    mommen = args.reg if alpha is None else alpha + args.reg
    params_gather.append(
        {'params': filter(lambda p: p.requires_grad, model.parameters()),
         'lr': args.lr,
         'weight_decay': mommen}
    )
    opt = 'adamw' if 'TransMIL' in args.mil_method else args.opt
    print(f'Optimizer: {args.opt} for {args.mil_method}')
    if opt == "adam":
        optimizer = optim.Adam(params_gather)
    elif opt == 'adamw':
        optimizer = optim.AdamW(params_gather)
    elif opt == 'sgd':
        optimizer = optim.SGD(params_gather, momentum=mommen, nesterov=True)
    elif opt == 'radam':
        optimizer = RAdam(params_gather)
    else:
        raise NotImplementedError
    return optimizer

def random_pertube(model, rho):
    new_model = deepcopy(model)
    for p in new_model.parameters():
        gauss = torch.normal(mean=torch.zeros_like(p), std=1)
        if p.grad is None:
            p.grad = gauss
        else:
            p.grad.data.copy_(gauss.data)

    norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in new_model.parameters() if p.grad is not None]), p=2)

    with torch.no_grad():
        scale = rho / (norm + 1e-12)
        scale = torch.clamp(scale, max=1.0)
        for p in new_model.parameters():
            if p.grad is not None:
                e_w = 1.0 * p.grad * scale.to(p)
                p.add_(e_w)

    return new_model

