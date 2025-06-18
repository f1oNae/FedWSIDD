from copy import deepcopy
import sys
sys.path.insert(0, '/scratch/iq24/cc0395/FedDDHist')
import torch
import os
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Attention Network without Gating (2 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
args:
    gate: whether to use gated attention network
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
    instance_loss_fn: loss function to supervise instance-level training
    subtyping: whether it's a subtyping problem
"""


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dfp_dict=None, raw_image=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                          "small": [1024, 512, 256], "big": [2048, 512, 384]}
        size = self.size_dict[size_arg]
        self.size = size
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.dfp = dfp_dict
        self.embed_length = size[0]
        self.prompt_disrim = False
        self.feature_extractor = resnet50_baseline(pretrained=True).eval() if raw_image else None
        if dfp_dict is not None:
            dfp_init = dfp_dict['init']
            num_tokens = dfp_dict['number_prompts']
            self.prompt_disrim = dfp_dict['prompt_disrim']
            emb_length = size[0]
            self.prompt_aggregation = dfp_dict['prompt_aggregation']
            if dfp_init == "random":
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                ).to(device))
                nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
                self.prompt_norm = tv.transforms.Normalize(
                    mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
                    std=[sum([0.229, 0.224, 0.225])] * 1,
                )
            elif dfp_init == 'zeros':
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                ).to(device))
                self.prompt_norm = tv.transforms.Normalize(
                    mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
                    std=[sum([0.229, 0.224, 0.225])] * 1,
                )
            elif dfp_init == "gaussian":
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                ).to(device))
                # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                nn.init.normal_(self.prompt_embeddings.data)
                self.prompt_norm = nn.Identity()
            elif dfp_init == "he_gaussian":
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                ).to(device))
                # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                nn.init.kaiming_normal_(self.prompt_embeddings.data)
                self.prompt_norm = nn.Identity()
            else:
                raise ValueError("Other initiation scheme is not supported")
            print('Prompt initialised with shape: ', self.prompt_embeddings.size())
        initialize_weights(self)
        if self.prompt_disrim:
            self.register_buffer('confidence_bank', torch.zeros(self.n_classes, 100, 1))
            self.register_buffer('instance_bank', torch.zeros(self.n_classes, 100, size[0]))

    def relocate(self, ddp=False):
        if ddp:
            local_rank = os.environ['LOCAL_RANK']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu", int(local_rank))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        if self.feature_extractor is not None:
            self.feature_extractor = self.feature_extractor.to(device)
        if self.prompt_disrim:
            self.confidence_bank = self.confidence_bank.to(device)
            self.instance_bank = self.instance_bank.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # print('======> Select from A ', A.size(), self.k_sample, h.size(), type(h))
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward_prompt(self, h):
        B = h.size(0)
        prompt = self.prompt_norm(
            self.prompt_embeddings).expand(B, -1, -1)
        prompt = torch.permute(prompt, (1, 0, 2)).to(device)
        h = h.expand(prompt.size(0), -1, -1)
        if self.prompt_aggregation == "add":
            h = h + prompt
        elif self.prompt_aggregation == "prepend":
            h = torch.cat((prompt, h), dim=1)
        elif self.prompt_aggregation == "multiply":
            h = h * prompt
        else:
            raise NotImplementedError
        h = torch.mean(h, dim=0)
        return h

    def update_memory_bank(self, h, h_score, label):
        # add to memory bank based on the value h, if the memory bank is full, pop the smallest one
        # h: k, 1024
        # label: 1
        # memory_bank: 2, 100, 1024
        label = label.item()
        # select idx of memory bank that are smaller than h, and replace them with h
        k = h.size(0)
        idx = torch.topk(self.confidence_bank[label], k, dim=0, largest=False)[1]
        if self.confidence_bank[label][idx][0] < h_score[0]:
            self.confidence_bank[label][idx] = h_score
            self.instance_bank[label][idx] = h

    def forward(self, h,
                label=None,
                instance_eval=False,
                return_features=False,
                attention_only=False,
                use_prompt=False,
                custom_features=None,
                augment_features=None):
        if custom_features is not None:
            M = custom_features
        else:
            if self.dfp is not None:
                h = self.forward_prompt(h)
            h_prompt = h
            h_org = h

            A, h = self.attention_net(h)  # NxK
            A = torch.transpose(A, 1, 0)  # KxN
            if attention_only:
                return A
            A_raw = A
            A = F.softmax(A, dim=1)  # softmax over N
            instance_plabel = torch.argmax(A, dim=0)
            if self.prompt_disrim:
                top_ids = torch.topk(A, self.k_sample)[1][-1]
                top_score = torch.index_select(A, dim=1, index=top_ids)
                top_instances = torch.index_select(h_org, dim=0, index=top_ids)
                self.update_memory_bank(top_instances, top_score, label)
            if instance_eval:
                total_inst_loss = 0.0
                all_preds = []
                all_targets = []
                inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
                for i in range(len(self.instance_classifiers)):
                    inst_label = inst_labels[i].item()
                    classifier = self.instance_classifiers[i]
                    if inst_label == 1:  # in-the-class:
                        instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:  # out-of-the-class
                        if self.subtyping:
                            instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                            all_preds.extend(preds.cpu().numpy())
                            all_targets.extend(targets.cpu().numpy())
                        else:
                            continue
                    total_inst_loss += instance_loss

                if self.subtyping:
                    total_inst_loss /= len(self.instance_classifiers)
            M = torch.mm(A, h)
        # print('Class size: ', self.classifiers.weight.data.size(), self.classifiers.weight.data[label,:].size())
        # print('correlation: ', scipy.stats.pearsonr(h.detach().cpu().numpy(), M.detach().cpu().numpy()))
        # 1xft_Size, 1
        # print('M size: ', M.size(), label.size())
        #TODO: Add Global Features
        if augment_features is not None:
            # print('Augmenting features 1', M.size(), augment_features.size())
            # augment_features = torch.unsqueeze(augment_features, dim=0)
            # M = torch.cat((M, augment_features), dim=0)
            # print('Augmented features 2', M.size())
            augment_features = torch.unsqueeze(augment_features, dim=0)
            M = M + augment_features
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if self.prompt_disrim:
            results_dict.update({'prompted_features': h_prompt, 'top_prompted_feature': top_instances, 'all_instance_labels': instance_plabel})
        if return_features:
            results_dict.update({'features': M, 'patch_features': h, 'class_centers': self.classifiers.weight.data[label,:]})
        return logits, Y_prob, Y_hat, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dfp_dict=None):
        nn.Module.__init__(self)
        self.size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                          "small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.dfp = dfp_dict
        if dfp_dict is not None:
            dfp_init = dfp_dict['init']
            num_tokens = dfp_dict['number_prompts']
            emb_length = dfp_dict['emb_length']
            if dfp_init == "random":
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                )).to(device)
                nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
                self.prompt_norm = tv.transforms.Normalize(
                    mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
                    std=[sum([0.229, 0.224, 0.225])] * 1,
                )

            elif dfp_init == "gaussian":
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                )).to(device)
                # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                nn.init.normal_(self.prompt_embeddings.data)
                self.prompt_norm = nn.Identity()
            else:
                raise ValueError("Other initiation scheme is not supported")
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, use_prompt=False):
        device = h.device
        if self.dfp is not None:
            B = h.size(0)
            prompt = self.prompt_norm(
                self.prompt_embeddings).expand(B, -1, -1)
            prompt = torch.permute(prompt, (1, 0, 2))
            h = h.expand(prompt.size(0), -1, -1)
            if self.prompt_aggregation == "add":
                h = h + prompt
            elif self.prompt_aggregation == "prepend":
                h = torch.cat((prompt, h), dim=1)
            elif self.prompt_aggregation == "multiply":
                h = h * prompt
            else:
                raise NotImplementedError
        A, h = self.attention_net(h)  # NxK        
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict


class CLAM_SB_prompted(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, dfp_dict=None):
        super(CLAM_SB_prompted, self).__init__()
        self.size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                          "small": [1024, 512, 256], "big": [2048, 512, 384]}
        size = self.size_dict[size_arg]
        self.size = size
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        self.dfp = dfp_dict
        self.embed_length = size[0]
        self.prompt_disrim = self.dfp['prompt_disrim']
        #     self.prompt_init(dfp_dict)
        if self.dfp is not None:
            self.prompt_init()
        initialize_weights(self)
        if self.prompt_disrim:
            self.register_buffer('confidence_bank', torch.zeros(self.n_classes, 100, 1))
            self.register_buffer('instance_bank', torch.zeros(self.n_classes, 100, size[0]))

    def prompt_init(self):
        dfp_init = self.dfp['init']
        num_tokens = self.dfp['number_prompts']
        self.prompt_aggregation = self.dfp['prompt_aggregation']
        if dfp_init == "random":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.size[0]
            ).to(device))
            nn.init.uniform_(self.prompt_embeddings.data, 0.0, 1.0)
            self.prompt_norm = tv.transforms.Normalize(
                mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
                std=[sum([0.229, 0.224, 0.225])] * 1,
            )
        elif dfp_init == 'zeros':
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.size[0]
            ).to(device))
            self.prompt_norm = tv.transforms.Normalize(
                mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
                std=[sum([0.229, 0.224, 0.225])] * 1,
            )
        elif dfp_init == "gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.size[0]
            ).to(device))
            # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            nn.init.normal_(self.prompt_embeddings.data)
            self.prompt_norm = nn.Identity()
        elif dfp_init == "he_gaussian":
            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.size[0]
            ).to(device))
            # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            nn.init.kaiming_normal_(self.prompt_embeddings.data)
            self.prompt_norm = nn.Identity()
        elif dfp_init == "class_center":
            mean_instance_per_cls = torch.mean(self.instance_bank, dim=1).detach()
            mean_instance_per_data = torch.mean(mean_instance_per_cls, dim=0).unsqueeze(0)
            random_noise = nn.init.normal_(mean_instance_per_data)
            self.prompt_embeddings = nn.Parameter((mean_instance_per_cls).data.to(device))
            self.prompt_norm = nn.Identity()
        else:
            raise ValueError("Other initiation scheme is not supported")
        print('Prompt initialised with shape: ', self.prompt_embeddings.size())
    def relocate(self, ddp=False):
        if ddp:
            local_rank = os.environ['LOCAL_RANK']
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu", int(local_rank))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)
        if self.prompt_disrim:
            self.confidence_bank = self.confidence_bank.to(device)
            self.instance_bank = self.instance_bank.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        # print('======> Select from A ', A.size(), self.k_sample, h.size(), type(h))
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward_prompt(self, h):
        B = h.size(0)
        prompt = self.prompt_norm(
            self.prompt_embeddings).expand(B, -1, -1)
        prompt = torch.permute(prompt, (1, 0, 2)).to(device)
        h = h.expand(prompt.size(0), -1, -1)
        if self.prompt_aggregation == "add":
            h = h + prompt
        elif self.prompt_aggregation == "prepend":
            h = torch.cat((prompt, h), dim=1)
        elif self.prompt_aggregation == "multiply":
            h = h * prompt
        else:
            raise NotImplementedError
        h = torch.mean(h, dim=0)
        return h

    def update_memory_bank(self, h, h_score, label):
        # add to memory bank based on the value h, if the memory bank is full, pop the smallest one
        # h: k, 1024
        # label: 1
        # memory_bank: 2, 100, 1024
        label = label.item()
        # select idx of memory bank that are smaller than h, and replace them with h
        k = h.size(0)
        idx = torch.topk(self.confidence_bank[label], k, dim=0, largest=False)[1]
        if self.confidence_bank[label][idx][0] < h_score[0]:
            self.confidence_bank[label][idx] = h_score
            self.instance_bank[label][idx] = h

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False, use_prompt=False):
        device = h.device
        h_org = h
        if self.dfp is not None:
            h = self.forward_prompt(h)
        h_prompt = h

        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        instance_plabel = torch.argmax(A, dim=0)
        if self.prompt_disrim:
            top_ids = torch.topk(A, self.k_sample)[1][-1]
            top_score = torch.index_select(A, dim=1, index=top_ids)
            top_instances = torch.index_select(h_org, dim=0, index=top_ids)
            self.update_memory_bank(top_instances, top_score, label)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        # print('Class size: ', self.classifiers.weight.data.size(), self.classifiers.weight.data[label,:].size())
        # print('correlation: ', scipy.stats.pearsonr(h.detach().cpu().numpy(), M.detach().cpu().numpy()))
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if self.prompt_disrim:
            results_dict.update({'prompted_features': h_prompt, 'top_prompted_feature': top_instances, 'all_instance_labels': instance_plabel})
        if return_features:
            results_dict.update({'features': M, 'patch_features': h, 'class_centers': self.classifiers.weight.data[label,:]})
        return logits, Y_prob, Y_hat, A_raw, results_dict


if __name__ == '__main__':
    model_dict = {"dropout": 0.8, 'n_classes': 2}
    model_dict.update({"size_arg": 'small'})
    model_dict.update({'k_sample': 8})
    model_dict.update({'raw_image': False})
    from topk.svm import SmoothTop1SVM
    from model.resnet_custom import resnet50_baseline
    feature_extractor = resnet50_baseline(pretrained=True).cuda()
    from torchvision import transforms
    instance_loss_fn = SmoothTop1SVM(n_classes=2)
    instance_loss_fn = instance_loss_fn.cuda()
    model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    # get number of trainable parameter in model
    print('Number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.relocate()
    # x_feature = torch.randn((10, 1024)).cuda()
    # x_feature_before = deepcopy(x_feature)

    x_image = torch.randn((1, 10, 3, 32, 32),
                        dtype=torch.float,
                        requires_grad=True,
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))#, requires_grad=False
    optimizer_img = torch.optim.Adam([x_image], lr=1.)
    feature_extractor = resnet50_baseline(pretrained=True).cuda().eval()
    # trsforms = [transforms.Resize(256)]
    trsforms = []
    trsforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    trsforms = transforms.Compose(trsforms)
    x_image = trsforms(x_image.squeeze(0))
    x_feature_before = deepcopy(x_image.detach())
    # print(x_feature[0][0][0][:10])
    x_feature = feature_extractor(x_image)
    # print(x_feature[0][:10])
    # x_feature_before = deepcopy(x_feature)
    y = torch.tensor([0]).cuda()
    logits, Y_prob, Y_hat, results_dict = model(x_feature, y, instance_eval=True, return_features=True)
    print('feature size ', x_feature.size(), results_dict['features'].size())
    loss_instance = results_dict['instance_loss']
    loss_slide = nn.CrossEntropyLoss()(logits, y)
    total_loss = loss_instance + loss_slide
    optimizer_img.zero_grad()
    total_loss.backward()
    optimizer_img.step()
    # print(x_feature[0][0][0][:10])
    # print(x_feature[0][:10])
    print(total_loss)
    # check if x is updated
    print('Check if x is not updated: ', torch.equal(x_feature, x_feature_before))

