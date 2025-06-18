import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import math, numpy as np


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, inst_mode=False):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if inst_mode:  # [N,K,D] --> [N,K,D]
            return O
        else:  # bag mode [N,1,D] --> [N,D]
            return O.squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FRMIL(nn.Module):
    def __init__(self, args, dims, dfp_dict=None):
        super().__init__()

        self.shift_feature = args.shift_feature  # cm16/msi
        self.num_outputs = args.n_classes  # 2,
        self.size = dims
        dim_hidden = dims[0]  # 512,
        num_heads = args.n_heads  # 1,
        self.k = 1

        self.enc = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

        self.cls_token = nn.Parameter(torch.Tensor(1, 1, dim_hidden))
        nn.init.xavier_uniform_(self.cls_token)

        self.conv_head = torch.nn.Conv2d(dim_hidden, dim_hidden, 3, 1, 3 // 2, groups=dim_hidden)
        torch.nn.init.xavier_uniform_(self.conv_head.weight)

        self.selt_att = MAB(dim_hidden, dim_hidden, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(dim_hidden, self.num_outputs),
        )

        self.mode = 0
        self.dfp = dfp_dict
        if dfp_dict is not None:
            dfp_init = dfp_dict['init']
            num_tokens = dfp_dict['number_prompts']
            emb_length = dim_hidden
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

            elif dfp_init == "gaussian":
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, emb_length
                ).to(device))
                # nn.init.uniform_(self.prompt_embeddings.data, -val, val)
                nn.init.normal_(self.prompt_embeddings.data)
                self.prompt_norm = nn.Identity()
            else:
                raise ValueError("Other initiation scheme is not supported")

    def recalib(self, inputs, option='max'):
        A1 = []
        Q = []
        bs = inputs.shape[0]

        if option == 'mean':
            Q = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1 = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                # print(a1.shape)
                _, m_indices = torch.sort(a1, 0, descending=True)
                # print(m_indices.shape)
                # print(inputs.size(), a1.shape, m_indices.shape)
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(self.k):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)

                feats = torch.stack(feat_q)

                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))

            A1 = torch.stack(A1)
            Q = torch.stack(Q)
            return A1, Q

    def forward_prompt(self, h):
        # h: 2, 50, 512
        B = h.size(1)
        n = h.size(0)
        prompt = self.prompt_norm(
            self.prompt_embeddings).expand(B, -1, -1) # 50, 1, 512
        prompt = torch.permute(prompt, (1, 0, 2)).to(device) # 1, 50, 512
        prompt = prompt.unsqueeze(1).repeat(1, n, 1, 1) # 1, 2, 50, 512
        h = h.expand(prompt.size(0), -1, -1, -1)
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

    def forward(self, inputs):
        # if self.dfp is not None:
        #     inputs = self.forward_prompt(inputs)
        if self.mode == 1:
            # used in feature magnitude analysis
            return self.selt_att(inputs, inputs, True)

        A1, Q = self.recalib(inputs, 'max')

        ##################################################################
        # shift features
        if not self.shift_feature:
            i_shift = inputs
        else:
            inputs = F.relu(inputs - Q)
            i_shift = inputs
        ##################################################################

        ##################################################################
        # ---->pad inputs
        H = inputs.shape[1]  # Number of Instances in Bag
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        inputs = torch.cat([inputs, inputs[:, :add_length, :]], dim=1)  # [B, N+29, D//2 ]

        # ---->cls_token
        B = inputs.shape[0]  # Batch Size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        inputs = torch.cat((cls_tokens, inputs), dim=1)

        # CNN Position Learning
        B, _, C = inputs.shape
        cls_token, feat_token = inputs[:, 0], inputs[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x = cnn_feat.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        ##################################################################

        ##################################################################
        # Bag pooling with critical feature
        bag = self.selt_att(Q, x)
        out = self.fc(bag)
        ##################################################################

        if self.training:
            return out, i_shift, A1
        else:
            return out
