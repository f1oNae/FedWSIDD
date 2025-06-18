import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            out, A = self.attn(self.norm(x), return_attn=True)
        else:
            out = self.attn(self.norm(x))
        x = x + out
        if return_attn:
            return x, A
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, model_size, n_classes, dfp_dict=None):
        super(TransMIL, self).__init__()
        self.size_dict = {"tiny": [192, 128], "ultra_small": [384, 192],
                          "small": [1024, 512], "big": [2048, 512]}
        self.size = self.size_dict[model_size]
        emb_length_1 = self.size[0]
        emb_length_2 = self.size[1]
        self.pos_layer = PPEG(dim=emb_length_2)
        self._fc1 = nn.Sequential(nn.Linear(emb_length_1,
                                            emb_length_2), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_length_2))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=emb_length_2)
        self.layer2 = TransLayer(dim=emb_length_2)
        self.norm = nn.LayerNorm(emb_length_2)
        self._fc2 = nn.Linear(emb_length_2, self.n_classes)

        self.dfp = dfp_dict
        self.prompt_disrim = False
        if dfp_dict is not None:
            dfp_init = dfp_dict['init']
            num_tokens = dfp_dict['number_prompts']
            self.prompt_disrim = dfp_dict['prompt_disrim']
            emb_length = emb_length_1
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
        if self.prompt_disrim:
            self.register_buffer('confidence_bank', torch.zeros(self.n_classes, 1000, 1))
            self.register_buffer('instance_bank', torch.zeros(self.n_classes, 1000, emb_length))


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

    def forward(self, data):
        h = data.float()  # [B, n, 1024]
        if self.dfp is not None:
            h = self.forward_prompt(h).unsqueeze(0)
        h = self._fc1(h)  # [B, n, 512]
        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        # h, A = self.layer2(h, return_attn=True)  # [B, N, 512]
        h = self.layer2(h)
        # ---->cls_token
        h = self.norm(h)[:, 0]
        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'features': h}
        return results_dict


if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024)).cuda()
    model = TransMIL('small', n_classes=2).cuda()
    results_dict = model(data=data)
    print(model.layer2.attn.to_qkv.weight.size())
    print(model.layer2.attn.to_out[0].weight.size())
