#!/usr/bin/env python3
"""
vit with adapter
"""
import torch
from functools import partial
import torch.nn as nn
from timm.models.vision_transformer import _cfg, default_cfgs, checkpoint_filter_fn, build_model_with_cfg, Attention, LayerScale
from timm.models.layers import Mlp, DropPath
from model.FExtractor.vit_base import BaseTransformer
from utils.core_util import load_pretrained_vit
from copy import deepcopy


def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class ADPT_Block(nn.Module):
    def __init__(
            self,
            adapter_config,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adapter_config = adapter_config
        if adapter_config['STYLE'] == "Pfeiffer":
            self.adapter_downsample = nn.Linear(
                dim,
                dim // adapter_config['REDUCATION_FACTOR']
            )
            self.adapter_upsample = nn.Linear(
                dim // adapter_config['REDUCATION_FACTOR'],
                dim
            )
            self.adapter_act_fn = ACT2FN["gelu"]

            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)

            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        else:
            raise ValueError("Other adapter styles are not supported.")
    def forward(self, x):
        if self.adapter_config['STYLE'] == "Pfeiffer":
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            h = x
            x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x
            x = x + h
        return x

class ADPTTransformer(nn.Module):
    def __init__(
            self, vit_kwargs=None, model_name=None, variant=None, adpt_kwargs=None,
    ):
        super(ADPTTransformer, self).__init__()
        base_model = BaseTransformer(vit_kwargs)
        variant = variant
        init_dict = base_model.init_params
        for key in vit_kwargs:
            init_dict[key] = vit_kwargs[key]
        self.base_model = load_pretrained_vit(base_model, model_name, variant, vit_kwargs, False, pretrained=True)
        norm_layer = init_dict['norm_layer'] or partial(nn.LayerNorm, eps=1e-6)
        act_layer = init_dict['act_layer'] or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, init_dict['drop_path_rate'], init_dict['depth'])]
        self.base_model.blocks = nn.Sequential(*[
            ADPT_Block(
                adpt_kwargs,
                dim=init_dict['embed_dim'],
                num_heads=init_dict['num_heads'],
                mlp_ratio=init_dict['mlp_ratio'],
                qkv_bias=init_dict['qkv_bias'],
                init_values=init_dict['init_values'],
                drop=init_dict['drop_rate'],
                attn_drop=init_dict['attn_drop_rate'],
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(init_dict['depth'])])
    def forward_features(self, x):
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)
        x = self.base_model.norm_pre(x)
        x = self.base_model.blocks(x)
        x = self.base_model.norm(x)
        return x

    def forward_head(self, x, return_feature=False):
        if self.base_model.global_pool:
            x = x[:, self.base_model.num_prefix_tokens:].mean(dim=1) \
                if self.base_model.global_pool == 'avg' else x[:, 0]
        x = self.base_model.fc_norm(x)
        return x if return_feature else self.base_model.head(x)

    def forward(self, x, return_feature=False):
        x = self.forward_features(x)
        x = self.forward_head(x, return_feature)
        return x

if __name__ == '__main__':
    import torch
    vit_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=3, global_pool='avg',
                      num_classes=2)
    variant = "vit_small_patch16_224"
    adpt_kwargs = dict(STYLE="Pfeiffer", REDUCATION_FACTOR=10)
    model = ADPTTransformer(vit_kwargs, adpt_kwargs)
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.size())
    print(model.base_model.blocks)
