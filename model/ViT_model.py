from copy import deepcopy
from collections import OrderedDict
from torchvision import models
import torch
# from timm.models.helpers import update_pretrained_cfg_and_kwargs, load_pretrained, load_custom_pretrained
# from timm.models.vision_transformer import default_cfgs, checkpoint_filter_fn
import timm.models.vision_transformer as vit
from torch import nn
import inspect

def load_pretrained_vit(model, model_name, variant, vit_kwargs, custom_pretrained, pretrained=True):
    if pretrained:
        print('==========> Load pretrained %s'%(model.__class__.__name__))
    pretrained_cfg = deepcopy(default_cfgs[variant])
    update_pretrained_cfg_and_kwargs(pretrained_cfg, vit_kwargs, None)
    pretrained_cfg.setdefault('architecture', variant)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    num_classes_pretrained = getattr(model, 'num_classes', vit_kwargs.get('num_classes', 1000))
    pretrained_custom_load = 'npz' in pretrained_cfg['url']
    if custom_pretrained is not None:
        model = custom_load_pretrained(model, model_name, custom_pretrained)
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=vit_kwargs.get('in_chans', 3),
                filter_fn=checkpoint_filter_fn,
                strict=False)
    return model

def custom_load_pretrained(model, model_name, custom_pretrained):
    print('Custom load pretrained model from ', custom_pretrained)
    ckp = torch.load(custom_pretrained)
    from collections import OrderedDict
    pretrained_state_dict = OrderedDict()
    if 'dino' in model_name:
        print('Load DINO model from ', custom_pretrained)
        new_state_dict = model.state_dict()
        for k, v in ckp['teacher'].items():
            name_list = k.split('.')
            if name_list[0] == 'backbone':
                name = '.'.join(name_list[1:])
            else:
                name = k
            pretrained_state_dict[name] = v
        for k, v in model.state_dict().items():
            if k not in pretrained_state_dict:
                print(k)
                new_state_dict[k] = v
            else:
                new_state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(new_state_dict, strict=True)
    elif 'simclr' in model_name:
        print('Load SimCLR model from ', custom_pretrained)
        new_state_dict = model.state_dict()
        for k, v in ckp.items():
            name_list = k.split('.')
            if 'module' in k and 'features' in k:
                name = '.'.join(name_list[2:])
            elif 'module' in k:
                name = '.'.join(name_list[1:])
            else:
                name = k
            pretrained_state_dict[name] = v
        for k, v in model.state_dict().items():
            if k not in pretrained_state_dict:
                print('Not found in pretrained: ', k)
                new_state_dict[k] = v
            else:
                new_state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(new_state_dict, strict=True)
    return model

def get_init_params(cls):
    signature = inspect.signature(cls.__init__)
    params = {}
    for name, param in signature.parameters.items():
        if param.default is param.empty:
            params[name] = "No default value"
        else:
            params[name] = param.default
    return params
class BaseTransformer(vit.VisionTransformer):
    """ Vision Transformer

        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
            - https://arxiv.org/abs/2010.11929
        """

    def __init__(
            self,
            vit_config,
    ):
        super().__init__(**vit_config)
        self.vit_config = vit_config
        self.init_params = get_init_params(super())
        self.norm = nn.LayerNorm(vit_config["embed_dim"], eps=1e-6)
        self.head = nn.Identity()
    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            super().train(mode)
        else:
            # eval:
            for module in self.children():
                module.train(mode)
    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, return_feature=False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) \
                if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if return_feature else self.head(x)

    def forward(self, x, return_feature=False):
        x = self.forward_features(x)
        x = self.forward_head(x, return_feature)
        return x

class ViT(nn.Module):
    def __init__(self, cfg, vit_kwargs, variant, pretrained, custom_pretrained=None):
        super(ViT, self).__init__()
        model = BaseTransformer(vit_kwargs)
        self.enc = load_pretrained_vit(model, 'ViT_T_16', variant, vit_kwargs, custom_pretrained,
                                    pretrained=pretrained)

        self.feat_dim = 384
        self.transfer_type = 'eval'
        self.froze_enc = True
        self.build_backbone()
        self.cfg = cfg
        self.setup_head(cfg)


    def build_backbone(self):
        # linear, prompt, cls, cls+prompt, partial_1
        if self.transfer_type == "partial-1":
            total_layer = len(self.enc.transformer.encoder.layer)
            # tuned_params = [
            #     "transformer.encoder.layer.{}".format(i-1) for i in range(total_layer)]
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False
        elif self.transfer_type == "partial-2":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 2) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False

        elif self.transfer_type == "partial-4":
            total_layer = len(self.enc.transformer.encoder.layer)
            for k, p in self.enc.named_parameters():
                if "transformer.encoder.layer.{}".format(
                        total_layer - 1) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 2) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 3) not in k and "transformer.encoder.layer.{}".format(
                        total_layer - 4) not in k and "transformer.encoder.encoder_norm" not in k:  # noqa
                    p.requires_grad = False

        elif self.transfer_type == "linear" or self.transfer_type == "side":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif self.transfer_type == "tinytl-bias":
            for k, p in self.enc.named_parameters():
                if 'bias' not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt" and self.prompt_location == "below":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "embeddings.patch_embeddings.weight" not in k and "embeddings.patch_embeddings.bias" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt+bias":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and 'bias' not in k:
                    p.requires_grad = False

        elif self.transfer_type == "prompt-noupdate":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False

        elif self.transfer_type == "cls":
            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "cls-reinit":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )

            for k, p in self.enc.named_parameters():
                if "cls_token" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "cls+prompt":
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "cls-reinit+prompt":
            nn.init.normal_(
                self.enc.transformer.embeddings.cls_token,
                std=1e-6
            )
            for k, p in self.enc.named_parameters():
                if "prompt" not in k and "cls_token" not in k:
                    p.requires_grad = False

        # adapter
        elif self.transfer_type == "adapter":
            for k, p in self.enc.named_parameters():
                if "adapter" not in k:
                    p.requires_grad = False

        elif self.transfer_type == "end2end":
            print("Enable all parameters update during training")
        elif self.transfer_type == "eval":
            for k, p in self.enc.named_parameters():
                p.requires_grad = False
        else:
            raise ValueError("transfer type {} is not supported".format(
                self.transfer_type))
    def setup_head(self, cfg):
        # self.head = MLP(
        #     input_dim=self.feat_dim,
        #     mlp_dims=[self.feat_dim] * self.cfg.MODEL.MLP_NUM + \
        #         [cfg.DATA.NUMBER_CLASSES], # noqa
        #     special_bias=True
        # )
        self.head = nn.Identity()

    def forward(self, x):
        if self.froze_enc and self.enc.training:
            self.enc.eval()
        x = self.enc(x)  # batch_size x self.feat_dim
        return x





if __name__ == '__main__':
    # model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, global_pool='avg',num_classes=2)

    pretrained = True

    variant = "vit_tiny_patch16_384"
    #
    model = PromptedTransformer(model_kwargs, 1, 0., deep_prompt=False, project_prompt_dim=-1)

    # def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    #     if not os.path.isfile(pretrained_weights):
    #         print("wrong weight path")
    #     else:
    #         state_dict = torch.load(pretrained_weights, map_location="cpu")
    #         if checkpoint_key is not None and checkpoint_key in state_dict:
    #             print(f"Take key {checkpoint_key} in provided checkpoint dict")
    #             state_dict = state_dict[checkpoint_key]
    #         # remove `module.` prefix
    #         state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #         # remove `backbone.` prefix induced by multicrop wrapper
    #         state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    #         msg = model.load_state_dict(state_dict, strict=False)
    #         print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    #
    #
    # lung_dino_path = "/data04/shared/skapse/Cell_guided/Experiments/Lung_cancer/DINO_5X/100_percent_data_ep100/vit_tiny_baseline_avgpool_fp16true_momentum996_outdim65536/checkpoint.pth"
    #
    # load_pretrained_weights(model, lung_dino_path, "teacher")

    pretrained_cfg = deepcopy(default_cfgs[variant])

    update_pretrained_cfg_and_kwargs(pretrained_cfg, model_kwargs, None)
    pretrained_cfg.setdefault('architecture', variant)

    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    num_classes_pretrained = getattr(model, 'num_classes', model_kwargs.get('num_classes', 1000))

    pretrained_custom_load = 'npz' in pretrained_cfg['url']
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=model_kwargs.get('in_chans', 3),
                filter_fn=checkpoint_filter_fn,
                strict=False)

    transfer_type = "prompt"
    if transfer_type == "prompt":
        for k, p in model.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False
    elif transfer_type == "cls":
        for k, p in model.named_parameters():
            if "cls_token" not in k:
                p.requires_grad = False
    elif transfer_type == "cls+prompt":
        for k, p in model.named_parameters():
            if "prompt" not in k and "cls_token" not in k:
                p.requires_grad = False
    elif transfer_type == "end2end":
        print("Enable all parameters update during training")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name) #, p.data)

    x = torch.randn(1, 3, 224, 224)
    print(model(x, return_feature=True).shape)
