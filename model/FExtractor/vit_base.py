import timm.models.vision_transformer as vit
from torch import nn
import inspect

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
