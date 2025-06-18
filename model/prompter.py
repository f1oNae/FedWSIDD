import time

import torch
import torch.nn as nn
import numpy as np
import torchvision as tv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prompt_init(dfp_config, instance_bank=None):
    dfp_init = dfp_config['init']
    num_tokens = dfp_config['number_prompts']
    prompt_size = dfp_config['prompt_size']
    if dfp_init == "random":
        prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_size
        ).to(device))
        nn.init.uniform_(prompt_embeddings.data, 0.0, 1.0)
        prompt_norm = tv.transforms.Normalize(
            mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
            std=[sum([0.229, 0.224, 0.225])] * 1,
        )
    elif dfp_init == 'zeros':
        prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_size
        ).to(device))
        prompt_norm = tv.transforms.Normalize(
            mean=[sum([0.485, 0.456, 0.406])] * 1,  # /3, self.num_tokens
            std=[sum([0.229, 0.224, 0.225])] * 1,
        )
    elif dfp_init == "gaussian":
        prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_size
        ).to(device))
        nn.init.normal_(prompt_embeddings.data)
        prompt_norm = nn.Identity()
    elif dfp_init == "he_gaussian":
        prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_size
        ).to(device))
        nn.init.kaiming_normal_(prompt_embeddings.data)
        prompt_norm = nn.Identity()
    elif dfp_init == "class_center":
        assert instance_bank is not None
        mean_instance_per_cls = torch.mean(instance_bank, dim=1).detach()
        mean_instance_per_data = torch.mean(mean_instance_per_cls, dim=0).unsqueeze(0)
        prompt_embeddings = nn.Parameter((mean_instance_per_cls).data.to(device))
        prompt_norm = nn.Identity()
    else:
        raise ValueError("Other initiation scheme is not supported")
    print('Prompt initialised with shape: ', prompt_size)
    return prompt_embeddings, prompt_norm

class Prompter(nn.Module):
    def __init__(self, args):
        super(Prompter, self).__init__()
        self.prompt_embeddings, self.prompt_norm = prompt_init(args)
        self.prompt_aggregation = args['prompt_aggregation']

    def forward(self, h):
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

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)