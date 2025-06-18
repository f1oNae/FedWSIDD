import torch.nn as nn
from GaussianSynthesisLabel import GaussianSynthesisLabel

class FeatureGenerator(nn.Module):
    def __init__(self, args, device='cpu'):
        super(FeatureGenerator, self).__init__()
        self.args = args
        self.device = device
        self.num_classes = self.args.num_classes

        self.max_epochs = 1000

        if self.args.dataset == 'cifar10':
            self.predefined_number_per_class = 5000
        elif self.args.dataset == 'fmnist':
            self.predefined_number_per_class = 6000
        elif self.args.dataset == 'SVHN':
            self.predefined_number_per_class = 7000
        elif self.args.dataset == 'cifar100':
            self.predefined_number_per_class = 600
        elif self.args.dataset == 'femnist':
            self.predefined_number_per_class = 10000
        else:
            raise NotImplementedError


        self.forward_count = 0

        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis = GaussianSynthesisLabel(args, device)
        else:
            raise NotImplementedError


    def update(self, progress, feat=None, labels=None, fake_feat=None, fake_labels=None):
        decode_error = 0.0
        if self.args.fed_split == "FeatureSynthesisLabel":
            decode_error = self.feature_synthesis.update(progress, feat, labels, fake_feat, fake_labels)
        else:
            raise NotImplementedError
        return decode_error


    def move_to_gpu(self, device):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.move_to_gpu(device)
        else:
            raise NotImplementedError


    def move_to_cpu(self):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.move_to_cpu()
        else:
            raise NotImplementedError


    def sample(self, x=None, labels=None):
        if self.args.fed_split == "FeatureSynthesisLabel":
            align_features, align_labels = self.feature_synthesis.sample(x, labels)
            return align_features, align_labels
        else:
            raise NotImplementedError


    def initial_model_params(self, feat, feat_length=None):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.initial_model_params(feat, feat_length)
        else:
            raise NotImplementedError


    def get_model_params(self, DP_degree=None):
        if self.args.fed_split == "FeatureSynthesisLabel":
            return self.feature_synthesis.get_model_params(DP_degree=DP_degree)
        else:
            raise NotImplementedError
        return model_params


    def set_model_params(self, model_parameters):
        if self.args.fed_split == "FeatureSynthesisLabel":
            self.feature_synthesis.set_model_params(model_parameters)
            # self.model.load_state_dict(model_parameters)
        else:
            raise NotImplementedError


    def __(self):
        if self.args.fed_split == "FeatureSynthesisLabel":
            pass
        else:
            raise NotImplementedError