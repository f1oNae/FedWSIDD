import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, size, n_classes, dfp_dict=None):
        super(Attention, self).__init__()
        self.size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                          "small": [1024, 512, 256], "big": [2048, 512, 384]}
        self.size = self.size_dict[size]
        self.M = self.size[0]
        self.L = self.size[1]
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.M),
        #     nn.ReLU(),
        # )
        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(140450, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_classes)
        )

        self.dfp = dfp_dict
        self.prompt_disrim = False
        if dfp_dict is not None:
            dfp_init = dfp_dict['init']
            num_tokens = dfp_dict['number_prompts']
            self.prompt_disrim = dfp_dict['prompt_disrim']
            emb_length = self.M
            self.prompt_aggregation = dfp_dict['prompt_aggregation']
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
        if self.prompt_disrim:
            self.register_buffer('confidence_bank', torch.zeros(self.n_classes, 1000, 1))
            self.register_buffer('instance_bank', torch.zeros(self.n_classes, 1000, self.M))

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

    def forward(self, x):
        # x = x.squeeze(0)
        # H = self.feature_extractor_part1(x)
        # n_b = H.size(0)
        # # H = H.view(-1, 50 * 4 * 4)
        # H = H.view(n_b, -1)
        # x = self.feature_extractor_part2(H)  # KxM
        if self.dfp is not None:
            x = self.forward_prompt(x)
        # print(x.size())
        A = self.attention(x)  # KxATTENTION_BRANCHES
        # print(A.size())
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K
        # print(A.size())
        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM
        # print(Z.size())
        logits = self.classifier(Z)
        # print(logits.size())
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, A, Z

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        logits, Y_prob, Y_hat, A = self.forward(X)
        # Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        neg_log_likelihood = nn.CrossEntropyLoss()(Y_prob, Y.long())

        return neg_log_likelihood, A, Y_hat

class GatedAttention(nn.Module):
    def __init__(self, size, n_classes, dfp_dict=None):
        super(GatedAttention, self).__init__()
        self.size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                          "small": [1024, 512, 256], "big": [2048, 512, 384]}
        self.size = self.size_dict[size]
        self.M = self.size[0]
        self.L = self.size[1]
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, n_classes),
        )
        self.dfp = dfp_dict
        self.prompt_disrim = False
        if dfp_dict is not None:
            dfp_init = dfp_dict['init']
            num_tokens = dfp_dict['number_prompts']
            self.prompt_disrim = dfp_dict['prompt_disrim']
            emb_length = self.M
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
            self.register_buffer('instance_bank', torch.zeros(self.n_classes, 1000, self.M))

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

    def forward(self, x):
        # x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM
        if self.dfp is not None:
            x = self.forward_prompt(x)

        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        logits = self.classifier(Z)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        return logits, Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        logits, Y_prob, Y_hat, A= self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        neg_log_likelihood = nn.CrossEntropyLoss()(Y_prob, Y.long())

        return neg_log_likelihood, A, Y_hat