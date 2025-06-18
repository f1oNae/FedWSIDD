import numpy as np
import torch
from sklearn.metrics import roc_curve
from tqdm import tqdm

from model.Client import AgentBase
from model.prompter import Prompter
from sklearn.preprocessing import label_binarize
from utils.trainer_util import RAdam
from copy import deepcopy
import torch.optim as optim


class FedPromptAgent(AgentBase):
    def __init__(self, args, global_model, logger, MIL_pool):
        super().__init__(args, global_model, logger, MIL_pool)
        self.num_coarse_stain_classes = 1
        if args.use_stain:
            stain_proto_path = f'./data/pre_extracted_color_feature/{self.args.task}'
            self.stain_prototype = torch.load('%s/Train/prototype.pt' % stain_proto_path).to(self.device)
            self.num_coarse_stain_classes = self.stain_prototype.size(0)
            print(f'========={self.num_coarse_stain_classes} stain prototype loaded=========')
            self.logger.info(f'========={self.num_coarse_stain_classes} stain prototype loaded=========')
        dfp_dict = {'init': args.prompt_initialisation,
                    'number_prompts': args.number_prompts,
                    'prompt_aggregation': args.prompt_aggregation,
                    'prompt_size': self.local_model.size[0]}
        self.prompter = Prompter(dfp_dict)

    def get_optim(self, weight, local_lr):
        self.prompter_gather, self.prompter_params_gather = [], []
        for i in range(self.num_coarse_stain_classes):
            self.prompter_gather.append(
                deepcopy(self.prompter)
            )
            # self.args.prompt_lr * weight if self.args.adaptive_prompt_lr else self.args.prompt_lr,
            local_lr = local_lr if local_lr is not None else self.args.prompt_lr
            self.prompter_params_gather.append(
                {'params': self.prompter_gather[i].parameters(),
                 'lr':local_lr,
                 'weight_decay':self.args.prompt_reg}
            )
        self.prompter_params_gather.append(
            {'params': filter(lambda p: p.requires_grad, self.local_model.parameters()),
             'lr': self.args.lr,
             'weight_decay': self.args.reg}
        )

        if self.args.opt == "adam":
            optimizer = optim.Adam(self.prompter_params_gather)
        elif self.args.opt == 'adamw':
            optimizer = optim.AdamW(self.prompter_params_gather)
        elif self.args.opt == 'sgd':
            for i in range(len(self.prompter_params_gather)):
                self.prompter_params_gather[i]['momentum'] = 0.9
            optimizer = optim.SGD(self.prompter_params_gather)
        elif self.args.opt == 'radam':
            optimizer = RAdam(self.prompter_params_gather)
        else:
            raise NotImplementedError
        return optimizer

    def get_prompted_ft_based_on_stain(self, h, h_stain, prompter_gather=None):
        prompted_image = []
        reform = False
        if len(h.size()) > 2:
            b, n, _ = h.size()
            for i in range(h.size(0)):
                prompted_image_batch = []
                h_i = h[i]
                h_stain_i = h_stain[i]
                unique_idx = torch.unique(h_stain_i).to(torch.long)
                for i in unique_idx:
                    idx_h = torch.where(h_stain_i == i)[0].to(torch.long)
                    prompted_image_batch.append(
                        prompter_gather[i](h_i[idx_h])
                    )
                prompted_image.append(torch.cat(prompted_image_batch, dim=0))
            prompted_image = torch.cat(prompted_image, dim=0)
            prompted_image = prompted_image.view(b, n, -1)
        else:
            indices = h_stain
            unique_idx = torch.unique(indices).to(torch.long)
            for i in unique_idx:
                idx_h = torch.where(indices == i)[0].to(torch.long)
                prompted_image.append(
                    prompter_gather[0](h[idx_h])
                )
            prompted_image = torch.cat(prompted_image, dim=0)
        return prompted_image

    def get_prompted_ft(self, h, prompter_gather=None):
        reform = False
        if len(h.size()) > 2:
            reform = True
            b, n, _ = h.size()
            h = h.view(-1, h.size(-1))
        h = prompter_gather[0](h)
        if reform:
            h = h.view(b, n, -1)
        return h

    def update_prompt(self, prompt):
        self.prompter_gather[0] = deepcopy(prompt)

    def local_train(self, agent_idx, agent_weight, epoch=None, local_model_train=True, local_lr=None):
        if local_model_train:
            self.turn_on_training()
        else:
            self.turn_off_training()
        optimizer = self.get_optim(agent_weight, local_lr)
        epoch_loss = 0.
        best_local_acc = 0.
        best_fpr = []
        best_tpr = []
        best_prompter = deepcopy(self.prompter_gather)
        epoch = epoch if epoch is not None else self.args.local_epochs
        for iter in tqdm(range(epoch)):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                prompted_data = self.get_prompted_ft(images, self.prompter_gather)
                loss, error, y_prob = self.mil_run(self.local_model, prompted_data, labels, self.mil_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()

            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
            if iter % 1 == 0:
                test_loss, test_error, fpr, tpr = self.local_test()
                # logging_info = f'Agent: {agent_idx}, Iter: {iter}, Train Loss: {batch_loss}, Test Acc: {1 - test_error}'
                # print(logging_info)
                # self.logger.info(logging_info)
                if 1 - test_error > best_local_acc:
                    best_local_acc = 1 - test_error
                    best_fpr = fpr
                    best_tpr = tpr
                    best_prompter = deepcopy(self.prompter_gather)

        return self.local_model.state_dict(), best_prompter, best_local_acc, best_fpr, best_tpr

    def local_test(self, **kwargs):
        self.local_model.eval()
        total_error = 0.
        total_loss = 0.
        all_probs = np.zeros((len(self.test_loader), self.args.n_classes))
        all_labels = np.zeros(len(self.test_loader))
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data_ft, label = data
                data_ft, label = data_ft.to(self.device), label.to(self.device)
                prompted_data = self.get_prompted_ft(data_ft, self.prompter_gather)
                loss, error, y_prob = self.mil_run(self.local_model, prompted_data, label, self.mil_loss)
                total_loss += loss.item()
                total_error += error
                probs = y_prob.detach().cpu().numpy()

                all_probs[batch_idx] = probs
                all_labels[batch_idx] = label.item()
            total_loss /= len(self.test_loader)
            total_error /= len(self.test_loader)
            if self.args.n_classes == 2:
                fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
            else:
                fpr = dict()
                tpr = dict()
                y_true_bin = label_binarize(all_labels, classes=list(range(self.args.n_classes)))
                for i in range(y_true_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        return total_loss, total_error, fpr, tpr