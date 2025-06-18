from copy import deepcopy

from tqdm import tqdm

from utils.swd_loss import ISEBSW
import torch.nn.functional as F
from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim, random_pertube
import torch

class FedAFAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)
        self.init_syn_data()

    def init_syn_data(self):
        self.feature_size = 1024 if 'ResNet50' in self.args.ft_model else 384
        self.ipc = self.args.ipc
        self.nps = self.args.nps
        self.rho = self.args.rho
        self.image_lr = self.args.image_lr
        self.dc_iterations = self.args.dc_iterations
        self.synthetic_images = torch.randn(
            size=(
                self.args.n_classes * self.ipc,
                self.nps,
                self.feature_size
            ),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )

    def update_class_logit(self):
        self.local_model.eval()
        self.local_model.to(self.device)
        class_logits = {}
        for batch_idx, (images, labels) in tqdm(enumerate(self.train_loader)):
            images, labels = images.to(self.device), labels.to(self.device)
            self.local_model.zero_grad()
            if 'CLAM' in self.args.mil_method:
                _, _, real_feature, real_logits = clam_runner(self.args,
                                          self.local_model,
                                          images,
                                          labels,
                                          self.mil_loss,
                                          return_lgt=True,
                                          return_feature=True)
            else:
                self.logger.error(f'{self.args.mil_method} not implemented')
                raise NotImplementedError
            if labels[0].cpu().detach().item() not in class_logits:
                class_logits[labels[0].cpu().detach().item()] = [torch.mean(real_feature.cpu().detach(), dim=0).unsqueeze(0)]
            else:
                class_logits[labels[0].cpu().detach().item()].append(torch.mean(real_feature.cpu().detach(), dim=0).unsqueeze(0))

        for cls in range(self.args.n_classes):
            class_logits[cls] = torch.mean(torch.cat(class_logits[cls], dim=0), dim=0)
        return class_logits

    def local_train(self, agent_idx):
        self.turn_on_training()
        # initialize S_k from real examples and initialize optimizer
        for i, c in enumerate(range(self.args.n_classes)):
            self.synthetic_images.data[i * self.ipc: (i + 1) * self.ipc] = self.train_dataset.get_image(c, self.ipc, self.nps).detach().data
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)
        optimizer_image.zero_grad()
        total_loss = 0.0
        v_real = self.update_class_logit()
        for dc_iteration in range(self.dc_iterations):
            # sample w ~ P_w(w_r)
            # sample_model = sample_random_model(self.global_model, self.rho)
            sample_model = random_pertube(self.local_model, self.rho)
            sample_model.eval()
            loss = torch.tensor(0.0).to(self.device)

            u_syn = {}
            r_soft = {}
            for i, c in enumerate(range(self.args.n_classes)):
                labels = torch.tensor([c] * self.ipc, dtype=torch.long, device=self.device)
                real_image = self.train_dataset.get_image(c, 1, 0)
                synthetic_image = self.synthetic_images[i * self.ipc: (i + 1) * self.ipc]
                real_image = real_image.to(self.device)

                for syn_i in range(self.ipc):
                    if 'CLAM' in self.args.mil_method:
                        _, _, real_feature, real_logits = clam_runner(self.args,
                                                  sample_model,
                                                  real_image,
                                                  labels[0].unsqueeze(0),
                                                  self.mil_loss,
                                                  return_lgt=True,
                                                  return_feature=True)
                        _, _, synthetic_feature, synthetic_logits = clam_runner(self.args,
                                                  sample_model,
                                                  synthetic_image[syn_i],
                                                  labels[0].unsqueeze(0),
                                                  self.mil_loss,
                                                  return_lgt=True,
                                                  return_feature=True)
                        Y_prob_real = F.softmax(real_logits, dim=1)
                    else:
                        self.logger.error(f'{self.args.mil_method} not implemented')

                    if labels[0].cpu().detach().item() not in u_syn:
                        u_syn[labels[0].cpu().detach().item()] = [torch.mean(synthetic_feature, dim=0).unsqueeze(0)]
                        r_soft[labels[0].cpu().detach().item()] = [Y_prob_real.unsqueeze(0)]
                    else:
                        u_syn[labels[0].cpu().detach().item()].append(torch.mean(synthetic_feature, dim=0).unsqueeze(0))
                        r_soft[labels[0].cpu().detach().item()].append(Y_prob_real.unsqueeze(0))

                    loss += torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
                    loss += torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0)) ** 2)
            # CDC loss
            for cls in range(self.args.n_classes):
                u_syn[cls] = torch.mean(torch.cat(u_syn[cls], dim=0), dim=0)
                r_soft[cls] = torch.mean(torch.cat(r_soft[cls], dim=0), dim=0).detach()
                loss += self.args.lambda_local * ISEBSW(v_real[cls].unsqueeze(0).to(self.device), u_syn[cls].unsqueeze(0), device=self.device)

            optimizer_image.zero_grad()
            loss.backward()
            optimizer_image.step()

            if dc_iteration % 20 == 0:
                self.logger.info(f'Agent: {agent_idx}, Iter: {dc_iteration}, DM Loss: {loss.item()}')
            total_loss += loss.item()
        # return S_k
        synthetic_labels = torch.cat([torch.ones(self.ipc) * c for c in range(self.args.n_classes)])
        return deepcopy(self.synthetic_images.detach()), synthetic_labels.long(), r_soft, total_loss / self.dc_iterations

