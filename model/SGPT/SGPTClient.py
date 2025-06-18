from copy import deepcopy

from tqdm import tqdm

from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch


def collate_fn(batch):
    # if len(batch[0]) == 2: # load feature, label
    for item in batch:
        print(item[0].size(), item[1].size())
    img = torch.cat([item[0] for item in batch], dim=0)
    idx = torch.LongTensor([item[1] for item in batch])
    return [img, idx]

class SGPTAgent(AgentBase):
    def __init__(self, args, global_model, logger, MIL_pool):
        super().__init__(args, global_model, logger, MIL_pool)

    def local_train(self, agent_idx):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                loss, error, y_prob = self.mil_run(self.local_model, images, labels, self.mil_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
            print(f'Agent: {agent_idx}, Iter: {iter},Loss: {batch_loss}')
            self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Loss: {batch_loss}')
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs
    # def __init__(self, args, global_model, logger):
    #     super().__init__(args, global_model, logger)
    #
    # def network_training_base(self, net, optimizer):
    #     for iter in tqdm(range(self.args.local_epochs)):
    #         # for slide_idx, (x, labels) in tqdm(enumerate(self.train_loader)): # load wsi and slide label
    #         for slide_idx in range(len(self.train_dataset)):
    #             x, labels = self.train_dataset[slide_idx]
    #             labels = labels.to(self.device)
    #             labels = labels.unsqueeze(0)
    #             X_image_loader = DataLoader(x,
    #                                         batch_size=self.args.image_batch_size,
    #                                         shuffle=True,
    #                                         num_workers=0)
    #                                         # collate_fn=collate_fn)
    #             image_ft = []
    #             for batch_idx, (x_image, x_idx) in enumerate(X_image_loader):
    #                 x_image = x_image.to(self.device)
    #                 output, x_ft = net(x_image)
    #                 image_ft.append(x_ft)
    #
    #                 if batch_idx > 0 and batch_idx % 3 == 0:
    #                     slide_ft = torch.cat(image_ft, dim=0)
    #                     loss, error = clam_runner(self.args,
    #                                               self.local_model,
    #                                               slide_ft,
    #                                               labels,
    #                                               self.mil_loss)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #                     break
    #             #     break
    #             # break
    #     return net
    #
    # def local_train(self, net, param_dict):
    #     dict_loss = param_dict['dict_loss']
    #     embedding_dict = param_dict['embedding_dict']
    #     round = param_dict['round']
    #     lr = self.args.lr
    #     net.train()
    #     net.selection = False
    #     params = []
    #     params.append({'params': [p for k, p in net.named_parameters() if p.requires_grad and ('head' in k or 'common' in k)],
    #              'lr': self.args.prompt_lr,
    #              'momentum': 0.9,
    #              'weight_decay': self.args.prompt_reg}
    #         )
    #     params.append(
    #         {'params': filter(lambda p: p.requires_grad, self.local_model.parameters()),
    #          'lr': self.args.lr,
    #          'momentum': 0.9,
    #          'weight_decay': self.args.reg}
    #     )
    #     optimizer = optim.SGD(params)
    #     print('Performing local training Stage 1')
    #     self.logger.info('Performing local training Stage 1')
    #     net = self.network_training_base(net, optimizer)
    #     net.selection = True
    #     net.train()
    #     params = []
    #     params.append({'params': [p for k,p in net.named_parameters() if p.requires_grad  and 'prompt_keys_pr' not in k and 'prompt_common_g' not in k  and 'common' not in k ],
    #              'lr': self.args.prompt_lr,
    #              'momentum': 0.9,
    #              'weight_decay': self.args.prompt_reg}
    #         )
    #     params.append(
    #         {'params': filter(lambda p: p.requires_grad, self.local_model.parameters()),
    #          'lr': self.args.lr,
    #          'momentum': 0.9,
    #          'weight_decay': self.args.reg}
    #     )
    #     optimizer = optim.SGD(params)
    #     print('Performing local training Stage 2')
    #     self.logger.info('Performing local training Stage 2')
    #     for iter in tqdm(range(self.args.local_epochs)):
    #         for slide_idx in range(len(self.train_dataset)):
    #             x, labels = self.train_dataset[slide_idx]
    #             labels = labels.to(self.device)
    #             labels = labels.unsqueeze(0)
    #             X_image_loader = DataLoader(x,
    #                                         batch_size=self.args.image_batch_size,
    #                                         shuffle=True,
    #                                         num_workers=0)
    #             image_ft = []
    #             reduced_sim = 0
    #             for batch_idx, (x_image, x_idx) in enumerate(X_image_loader):
    #                 x_image = x_image.to(self.device)
    #                 output, x_ft = net(x_image, x_idx, embedding_dict)
    #                 image_ft.append(x_ft)
    #                 reduced_sim += output['reduced_sim']
    #
    #                 if batch_idx > 0 and batch_idx % 3 == 0:
    #                     slide_ft = torch.cat(image_ft, dim=0)
    #                     loss, error = clam_runner(self.args,
    #                                               self.local_model,
    #                                               slide_ft,
    #                                               labels,
    #                                               self.mil_loss)
    #                     if iter < 5:
    #                         loss += reduced_sim / len(X_image_loader)
    #
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #                     break
    #             #     break
    #             # break
    #     return output['embedding_dict']
    #
    # def local_test(self, model=None):
    #     total_loss = 0.
    #     total_error = 0.
    #     with torch.no_grad():
    #         for slide_idx in tqdm(range(len(self.test_dataset))):
    #             x, labels = self.test_dataset[slide_idx]
    #             labels = labels.to(self.device)
    #             labels = labels.to(self.device)
    #             labels = labels.unsqueeze(0)
    #             X_image_loader = DataLoader(x,
    #                                         batch_size=self.args.image_batch_size,
    #                                         shuffle=True,
    #                                         num_workers=0)
    #             image_ft = []
    #             for batch_idx, (x_image, x_idx) in enumerate(X_image_loader):
    #                 x_image = x_image.to(self.device)
    #                 output, x_ft = model(x_image)
    #                 image_ft.append(x_ft)
    #             image_ft = torch.cat(image_ft, dim=0)
    #             loss, error = clam_runner(self.args,
    #                                       self.local_model,
    #                                       image_ft,
    #                                       labels,
    #                                       self.mil_loss)
    #             total_loss += loss.item()
    #             total_error += error
    #         total_loss /= len(self.test_loader)
    #         total_error /= len(self.test_loader)
    #         return total_loss, total_error
