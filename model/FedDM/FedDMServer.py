from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model.Server import ServerBase
from model.FedDM.FedDMClient import FedDMAgent
from utils.trainer_util import get_optim, get_loss
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
import os


class FedDM(ServerBase):
    def __init__(self, args, logger):
        super(FedDM, self).__init__(args, logger)
        self.setup_clients()

    def setup_clients(self):
        for idx in self.n_clients:
            curr_client = FedDMAgent(self.args, self.global_model, self.logger)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)

    def global_model_train(self, local_syn_data, local_syn_label):
        global_syn_data = torch.cat(local_syn_data, dim=0).cpu()
        global_syn_label = torch.cat(local_syn_label, dim=0).cpu()
        synthetic_dataset = TensorDataset(global_syn_data, global_syn_label)
        synthetic_dataloader = DataLoader(synthetic_dataset, 1, shuffle=True, num_workers=4)
        self.global_model.train()
        optimizer = get_optim(self.args, self.global_model)
        optimizer.zero_grad()
        loss_fn = get_loss(self.args)
        for batch_idx, (images, labels) in enumerate(synthetic_dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.global_model.zero_grad()
            if 'CLAM' in self.args.mil_method:
                loss, error = clam_runner(self.args,
                                          self.global_model,
                                          images.squeeze(0),
                                          labels,
                                          loss_fn,
                                          raw_image=True)
            else:
                self.logger.error(f'{self.args.mil_method} not implemented')
                raise NotImplementedError
            loss.backward()
            optimizer.step()
        return loss.item()


    def run(self, iter):
        # include training and testing
        self.global_model.to(self.device)
        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        best_data_save_pth = os.path.join(self.args.results_dir, "best_data_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_syn_data, local_syn_label, local_losses = [], [], []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()
            # dispatch global model to all clients
            for idx in self.n_clients:
                self.clients[idx].update_local_model(self.global_model)
            self.logger.info(f'|Performing Local Training for {len(self.clients)} clients|')
            for idx in self.n_clients:
                syn_img, syn_label, agent_loss = self.clients[idx].local_train_new(idx)
                local_syn_data.append(syn_img.cpu())
                local_syn_label.append(syn_label.cpu())
                local_losses.append(deepcopy(agent_loss))

            # update global weights
            self.logger.info(f'|Performing Global Training Using Distilled Data|')
            for epoch_dm in tqdm(range(self.args.global_epochs_dm)):
                epoch_loss = self.global_model_train(local_syn_data, local_syn_label)
                self.logger.info(f'Global Model Epoch: {epoch_dm}, Loss: {epoch_loss}')

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            self.logger.info(f'|Performing Local Testing|')
            for idx in self.n_clients:
                agent_loss, agent_error = self.clients[idx].local_test()
                list_acc.append(1-agent_error)
                list_loss.append(agent_loss)
            test_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                self.logger.info(f'Training DM Loss : {np.mean(np.array(train_loss))}')
                self.logger.info('Test Accuracy: {:.2f}% \n'.format(100 * test_acc))
                if test_acc > best_accuracy:
                    best_accuracy = test_acc
                    best_accuracy_per_agent = list_acc
                    best_model = deepcopy(self.global_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)
                    data_to_save = torch.cat(local_syn_data, dim=0).cpu()
                    torch.save(data_to_save, best_data_save_pth)


        return best_accuracy, best_accuracy_per_agent