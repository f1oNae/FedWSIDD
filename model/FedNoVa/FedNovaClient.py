from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
from copy import deepcopy
import torch


class FedNovaAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)
        self.rho = 0.9
        self._momentum = self.rho

    def local_train(self, agent_idx):
        self.turn_on_training()
        self.local_normalizing_vec = 0
        global_weights = deepcopy(self.local_model.state_dict())
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        tau = 0
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = clam_runner(self.args,
                                              self.local_model,
                                              images,
                                              labels,
                                              self.mil_loss)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError
                loss.backward()
                optimizer.step()
                self.local_normalizing_vec += 1
                if batch_idx % 40 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                tau += 1
                # break
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        coeff = (tau - self.rho * (1 - pow(self.rho, tau)) / (1 - self.rho)) / (1 - self.rho)
        state_dict = self.local_model.state_dict()
        norm_grad = deepcopy(global_weights)
        for key in norm_grad:
            norm_grad[key] = torch.div(global_weights[key] - state_dict[key], coeff)
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs, coeff, norm_grad
