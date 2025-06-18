from model.Client import AgentBase
from utils.Get_model import define_model
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
from copy import deepcopy
import torch


class FedScaffoldAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)
        # server control variate
        self.scv = define_model(args)
        # client control variate
        self.ccv = define_model(args)

    def update(self, model_state_dict, scv_state):
        """
        SCAFFOLD client updates local models and server control variate
        :param model_state_dict:
        :param scv_state:
        """
        self.local_model = define_model(self.args)
        self.local_model.load_state_dict(model_state_dict)
        self.scv = define_model(self.args)
        self.scv.load_state_dict(scv_state)

    def local_train(self, agent_idx):
        self.turn_on_training()
        self.local_normalizing_vec = 0
        optimizer = get_optim(self.args, self.local_model)
        global_state_dict = deepcopy(self.local_model.state_dict())
        scv_state = self.scv.state_dict()
        ccv_state = self.ccv.state_dict()

        epoch_loss = 0.
        cnt = 0
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                loss, error, y_prob = self.mil_run(self.local_model, images, labels, self.mil_loss)
                loss.backward()
                optimizer.step()
                if batch_idx % 40 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                state_dict = self.local_model.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key] - self.args.lr * (scv_state[key] - ccv_state[key])
                self.local_model.load_state_dict(state_dict)

                cnt += 1
                # break
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss

        delta_model_state = deepcopy(self.local_model.state_dict())

        new_ccv_state = deepcopy(self.ccv.state_dict())
        delta_ccv_state = deepcopy(new_ccv_state)
        state_dict = self.local_model.state_dict()
        for key in state_dict:
            new_ccv_state[key] = ccv_state[key] - scv_state[key] + (global_state_dict[key] - state_dict[key]) / (
                        cnt * self.args.lr)
            delta_ccv_state[key] = new_ccv_state[key] - ccv_state[key]
            delta_model_state[key] = state_dict[key] - global_state_dict[key]

        self.ccv.load_state_dict(new_ccv_state)

        return state_dict, epoch_loss / self.args.local_epochs, new_ccv_state
