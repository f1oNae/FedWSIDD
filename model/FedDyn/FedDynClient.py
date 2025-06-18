from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
import torch

class FedDynAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)

    def local_train(self, agent_idx,
                    alpha_coef_adpt,
                    cld_mdl_param_tensor,
                    local_param_list_curr):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                loss, error, y_prob = self.mil_run(self.local_model, images, labels, self.mil_loss)
                local_par_list = None
                for param in self.local_model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                loss_algo = alpha_coef_adpt * torch.sum(
                    local_par_list * (-cld_mdl_param_tensor + local_param_list_curr))
                # current_local_parameter * (last_step_local_parameter - global_parameter)
                loss = loss + loss_algo
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
            print(f'Agent: {agent_idx}, Iter: {iter},Loss: {batch_loss}')
            self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Loss: {batch_loss}')
        # Freeze model
        for params in self.local_model.parameters():
            params.requires_grad = False
        self.local_model.eval()
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs