from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
import torch

class FedMOONAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)

    def local_train(self, agent_idx, global_model=None, prev_model_pool=None):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][agent_idx])
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                loss, error, y_prob, bag_ft_1 = self.mil_run(self.local_model,
                                                                images,
                                                                labels,
                                                                self.mil_loss,
                                                                return_feature=True)
                _, _, _, bag_ft_2 = self.mil_run(global_model,
                                                             images,
                                                             labels,
                                                             self.mil_loss,
                                                             return_feature=True)

                # contrastive loss = max_dis(local, prev_local)+min_dis(local, global)
                posi = self.cos_loss(bag_ft_1, bag_ft_2)
                con_logits = posi.reshape(-1, 1)
                for previous_model in prev_models:
                    previous_model.cuda()
                    _, _, _, bag_ft_3 = self.mil_run(previous_model,
                                                     images,
                                                     labels,
                                                     self.mil_loss,
                                                     return_feature=True)
                    nega = self.cos_loss(bag_ft_1, bag_ft_3)
                    con_logits = torch.cat((con_logits, nega.reshape(-1, 1)), dim=1)
                con_logits /= self.args.temperature
                con_labels = torch.zeros(labels.size(0)).cuda().long()
                loss_con = self.args.contrast_mu * self.CE_loss(con_logits, con_labels)
                loss += loss_con

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs