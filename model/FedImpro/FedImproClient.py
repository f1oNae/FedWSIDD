import torch
from model.Client import AgentBase

from utils.trainer_util import get_optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FedImproAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)

    def local_train(self, agent_idx, augment_feature=None):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        # fake_data = torch.randn(1, self.local_model.size[0])
        # feature_generator = FeatureGenerator(self.args, device)
        # with torch.no_grad():
        #     fake_data = fake_data.to(device)
        #     loss, error, y_prob, feat_list = self.mil_run(self.local_model,
        #                                                  fake_data,
        #                                                  [0],
        #                                                  self.mil_loss,
        #                                                  return_feature=True,
        #                                                  aug_feature=augment_feature)
        #     feature_generator.initial_model_params(feat_list[0].to("cpu"), self.local_model.size[0])
        if augment_feature is not None:
            self.logger.info(f'========> Using augmented feature {augment_feature.size()}')
            # print('Augment_feature is used ', augment_feature.size())
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            local_ft_list = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                loss, error, y_prob, local_ft = self.mil_run(self.local_model,
                                                   images,
                                                   labels,
                                                   self.mil_loss,
                                                   return_feature=True,
                                                   aug_feature=augment_feature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                local_ft_list.append(local_ft)
            local_ft_list = torch.cat(local_ft_list, dim=0)
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs, local_ft_list
