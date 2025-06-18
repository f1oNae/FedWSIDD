from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim


class FedAvgAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)

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
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs
