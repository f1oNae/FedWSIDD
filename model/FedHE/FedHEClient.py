from copy import deepcopy

from torch import Tensor

from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim


class FedHEAgent(AgentBase):
    def __init__(self, args, global_model, logger, MIL_pool):
        super().__init__(args, global_model, logger, MIL_pool)

    def local_train(self, agent_idx, global_lgt):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            agg_lgt_label = {}
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                loss, error, y_prob, bag_lgt = self.mil_run(self.local_model, images, labels, self.mil_loss, return_lgt=True)
                # apply proto loss
                if len(global_lgt) == 0:
                    loss_lgt = Tensor([0.]).to(self.device)
                else:
                    lgt_new = deepcopy(bag_lgt.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_lgt.keys():
                            lgt_new[i, :] = global_lgt[label.item()][0].data
                        i += 1
                    loss_lgt = self.dist_loss(lgt_new, bag_lgt)

                loss = loss + loss_lgt * self.args.ld_proto
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_lgt_label:
                        agg_lgt_label[labels[i].item()].append(bag_lgt[i,:])
                    else:
                        agg_lgt_label[labels[i].item()] = [bag_lgt[i,:]]
                # if batch_idx % 20 == 0:
                #     self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}, Loss Proto: {loss_lgt.item()}')
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs, agg_lgt_label
