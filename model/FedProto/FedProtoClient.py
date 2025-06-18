from copy import deepcopy

from torch import Tensor

from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim


class FedProtoAgent(AgentBase):
    def __init__(self, args, global_model, logger, MIL_pool):
        super().__init__(args, global_model, logger, MIL_pool)

    def local_train(self, agent_idx, global_proto):
        self.turn_on_training()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            agg_protos_label = {}
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.local_model.zero_grad()
                loss, error, y_prob, bag_feature = self.mil_run(self.local_model, images, labels, self.mil_loss, return_feature=True)
                # apply proto loss
                if len(global_proto) == 0:
                    loss_proto = Tensor([0.]).to(self.device)
                else:
                    proto_new = deepcopy(bag_feature.data)
                    i = 0
                    for label in labels:
                        if label.item() in global_proto.keys():
                            proto_new[i, :] = global_proto[label.item()][0].data
                        i += 1
                    loss_proto = self.dist_loss(proto_new, bag_feature)

                loss = loss + loss_proto * self.args.ld_proto
                loss.backward()
                optimizer.step()

                for i in range(len(labels)):
                    if labels[i].item() in agg_protos_label:
                        agg_protos_label[labels[i].item()].append(bag_feature[i,:])
                    else:
                        agg_protos_label[labels[i].item()] = [bag_feature[i,:]]
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
            print(f'Agent: {agent_idx}, Iter: {iter},Loss: {batch_loss}')
            self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Loss: {batch_loss}')
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs, agg_protos_label
