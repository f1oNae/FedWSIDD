import numpy as np
import torch
from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim
import torch.nn.functional as F


class FedGenAgent(AgentBase):
    def __init__(self, args, global_model, generator_model, logger):
        super().__init__(args, global_model, logger)
        self.generator_model = generator_model

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def local_train(self, agent_idx, epoch):
        self.turn_on_training()
        self.generator_model.eval()
        optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        TEACHER_LOSS, DIST_LOSS, LATENT_LOSS = 0, 0, 0
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                n_patches = images.size(0)
                self.local_model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error, bag_lgt = clam_runner(self.args,
                                          self.local_model,
                                          images,
                                          labels,
                                          self.mil_loss,
                                          return_lgt=True)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError

                #### sample y and generate z
                if epoch > 0 and iter < 20:
                    generative_alpha = self.exp_lr_scheduler(epoch, decay=0.98, init_lr=self.args.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(epoch, decay=0.98, init_lr=self.args.generative_beta)
                    ### get generator output(latent representation) of the same label
                    gen_output = self.generator_model(labels, latent_layer_idx=-1)['output']
                    logit_given_gen = clam_runner(self.args,
                                                  self.local_model,
                                                  images,
                                                  labels,
                                                  self.mil_loss,
                                                  instance_eval=False,
                                                  return_lgt=True,
                                                  return_feature=True,
                                                  custom_input=gen_output)
                    target_p = F.softmax(logit_given_gen, dim=1).clone().detach()
                    user_latent_loss = generative_beta * self.ensemble_loss(bag_lgt, target_p)

                    sampled_y = np.random.choice(np.array([cls for cls in range(self.args.n_classes)]), n_patches)
                    sampled_y = torch.tensor(sampled_y).to(self.device)
                    gen_output = self.generator_model(sampled_y, latent_layer_idx=-1)['output']
                    user_output_logp = clam_runner(self.args,
                                                   self.local_model,
                                                   images,
                                                   labels,
                                                   self.mil_loss,
                                                   instance_eval=False,
                                                   return_lgt=True,
                                                   return_feature=True,
                                                   custom_input=gen_output)
                    teacher_loss = generative_alpha * torch.mean(
                        self.generator_model.crossentropy_loss(user_output_logp, sampled_y)
                    )
                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = 1#self.gen_batch_size / self.batch_size
                    loss = loss + gen_ratio * teacher_loss + user_latent_loss
                    TEACHER_LOSS += teacher_loss
                    LATENT_LOSS += user_latent_loss
                else:
                    #### get loss and perform optimization
                    loss = loss

                loss.backward()
                optimizer.step()
                if batch_idx % 20 == 0:
                    if epoch > 0 and iter < 20:
                        self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}, Teacher Loss: {teacher_loss.item()}, User Latent Loss: {user_latent_loss.item()}')
                    else:
                        self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs