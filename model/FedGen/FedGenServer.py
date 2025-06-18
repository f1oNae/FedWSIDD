from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from model.Server import ServerBase
from model.FedGen.FedGenClient import FedGenAgent
from model.FedGen.Generator import Generator
import os


class FedGen(ServerBase):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.generator = Generator(dataset=args.task).to(self.device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generator.parameters(),
            lr=self.args.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.args.reg, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.global_model.parameters(),
            lr=self.args.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)
        self.setup_clients()
        self.n_teacher_iters = 5

    def setup_clients(self):
        self.clients = []
        for idx in self.n_clients:
            curr_client = FedGenAgent(self.args, self.global_model, self.generator, self.logger)
            curr_client.init_dataset(self.train_dataset[idx], self.test_dataset[idx])
            self.clients.append(curr_client)

    def run(self, iter):
        # include training and testing
        self.global_model.to(self.device)
        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_weights, local_losses = [], []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()
            for idx in self.n_clients:
                # send global model to all clients
                self.clients[idx].update_local_model(self.global_model)
                w, agent_loss = self.clients[idx].local_train(idx, epoch)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))
            self.logger.info(f'\n | Generator Training Start |\n')
            self.train_generator(1, self.args.ensemble_epochs)
            # update global weights
            self.aggregate_parameter(local_weights, method='average')
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # dispatch global model to all clients
            for idx in self.n_clients:
                self.clients[idx].update_local_model(self.global_model)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in self.n_clients:
                agent_loss, agent_error = self.clients[idx].local_test(self.global_model)
                list_acc.append(1-agent_error)
                list_loss.append(agent_loss)
            train_acc = sum(list_acc) / len(list_acc)
            if (epoch + 1) % 1 == 0:
                self.logger.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
                self.logger.info(f'Training Loss : {np.mean(np.array(train_loss))}')
                self.logger.info('Train Accuracy: {:.2f}% \n'.format(100 * train_acc))
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    best_accuracy_per_agent = list_acc
                    best_model = deepcopy(self.global_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)

        return best_accuracy, best_accuracy_per_agent

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.args.n_classes):
            weights = []
            for user in self.clients:
                weights.append(user.label_counts(label))
            if np.max(weights) > 1:
                qualified_labels.append(label)
            # uniform
            label_weights.append( np.array(weights) / np.sum(weights) )
        label_weights = np.array(label_weights).reshape((self.args.n_classes, -1))
        return label_weights, qualified_labels

    #TODO
    def train_generator(self, batch_size, epoches=1, latent_layer_idx=-1, verbose=False):
        TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS, STUDENT_LOSS2 = 0, 0, 0, 0
        self.label_weights, self.qualified_labels = self.get_label_weights()
        def update_generator_(n_iters, student_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS):
            self.generator.train()
            student_model.eval()
            for i in range(n_iters):
                self.generative_optimizer.zero_grad()
                y=np.random.choice(self.qualified_labels, batch_size)
                y_input=torch.LongTensor(y).to(self.device)
                ## feed to generator
                gen_result=self.generator(y_input, latent_layer_idx=latent_layer_idx, verbose=True)
                # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                gen_output, eps=gen_result['output'], gen_result['eps']
                ##### get losses ####
                # decoded = self.generative_regularizer(gen_output)
                # regularization_loss = beta * self.generative_model.dist_loss(decoded, eps) # map generated z back to eps
                diversity_loss=self.generator.diversity_loss(eps, gen_output)  # encourage different outputs

                ######### get teacher loss ############
                teacher_loss=0
                teacher_logit=0
                for user_idx in self.n_clients:
                    user = self.clients[user_idx]
                    user.local_model.eval()
                    weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
                    expand_weight=np.tile(weight, (1, self.args.n_classes))
                    logits, Y_prob, Y_hat, instance_dict=user.local_model(gen_output,
                                                    label=y_input,
                                                    instance_eval=False,
                                                    custom_features=gen_output)
                    user_output_logp_=F.log_softmax(logits, dim=1)
                    teacher_loss_=torch.mean(self.generator.crossentropy_loss(user_output_logp_, y_input) * \
                        torch.tensor(weight, dtype=torch.float32).to(self.device))
                    teacher_loss+=teacher_loss_
                    teacher_logit+=logits * torch.tensor(expand_weight, dtype=torch.float32).to(self.device)

                ######### get student loss ############
                student_logits, _, _, _ = student_model(gen_output,
                                                           label=y_input,
                                                           instance_eval=False,
                                                           custom_features=gen_output)
                student_loss=F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logit, dim=1))
                if self.args.ensemble_beta > 0:
                    loss=self.args.ensemble_alpha * teacher_loss - self.args.ensemble_beta * student_loss + self.args.ensemble_eta * diversity_loss
                else:
                    loss=self.args.ensemble_alpha * teacher_loss + self.args.ensemble_eta * diversity_loss
                loss.backward()
                self.generative_optimizer.step()
                TEACHER_LOSS += self.args.ensemble_alpha * teacher_loss.cpu().detach().item()#(torch.mean(TEACHER_LOSS.double())).item()
                STUDENT_LOSS += self.args.ensemble_beta * student_loss.cpu().detach().item()#(torch.mean(student_loss.double())).item()
                DIVERSITY_LOSS += self.args.ensemble_eta * diversity_loss.cpu().detach().item()#(torch.mean(diversity_loss.double())).item()
            return TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS

        for i in range(epoches):
            TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS=update_generator_(
                self.n_teacher_iters, self.global_model, TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)

        TEACHER_LOSS = TEACHER_LOSS / (self.n_teacher_iters * epoches)
        STUDENT_LOSS = STUDENT_LOSS/ (self.n_teacher_iters * epoches)
        DIVERSITY_LOSS = DIVERSITY_LOSS / (self.n_teacher_iters * epoches)
        info="Generator: Teacher Loss= {:.4f}, Student Loss= {:.4f}, Diversity Loss = {:.4f}, ". \
            format(TEACHER_LOSS, STUDENT_LOSS, DIVERSITY_LOSS)
        self.logger.info(info)
        self.generative_lr_scheduler.step()