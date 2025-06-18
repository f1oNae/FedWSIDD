import os
# from timm.models.helpers import update_pretrained_cfg_and_kwargs, load_pretrained, load_custom_pretrained
# from timm.models.vision_transformer import _cfg, default_cfgs, checkpoint_filter_fn, build_model_with_cfg
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import timm
from copy import deepcopy
from model.resnet_custom import resnet50_baseline
# from model.ViT_model import ViT
from torch.utils.data import DataLoader
from utils.trainer_util import calculate_error, compute_accuracy, RAdam, FeatMag, average_weights, get_mdl_params, set_client_from_params
from utils.data_utils import get_split_loader, CategoriesSampler
from utils.Get_model import define_model
from utils.Get_data import define_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def raw_feature_extract(args, data):
    if 'ViT' in args.ft_model:
        vit_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=3, global_pool='avg',
                          num_classes=2)
        # variant = "vit_small_patch16_224"
        # # custom_pretrained = '/scratch/iq24/cc0395/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth'
        # feature_extractor = ViT(None, vit_kwargs, variant, True).to(device).eval()  # ,
        feature_extractor = timm.create_model('vit_small_patch16_224', pretrained=True).to(device).eval()
        trsforms = [transforms.Resize(224)]  # transforms.Resize(256)
    else:
        feature_extractor = resnet50_baseline(pretrained=True).to(device).eval()
        trsforms = []  # transforms.Resize(256)

    trsforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    trsforms = transforms.Compose(trsforms)
    if len(data.size()) == 5:#n_slide, n_patch, c, h, w
        data_feature = []
        for slide_idx in range(data.size(0)):
            slide_data = data[slide_idx]
            slide_data = trsforms(slide_data)
            slide_data = feature_extractor(slide_data)
            data_feature.append(slide_data)
        data_feature = torch.cat(data_feature, dim=0)
    else:
        data = trsforms(data)
        data_feature = feature_extractor(data)
    return data_feature

def clam_runner(args,
                model,
                data,
                label,
                loss_fn,
                instance_eval=True,
                return_lgt=False,
                return_feature=False,
                custom_input=None,
                raw_image=False,
                aug_feature=None):
    if raw_image:
        data = raw_feature_extract(args, data)
    # print('Augment feature ', aug_feature)
    logits, Y_prob, Y_hat, instance_dict = model(data,
                                                    label=label,
                                                    instance_eval=instance_eval,
                                                    return_features=return_feature,
                                                    custom_features=custom_input,
                                                    augment_features=aug_feature)
    if not instance_eval:
        return logits

    loss = loss_fn(logits, label)
    instance_loss = instance_dict['instance_loss']
    total_loss = args.bag_weight * loss + (1 - args.bag_weight) * instance_loss

    error = calculate_error(Y_hat, label)
    if return_feature and return_lgt:
        return total_loss, error, Y_prob, instance_dict['features'], logits
    elif return_feature and not return_lgt:
        return total_loss, error, Y_prob, instance_dict['features']
    elif not return_feature and return_lgt:
        return total_loss, error, Y_prob, logits
    else:
        return total_loss, error, Y_prob

def acmil_runner(model, data, label, loss_fn):
    n_token = 5
    sub_preds, slide_preds, attn = model(data)
    loss0 = loss_fn(sub_preds, label.repeat_interleave(n_token))
    loss1 = loss_fn(slide_preds, label)
    diff_loss = torch.tensor(0).to(device, dtype=torch.float)
    attn = torch.softmax(attn, dim=-1)
    for i in range(n_token):
        for j in range(i + 1, n_token):
            diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                        n_token * (n_token - 1) / 2)
    total_loss = diff_loss + loss0 + loss1
    pred = torch.argmax(slide_preds, dim=-1)
    # pred = torch.softmax(slide_preds, dim=-1)
    error = calculate_error(pred, label)
    return total_loss, error, slide_preds[:, 1]

def hipt_runner(model, data, label, loss_fn):
    data = data.unsqueeze(0)
    logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
    total_loss = loss_fn(logits, label)
    error = calculate_error(Y_hat, label)
    return total_loss, error, Y_prob

def transmil_runner(args, model, data, label, loss_fn, return_lgt=False, return_feature=False, raw_image=False):
    if raw_image:
        data = raw_feature_extract(args, data)
    data = data.unsqueeze(0)
    results_dict = model(data)
    logits = results_dict['logits']
    Y_hat = results_dict['Y_hat']
    Y_prob = results_dict['Y_prob']
    feature = results_dict['features']
    total_loss = loss_fn(logits, label)
    error = calculate_error(Y_hat, label)
    if return_feature and return_lgt:
        return total_loss, error, Y_prob, feature, logits
    elif return_feature and not return_lgt:
        return total_loss, error, Y_prob, feature
    elif not return_feature and return_lgt:
        return total_loss, error, Y_prob, logits
    else:
        return total_loss, error, Y_prob

def abmil_runner(args, model, data, label, loss_fn, return_lgt=False, return_feature=False, raw_image=False):
    if raw_image:
        data = raw_feature_extract(args, data)
    logits, Y_prob, Y_hat, A, feature = model.forward(data)
    total_loss = loss_fn(logits, label)
    error = 1. - Y_hat.eq(label).cpu().float().mean().item()
    if return_feature and return_lgt:
        return total_loss, error, Y_prob, feature, logits
    elif return_feature and not return_lgt:
        return total_loss, error, Y_prob, feature
    elif not return_feature and return_lgt:
        return total_loss, error, Y_prob, logits
    else:
        return total_loss, error, Y_prob

def frmil_runner(args, model, data, label, loss_fn, bce_weight, ce_weight):
    norm_idx = torch.where(label.cpu() == 0)[0].numpy()[0]
    ano_idx = 1 - norm_idx
    if args.drop_data:
        data = F.dropout(data, p=0.20)
    logits, query, max_c = model(data)
    # all losses
    max_c = torch.max(max_c, 1)[0]
    loss_max = F.binary_cross_entropy(max_c, label.float(), weight=bce_weight)
    loss_bag = F.cross_entropy(logits, label, weight=ce_weight)
    loss_ft = loss_fn(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0),
                      w_scale=query.shape[1])
    loss = (loss_bag + loss_ft + loss_max) * (1. / 3)
    acc = compute_accuracy(logits, label)
    return loss, 1 - acc / 100


class Meta(nn.Module):
    def __init__(self, args, logger=None):
        super(Meta, self).__init__()
        self.args = args
        self.global_model = define_model(args)
        self.logger = logger
        self.logger.info(' '.join(f'--{k}={v} \n' for k, v in vars(args).items()))
        self.device = device

    def get_optim(self, model, alpha=None):
        params_gather = []
        mommen = self.args.reg if alpha is None else alpha + self.args.reg
        params_gather.append(
            {'params': filter(lambda p: p.requires_grad, model.parameters()),
             'lr': self.args.lr,
             'weight_decay': mommen}
        )
        if self.args.opt == "adam":
            optimizer = optim.Adam(params_gather)
        elif self.args.opt == 'adamw':
            optimizer = optim.AdamW(params_gather)
        elif self.args.opt == 'sgd':

            optimizer = optim.SGD(params_gather, momentum=mommen, nesterov=True)
        elif self.args.opt == 'radam':
            optimizer = RAdam(params_gather)
        else:
            raise NotImplementedError
        return optimizer

    def get_loss(self):
        if self.args.bag_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            loss_fn = SmoothTop1SVM(n_classes=self.args.n_classes)
            loss_fn = loss_fn.cuda()
        elif self.args.bag_loss == 'mag':
            loss_fn = FeatMag(margin=self.args.mag).cuda()
        else:
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def clam_runner(self, model, data, label, loss_fn, return_feature=False):
        logits, Y_prob, Y_hat, _, instance_dict = model(data,
                                                        label=label,
                                                        instance_eval=True,
                                                        return_features=return_feature)
        loss = loss_fn(logits, label)
        instance_loss = instance_dict['instance_loss']
        total_loss = self.args.bag_weight * loss + (1 - self.args.bag_weight) * instance_loss
        error = calculate_error(Y_hat, label)
        if return_feature:
            return total_loss, error, instance_dict['features']
        return total_loss, error

    def hipt_runner(self, model, data, label, loss_fn):
        data = data.unsqueeze(0)
        logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
        total_loss = loss_fn(logits, label)
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def transmil_runner(self, model, data, label, loss_fn):
        data = data.unsqueeze(0)
        results_dict = model(data=data, label=label)
        logits = results_dict['logits']
        Y_hat = results_dict['Y_hat']
        total_loss = loss_fn(logits, label)
        error = calculate_error(Y_hat, label)
        return total_loss, error

    def abmil_runner(self, model, data, label, loss_fn):
        logits, Y_prob, Y_hat, A = model.forward(data)
        total_loss = loss_fn(logits, label)
        error = 1. - Y_hat.eq(label).cpu().float().mean().item()
        return total_loss, error

    def frmil_runner(self, model, data, label, loss_fn, bce_weight, ce_weight):
        norm_idx = torch.where(label.cpu() == 0)[0].numpy()[0]
        ano_idx = 1 - norm_idx
        if self.args.drop_data:
            data = F.dropout(data, p=0.20)
        logits, query, max_c = model(data)

        # all losses
        max_c = torch.max(max_c, 1)[0]
        loss_max = F.binary_cross_entropy(max_c, label.float(), weight=bce_weight)
        loss_bag = F.cross_entropy(logits, label, weight=ce_weight)
        loss_ft = loss_fn(query[ano_idx, :, :].unsqueeze(0), query[norm_idx, :, :].unsqueeze(0),
                           w_scale=query.shape[1])
        loss = (loss_bag + loss_ft + loss_max) * (1. / 3)
        acc = compute_accuracy(logits, label)

        return loss, 1 - acc/100

    def get_train_loader(self, train_dataset):
        if 'frmil' in self.args.mil_method:
            train_sampler = CategoriesSampler(train_dataset.labels,
                                              n_batch=len(train_dataset.slide_data),
                                              n_cls=self.args.n_classes,
                                              n_per=1)
            train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
        else:
            train_loader = get_split_loader(train_dataset, training=True, weighted=self.args.weighted_sample)
        return train_loader

    def get_test_loader(self, test_dataset):
        if 'frmil' in self.args.mil_method:
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            test_loader = get_split_loader(test_dataset)
        return test_loader

    def local_update(self, agent_idx, train_dataset, model, loss_fn):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)

        model.to(device)
        model.train()
        optimizer = self.get_optim(model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = self.clam_runner(model, images, labels, loss_fn)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError
                if self.args.fed_method == 'fed_prox':
                    proximal_loss = (self.args.mu/2) * sum((w - v).norm(2) for w, v in zip(model.parameters(), self.global_model.parameters()))
                    loss += proximal_loss
                loss.backward()
                optimizer.step()

                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
            batch_loss /= len(local_train_loader)
            batch_error /= len(local_train_loader)
            epoch_loss += batch_loss
        return model.state_dict(), epoch_loss/self.args.local_epochs

    def local_inference(self, agent_idx, test_dataset, model, loss_fn):
        local_test_dataset = test_dataset[agent_idx]
        local_test_loader = self.get_test_loader(local_test_dataset)

        model.to(device)
        model.eval()
        total_loss = 0.
        total_error = 0.
        for batch_idx, (images, labels) in enumerate(local_test_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            if 'CLAM' in self.args.mil_method:
                loss, error = self.clam_runner(model, images, labels, loss_fn)
            else:
                self.logger.error(f'{self.args.mil_method} not implemented')
                raise NotImplementedError

            total_loss += loss.item()
            total_error += error
        total_loss /= len(local_test_loader)
        total_error /= len(local_test_loader)
        return total_loss, total_error

    def forward_fedavg(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        self.global_model.to(device)
        self.global_model.train()

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_weights, local_losses = [], []
            # print(f'\n | Global Training Round : {epoch + 1} |\n')
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.train()

            for idx in agents:
                w, agent_loss = self.local_update(idx,
                                                  train_dataset,
                                                  deepcopy(self.global_model),
                                                  loss_fn)
                local_weights.append(deepcopy(w))
                local_losses.append(deepcopy(agent_loss))

            # update global weights
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                               test_dataset,
                                               self.global_model,
                                               loss_fn)
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

    def local_update_feddyn(self, agent_idx,
                            train_dataset,
                            model, loss_fn,
                            alpha_coef_adpt,
                            cld_mdl_param_tensor,
                            local_param_list_curr):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)

        model.to(device)
        model.train()
        optimizer = self.get_optim(model, alpha=alpha_coef_adpt)

        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = self.clam_runner(model, images, labels, loss_fn)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                loss_algo = alpha_coef_adpt * torch.sum(
                    local_par_list * (-cld_mdl_param_tensor + local_param_list_curr))
                # current_local_parameter * (last_step_local_parameter - global_parameter)
                loss = loss + loss_algo
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=10)  # Clip gradients
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                batch_loss /= len(local_train_loader)
            epoch_loss += batch_loss
            model.train()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        return model, epoch_loss/self.args.local_epochs

    def local_update_fedscaf(self, agent_idx,
                            train_dataset,
                            model, loss_fn,
                            state_params_diff_curr):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)

        model.to(device)
        model.train()
        optimizer = self.get_optim(model)
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error = self.clam_runner(model, images, labels, loss_fn)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError

                # Get linear penalty on the current parameter estimates
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                loss_algo = torch.sum(local_par_list * state_params_diff_curr)
                loss = loss + loss_algo
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=10)  # Clip gradients
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                batch_loss /= len(local_train_loader)
            epoch_loss += batch_loss
            model.train()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        return model, epoch_loss / self.args.local_epochs

    def forward_feddyn(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        n_clnt = len(train_dataset)
        weight_list = np.asarray([len(train_dataset[i]) for i in range(n_clnt)])
        weight_list = weight_list / np.sum(weight_list) * n_clnt

        n_par = len(get_mdl_params([self.global_model])[0])
        local_param_list = np.zeros((n_clnt, n_par)).astype('float32') # [n_clnt X n_par]
        init_par_list = get_mdl_params([self.global_model], n_par)[0]
        clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                        -1)  # [n_clnt X n_par]
        clnt_models = list(range(n_clnt))
        avg_model = deepcopy(self.global_model).to(device)
        cld_mdl_param = get_mdl_params([avg_model], n_par)[0]

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_losses = []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            cld_mdl_param_tensor = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device) # [n_par]
            for idx in agents:
                # Warm start from current avg model
                clnt_models[idx] = deepcopy(avg_model).to(device)
                model = clnt_models[idx]
                for params in model.parameters():
                    params.requires_grad = True

                alpha_coef_adpt = self.args.alpha_coef / weight_list[idx]  # adaptive alpha coef
                local_param_list_curr = torch.tensor(local_param_list[idx], dtype=torch.float32, device=device)
                local_trained_model, agent_loss = self.local_update_feddyn(idx,
                                                                            train_dataset,
                                                                            model,
                                                                            loss_fn,
                                                                            alpha_coef_adpt,
                                                                            cld_mdl_param_tensor,
                                                                            local_param_list_curr)
                clnt_models[idx] = local_trained_model
                curr_model_par = get_mdl_params([clnt_models[idx]], n_par)[0]

                # No need to scale up hist terms. They are -\nabla/alpha and alpha is already scaled.
                local_param_list[idx] += curr_model_par-cld_mdl_param
                clnt_params_list[idx] = curr_model_par

                local_losses.append(agent_loss)

            avg_mdl_param = np.mean(clnt_params_list, axis = 0)
            cld_mdl_param = avg_mdl_param + np.mean(local_param_list, axis=0)
            avg_model = set_client_from_params(self.global_model, avg_mdl_param)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            avg_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                               test_dataset,
                                               avg_model,
                                               loss_fn)
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
                    best_model = deepcopy(avg_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)
        return best_accuracy, best_accuracy_per_agent

    def forward_fedscaf(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        n_clnt = len(train_dataset)
        weight_list = np.asarray([len(train_dataset[i]) for i in range(n_clnt)])
        weight_list = weight_list / np.sum(weight_list) * n_clnt

        n_par = len(get_mdl_params([self.global_model])[0])
        state_param_list = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
        init_par_list = get_mdl_params([self.global_model], n_par)[0]
        clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1,
                                                                                                    -1)  # n_clnt X n_par
        clnt_models = list(range(n_clnt))
        avg_model = deepcopy(self.global_model).to(device)

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_losses = []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            delta_c_sum = np.zeros(n_par)
            prev_params = get_mdl_params([avg_model], n_par)[0]

            for idx in agents:
                # Warm start from current avg model
                clnt_models[idx] = deepcopy(avg_model).to(device)
                model = clnt_models[idx]
                for params in model.parameters():
                    params.requires_grad = True

                # Scale down c
                state_params_diff_curr = torch.tensor(
                    -state_param_list[idx] + state_param_list[-1] / weight_list[idx], dtype=torch.float32,
                    device=device)

                local_trained_model, agent_loss = self.local_update_fedscaf(idx,
                                                                            train_dataset,
                                                                            model,
                                                                            loss_fn,
                                                                            state_params_diff_curr)
                clnt_models[idx] = local_trained_model
                curr_model_par = get_mdl_params([clnt_models[idx]], n_par)[0]

                new_c = state_param_list[idx] - state_param_list[-1] + 1 / self.args.global_epochs / self.args.lr * (
                            prev_params - curr_model_par)
                delta_c_sum += (new_c - state_param_list[idx]) * weight_list[idx]
                state_param_list[idx] = new_c
                clnt_params_list[idx] = curr_model_par
                local_losses.append(agent_loss)

            avg_model_params = np.mean(clnt_params_list, axis=0)
            state_param_list[-1] += 1 / n_clnt * delta_c_sum
            avg_model = set_client_from_params(self.global_model, avg_model_params)
            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            avg_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                               test_dataset,
                                               avg_model,
                                               loss_fn)
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
                    best_model = deepcopy(avg_model)
                    torch.save(best_model.state_dict(), best_model_save_pth)
        return best_accuracy, best_accuracy_per_agent

    def local_update_fedmoon(self,
                             train_dataset,
                             agent_idx,
                             loss_fn,
                             model,
                             global_model = None,
                             prev_model_pool = None,):
        local_train_dataset = train_dataset[agent_idx]
        local_train_loader = self.get_train_loader(local_train_dataset)
        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][agent_idx])
        # performing model contrastive learning
        model.train()
        optimizer = self.get_optim(model)
        for previous_model in prev_models:
            previous_model.cuda()
        cos = torch.nn.CosineSimilarity(dim=-1)
        criterion_con = nn.CrossEntropyLoss().cuda()
        epoch_loss = 0.
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            for batch_idx, (images, labels) in enumerate(local_train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                model.zero_grad()
                optimizer.zero_grad()
                if 'CLAM' in self.args.mil_method:
                    loss, error, bag_ft_1 = self.clam_runner(model, images, labels, loss_fn, return_feature=True)
                    _, _, bag_ft_2 = self.clam_runner(global_model, images, labels, loss_fn, return_feature=True)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError
                # contrastive loss = max_dis(local, prev_local)+min_dis(local, global)
                posi = cos(bag_ft_1, bag_ft_2)
                con_logits = posi.reshape(-1, 1)
                for previous_model in prev_models:
                    previous_model.cuda()
                    _, _, bag_ft_3 = self.clam_runner(previous_model, images, labels, loss_fn, return_feature=True)
                    nega = cos(bag_ft_1, bag_ft_3)
                    con_logits = torch.cat((con_logits, nega.reshape(-1, 1)), dim=1)
                con_logits /= self.args.temperature
                con_labels = torch.zeros(labels.size(0)).cuda().long()
                loss_con = self.args.contrast_mu * criterion_con(con_logits, con_labels)
                loss += loss_con

                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                               max_norm=10)  # Clip gradients
                optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
                batch_loss /= len(local_train_loader)
            epoch_loss += batch_loss
            model.train()
        # Freeze model
        for params in model.parameters():
            params.requires_grad = False
        model.eval()
        return model, epoch_loss / self.args.local_epochs


    def forward_fedmoon(self, iter):
        train_dataset, test_dataset, agents = define_data(self.args, self.logger)
        print('\nInit loss ...', end=' ')
        loss_fn = self.get_loss()
        print('Done!')

        n_clnt = len(train_dataset)
        weight_list = np.asarray([len(train_dataset[i]) for i in range(n_clnt)])
        weight_list = weight_list / np.sum(weight_list) * n_clnt
        local_models = {}
        for i in range(n_clnt):
            local_models[i] = define_model(self.args)
            local_models[i].to(device)
        old_models_pool = []
        if len(old_models_pool) < self.args.model_buffer_size:
            old_nets = deepcopy(local_models)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False

        train_loss = []
        best_accuracy = 0.
        best_accuracy_per_agent = []
        best_model_save_pth = os.path.join(self.args.results_dir, "best_model_%d.pt" % iter)
        for epoch in range(self.args.global_epochs):
            local_losses = []
            self.logger.info(f'\n | Global Training Round : {epoch + 1} |\n')
            self.global_model.eval()
            for param in self.global_model.parameters():
                param.requires_grad = False
            global_w = self.global_model.state_dict()

            # since we do not have enought of agents, we use them all
            nets_this_round = {k: local_models[k] for k in local_models}
            for net in nets_this_round.values():
                net.load_state_dict(global_w)

            # perform local training
            for idx in agents:
                local_model = nets_this_round[idx]
                local_model, agent_loss = self.local_update_fedmoon(train_dataset,
                                                               idx,
                                                               loss_fn,
                                                               local_model,
                                                               global_model=self.global_model,
                                                               prev_model_pool=old_models_pool)
                nets_this_round[idx] = local_model
                local_losses.append(agent_loss)
            for net_id, net in enumerate(nets_this_round.values()):
                net_para = net.state_dict()
                if net_id == 0:
                    for key in net_para:
                        global_w[key] = net_para[key] * weight_list[net_id]
                else:
                    for key in net_para:
                        global_w[key] += net_para[key] * weight_list[net_id]

            # update model on server
            self.global_model.load_state_dict(global_w, strict=False)
            if len(old_models_pool) < self.args.model_buffer_size:
                old_nets = deepcopy(local_models)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                old_models_pool.append(old_nets)
            elif self.args.pool_option == 'FIFO':
                old_nets = deepcopy(local_models)
                for _, net in old_nets.items():
                    net.eval()
                    for param in net.parameters():
                        param.requires_grad = False
                for i in range(self.args.model_buffer_size - 2, -1, -1):
                    old_models_pool[i] = old_models_pool[i + 1]
                old_models_pool[self.args.model_buffer_size - 1] = old_nets

            # Calculate avg training accuracy over all users at every epoch
            list_acc, list_loss = [], []
            self.global_model.eval()
            for idx in agents:
                agent_loss, agent_error = self.local_inference(idx,
                                                               test_dataset,
                                                               self.global_model,
                                                               loss_fn)
                list_acc.append(1 - agent_error)
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

import math
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all',
                base_temperature=0.07, device=None):
        super(SupConLoss, self).__init__()
        # self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, features, labels=None, temperature=0.07, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), temperature)
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast}")
        # logging.info(f"In SupCon, anchor_dot_contrast.shape: {anchor_dot_contrast.shape}, anchor_dot_contrast: {anchor_dot_contrast.mean()}")
        # logging.info(f"In SupCon, anchor_dot_contrast.device: {anchor_dot_contrast.device}, self.device: {self.device}")


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # logging.info(f"In SupCon, exp_logits.shape: {exp_logits.shape}, exp_logits: {exp_logits.mean()}")
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # if torch.any(torch.isnan(log_prob)):
        #     log_prob[torch.isnan(log_prob)] = 0.0
        # logging.info(f"In SupCon, log_prob.shape: {log_prob.shape}, log_prob: {log_prob.mean()}")

        mask_sum = mask.sum(1)
        mask_sum[mask_sum == 0] += 1

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # loss
        loss = - (temperature / self.base_temperature) * mean_log_prob_pos
        # loss[torch.isnan(loss)] = 0.0
        if torch.any(torch.isnan(loss)):
            # loss[torch.isnan(loss)] = 0.0
            logging.info(f"In SupCon, features.shape: {features.shape}, loss: {loss}")
            raise RuntimeError
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class Distance_loss(nn.Module):
    def __init__(self, distance="SupCon", device=None):
        super(Distance_loss, self).__init__()
        self.distance = distance
        self.device = device
        if self.distance == "SupCon":
            self.supcon_loss = SupConLoss(contrast_mode='all', base_temperature=0.07, device=self.device)
        else:
            self.supcon_loss = None


    def forward(self, x1, x2, label1=None, label2=None):
        if self.distance == "L2_norm":
            loss = self.L2_norm(x1, x2)
        elif self.distance == "cosine":
            loss = self.cosine(x1, x2)
        elif self.distance == "SupCon":
            loss = self.supcon(x1, x2, label1, label2)
        else:
            raise NotImplementedError
        return loss


    def L2_norm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    def cosine(self, x1, x2):
        cos = F.cosine_similarity(x1, x2, dim=-1)
        loss = 1 - cos.mean()
        return loss

    def supcon(self, feature1, feature2, label1, label2):

        all_features = torch.cat([feature1, feature2], dim=0)

        all_features = F.normalize(all_features, dim=1)
        all_features = all_features.unsqueeze(1)

        align_cls_loss = self.supcon_loss(
            features=all_features,
            labels=torch.cat([label1, label2], dim=0),
            temperature=0.07, mask=None)
        return align_cls_loss











class MMD_loss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                        for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                    for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

def custom_load_pretrained(model, model_name, custom_pretrained):
    print('Custom load pretrained model from ', custom_pretrained)
    ckp = torch.load(custom_pretrained)
    from collections import OrderedDict
    pretrained_state_dict = OrderedDict()
    if 'dino' in model_name:
        print('Load DINO model from ', custom_pretrained)
        new_state_dict = model.state_dict()
        for k, v in ckp['teacher'].items():
            name_list = k.split('.')
            if name_list[0] == 'backbone':
                name = '.'.join(name_list[1:])
            else:
                name = k
            pretrained_state_dict[name] = v
        for k, v in model.state_dict().items():
            if k not in pretrained_state_dict:
                print(k)
                new_state_dict[k] = v
            else:
                new_state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(new_state_dict, strict=True)
    elif 'simclr' in model_name:
        print('Load SimCLR model from ', custom_pretrained)
        new_state_dict = model.state_dict()
        for k, v in ckp.items():
            name_list = k.split('.')
            if 'module' in k and 'features' in k:
                name = '.'.join(name_list[2:])
            elif 'module' in k:
                name = '.'.join(name_list[1:])
            else:
                name = k
            pretrained_state_dict[name] = v
        for k, v in model.state_dict().items():
            if k not in pretrained_state_dict:
                print('Not found in pretrained: ', k)
                new_state_dict[k] = v
            else:
                new_state_dict[k] = pretrained_state_dict[k]
        model.load_state_dict(new_state_dict, strict=True)
    return model

def load_pretrained_vit(model, model_name, variant, vit_kwargs, custom_pretrained, pretrained=True):
    if pretrained:
        print('==========> Load pretrained %s'%(model.__class__.__name__))
    pretrained_cfg = deepcopy(default_cfgs[variant])
    update_pretrained_cfg_and_kwargs(pretrained_cfg, vit_kwargs, None)
    pretrained_cfg.setdefault('architecture', variant)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    num_classes_pretrained = getattr(model, 'num_classes', vit_kwargs.get('num_classes', 1000))
    pretrained_custom_load = 'npz' in pretrained_cfg['url']
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=vit_kwargs.get('in_chans', 3),
                filter_fn=checkpoint_filter_fn,
                strict=False)
    if 'dino' in model_name:
        model = custom_load_pretrained(model, model_name, custom_pretrained)
    return model