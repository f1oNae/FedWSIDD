import time
import numpy as np
from tqdm import tqdm
from model.ViT_model import ViT
from model.resnet_custom import resnet50_baseline
import torch
from torchvision import transforms
from utils.Get_model import define_model
from utils.core_util import clam_runner
from utils.augment import DiffAug
import sys
import random

from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def distribution_matching(dm_iter, image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, nps, args=None, loss_fn=None):
    lambda_sim = 0.5

    net = define_model(args)
    net.train()
    # default we use ConvNet
    # if net == None:
    #     net = get_network('ConvNet', channel, num_classes, im_size).to(device)  # get a random model
    #     net.train()
    #     # for param in list(net.parameters()):
    #     #     param.requires_grad = False
    # elif net == 'clam':
    #     net = define_model(args)
    # else:
    #     net.train()
    #     # for param in list(net.parameters()):
    #     #     param.requires_grad = False
    #
    # embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel
    #
    # loss_avg = 0

    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        img_real = image_real[c]
        labels = torch.tensor([c] * ipc, dtype=torch.long, device=img_real.device)
        if img_real.size(0) == 0:
            continue
        img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, nps, channel, im_size, im_size))
        # print('DM image size check ', img_real.size(), img_syn.size())

        seed = int(time.time() * 1000) % 100000
        # dsa_param = ParamDiffAug()
        # img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
        # img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)

        # output_real = embed(img_real).detach()
        # output_syn = embed(img_syn)
        img_real = img_real.to(device)
        img_syn = img_syn.to(device)
        labels = labels.to(device)
        _, _, output_real = clam_runner(args,
                                         net,
                                         img_real.squeeze(0),
                                         labels[0].unsqueeze(0),
                                         loss_fn,
                                         return_feature=True)
        output_syn = []
        out_ft = []
        for slide_idx in range(ipc):
            _, _, output_syn_per_slide, data_feature = clam_runner(args,
                                                                     net,
                                                                     img_syn[slide_idx],
                                                                     labels[0].unsqueeze(0),
                                                                     loss_fn,
                                                                     return_feature=True,
                                                                     raw_image=True,
                                                                     return_raw_image=True)

            output_syn.append(output_syn_per_slide)
            out_ft.append(data_feature)
            # print('Size of output data feature', data_feature.size())
        output_syn = torch.cat(output_syn, dim=0)
        out_ft = torch.cat(out_ft, dim=0)
        output_real = output_real.detach()
        img_real = img_real.detach()
        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
        loss += torch.sum((torch.mean(img_real.squeeze(0), dim=0) - torch.mean(out_ft, dim=0)) ** 2)

    # # l2 and total variation loss
    # loss += lambda_sim * l2_norm(img_syn)
    # loss += lambda_sim * total_variation(img_syn)

    optimizer_img.zero_grad()
    loss.backward()
    # total_norm = 0.
    # for group in optimizer_img.param_groups:
    #     for param in group['params']:
    #         if param.requires_grad:
    #             total_norm += param.grad.data.norm(2).item() ** 2.
    # total_norm = total_norm ** .5
    optimizer_img.step()
    #print the gradient of the synthetic image
    print(f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    return loss.item(), image_syn  # , total_norm

def distribution_matching_woMIL(image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, nps, args=None, loss_fn=None):
    lambda_sim = 0.5
    # net = resnet50_baseline(pretrained=True).to(device).train()
    if 'ViT' in args.ft_model:
        vit_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=3, global_pool='avg',
                          num_classes=2)
        variant = "vit_small_patch16_224"
        net = ViT(None, vit_kwargs, variant, True).to(device).train()  # ,
    else:
        net = resnet50_baseline(pretrained=True).to(device).train()
    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        output_real = image_real[c].to(device)  # real images are already compressed to features
        output_real = output_real.squeeze(0)
        output_real = output_real.detach()
        if output_real.size(0) == 0:
            continue
        img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc * nps, channel, im_size, im_size))
        # form a dataloader using the synthetic images
        syn_loader = DataLoader(TensorDataset(img_syn), batch_size=ipc, shuffle=True)
        output_syn = []
        for img_syn in syn_loader:
            img_syn = img_syn[0].to(device)
            if 'ViT' in args.ft_model:
                img_syn = transforms.Resize(224)(img_syn)
            if args.sn:
                im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                aug, aug_rand = diffaug(args, mean=im_mean, std=im_std)
                img_syn = aug(img_syn)
            output_syn.append(net(img_syn))
        output_syn = torch.cat(output_syn, dim=0)

        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()
    # print the gradient of the synthetic image
    print(
        f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    return loss.item(), image_syn#,  image_syn_ft# , total_norm

class Normalize():
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
def diffaug(args, mean, std, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type

    normalize = Normalize(mean=mean, std=std, device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand
# def compute_img_mean_std_per_class(img_set, im_size, num_classes):
#     means = [torch.tensor([0.0, 0.0, 0.0]) for i in range(num_classes)]
#     vars = [torch.tensor([0.0, 0.0, 0.0]) for i in range(num_classes)]
#     counts = [0 for i in range(num_classes)]
#     for i in range(len(img_set)):
#         img, label = img_set[i]
#         means[label] += img.sum(axis=[1, 2])
#         vars[label] += (img ** 2).sum(axis=[1, 2])
#         counts[label] += 1
#
#     counts = [count * im_size[0] * im_size[1] for count in counts]
#
#     total_means = [mean / count for (mean, count) in zip(means, counts)]
#     total_vars = [(var / count) - (total_mean ** 2) for (var, total_mean, count) in zip(vars, total_means, counts)]
#     total_stds = [torch.sqrt(total_var) for total_var in total_vars]
#
#     return total_means, total_stds
#
#
# def compute_img_mean_std(img_set, im_size, num_classes):
#     mean = torch.tensor([0.0, 0.0, 0.0])
#     var = torch.tensor([0.0, 0.0, 0.0])
#     count = len(img_set) * im_size[0] * im_size[1]
#     for i in range(len(img_set)):
#         img, label = img_set[i]
#         mean += img.sum(axis=[1, 2])
#         var += (img ** 2).sum(axis=[1, 2])
#
#     total_mean = mean / count
#     total_var = (var / count) - (total_mean ** 2)
#     total_std = torch.sqrt(total_var)
#
#     return total_mean, total_std
#
#
# def get_initial_normal(train_set, im_size, num_classes, ipc):
#     # compute mean and std
#     means, stds = compute_img_mean_std_per_class(train_set, im_size, num_classes)
#     mean, std = compute_img_mean_std(train_set, im_size, num_classes)
#     # print(means)
#
#     # initialize random image
#     image_syn_classes = []
#     for c in range(num_classes):
#         image_syn1 = torch.normal(mean=means[c][0], std=stds[c][0], size=(ipc, 1, im_size[0], im_size[1]),
#                                   dtype=torch.float, requires_grad=False, device=device)  # [2*50, 1, 256, 256]
#         image_syn2 = torch.normal(mean=means[c][1], std=stds[c][1], size=(ipc, 1, im_size[0], im_size[1]),
#                                   dtype=torch.float, requires_grad=False, device=device)  # [2*50, 1, 256, 256]
#         image_syn3 = torch.normal(mean=means[c][2], std=stds[c][2], size=(ipc, 1, im_size[0], im_size[1]),
#                                   dtype=torch.float, requires_grad=False, device=device)  # [2*50, 1, 256, 256]
#         image_syn = torch.cat([image_syn1, image_syn2, image_syn3], dim=1).detach()
#         image_syn[image_syn < 0] = 0.0
#         image_syn[image_syn > 1] = 1.0
#         for ch in range(3):
#             image_syn[:, ch] = (image_syn[:, ch] - mean[ch]) / std[ch]
#         image_syn_classes.append(image_syn)
#     image_syn = torch.cat(image_syn_classes, dim=0)
#     label_syn = torch.tensor(np.array([np.ones(ipc) * i for i in range(num_classes)]), dtype=torch.long,
#                              requires_grad=False, device=device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
#     # label_syns = [copy.deepcopy(local_label_tmp).to(device) for idx in range(client_num)]
#
#     # # initializa server synthetic data (10 ipcs)
#     # server_mean_, server_std_ = [[0, 0, 0] for c in range(num_classes)], [[0, 0, 0] for c in range(num_classes)]
#     # for mean_, std_ in zip(means_, stds_):
#     #     for c in range(num_classes):
#     #         server_mean_[c][0] += mean_[c][0]/client_num
#     #         server_mean_[c][1] += mean_[c][1]/client_num
#     #         server_mean_[c][2] += mean_[c][2]/client_num
#     #         server_std_[c][0] += std_[c][0]/client_num
#     #         server_std_[c][1] += std_[c][1]/client_num
#     #         server_std_[c][2] += std_[c][2]/client_num
#     # server_image_syn = []
#     # for c in range(num_classes):
#     #     image_syn1 = torch.normal(mean=server_mean_[c][0], std=server_std_[c][0], size=(server_ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
#     #     image_syn2 = torch.normal(mean=server_mean_[c][1], std=server_std_[c][1], size=(server_ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
#     #     image_syn3 = torch.normal(mean=server_mean_[c][2], std=server_std_[c][2], size=(server_ipc, 1, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
#     #     server_image_syn.append(torch.cat([image_syn1,image_syn2,image_syn3], dim=1).detach())
#     # server_image_syn = torch.cat(server_image_syn, dim=0)
#     # server_image_syn.requires_grad = True
#     # server_image_syn = server_image_syn.to(device)
#     # # server_label_syn = torch.tensor(np.array([np.ones(server_ipc)*i for i in range(num_classes)]), dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
#
#     return image_syn, label_syn
#
#
# def total_variation(x, signed_image=True):
#     if signed_image:
#         x = torch.abs(x)
#     dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
#     dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
#     return dx + dy
#
#
# def l2_norm(x, signed_image=True):
#     if signed_image:
#         x = torch.abs(x)
#     batch_size = len(x)
#     loss_l2 = torch.norm(x.view(batch_size, -1), dim=1).mean()
#     return loss_l2
#
#
# class BNFeatureHook:
#     """
#     Implementation of the forward hook to track feature statistics and compute a loss on them.
#     Will compute mean and variance, and will use l2 as a loss
#     """
#
#     def __init__(self, module):
#         self.hook = module.register_forward_hook(self.hook_fn)
#
#     def hook_fn(self, module, input, output):
#         # hook co compute deepinversion's feature distribution regularization
#         nch = input[0].shape[1]
#         mean = input[0].mean([0, 2, 3])
#         var = (input[0].permute(1, 0, 2,
#                                 3).contiguous().view([nch,
#                                                       -1]).var(1,
#                                                                unbiased=False))
#
#         # forcing mean and variance to match between two distributions
#         # other ways might work better, i.g. KL divergence
#         r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
#             module.running_mean.data - mean, 2)
#         self.mean = mean
#         self.var = var
#         self.r_feature = r_feature
#         # must have no output
#
#     def close(self):
#         self.hook.remove()
#
#
# def deep_inversion(net, criterion, optimizer_img, image_syn, label_syn, im_size, loss_r_feature_layers,
#                    supcon_loss=None):
#     net.eval()
#
#     # image_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
#     # # lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
#     # label_syn = label_syn[c*ipc:(c+1)*ipc]
#     # loss_all = 0
#     # train_iter = iter(train_loader)
#     # for step in range(len(train_iter)):
#
#     #     image_syn, label_syn = next(train_iter)
#
#     #     # image_syn.requires_grad = True
#     #     # print(image_syn)
#
#     # apply augmentation for each iteration
#     inputs_jit = image_syn
#
#     # apply random jitter offsets
#     lim_0 = im_size[0] // 10
#     lim_1 = im_size[1] // 10
#     off1 = random.randint(-lim_0, lim_0)
#     off2 = random.randint(-lim_1, lim_1)
#     inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))
#
#     # Flipping
#     flip = random.random() > 0.5
#     if flip:
#         inputs_jit = torch.flip(inputs_jit, dims=(3,))
#
#     feature_syn, output_syn = net(inputs_jit)
#     loss_ce = criterion(output_syn, label_syn)
#
#     # # supervised contrastive loss
#     # all_features = nn.functional.normalize(feature_syn, dim=1)
#     # all_features = all_features.unsqueeze(1)
#     # align_cls_loss = supcon_loss(
#     #         features=all_features,
#     #         labels=label_syn,
#     #         temperature=0.07, mask=None)
#     # sys.exit()
#
#     # l2 and total variation loss
#     loss_l2 = l2_norm(inputs_jit)
#     loss_tv = total_variation(inputs_jit)
#
#     # BN loss
#     first_bn_multiplier = 10.
#     rescale = [first_bn_multiplier] + [
#         10 * i for i in range(1, len(loss_r_feature_layers))
#     ]
#     # rescale = [10*i for i in range(len(loss_r_feature_layers))]
#     # rescale = [1., 10., 10., 50., 100.]
#     # print(rescale)
#     # sys.exit()
#     loss_r_feature = sum([
#         mod.r_feature * rescale[idx]
#         for (idx, mod) in enumerate(loss_r_feature_layers)
#     ])
#     # print(f'l2 loss {loss_l2}')
#     # print(f'tv loss {loss_tv}')
#     # print(f'BN loss {loss_r_feature}')
#
#     l2_reg, tv_reg, bn_reg = 3e-8, 2.5e-5, 0.1
#     loss = loss_ce + l2_reg * loss_l2 + tv_reg * loss_tv + bn_reg * loss_r_feature
#     # print(loss_r_feature)
#
#     optimizer_img.zero_grad()
#     loss.backward()
#     # loss_all += loss.item()
#     optimizer_img.step()
#
#     return loss.item()
#
#
# def distribution_matching_bn(image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc,
#                              image_server=None, net_name=None):
#     lambda_sim = 0.1
#
#     # get net
#     net = get_network(net_name, channel, num_classes, im_size).to(device)  # get a random model
#     net.train()
#     for param in list(net.parameters()):
#         param.requires_grad = False
#
#     embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel
#
#     loss_avg = 0
#
#     ''' update synthetic data '''
#     loss = torch.tensor(0.0).to(device)
#     images_real_all = []
#     images_syn_all = []
#     batch_real = image_real[0].size(0)
#     if image_server is not None:
#         images_server_all = []
#     for c in range(num_classes):
#         img_real = image_real[c]
#         img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, channel, im_size[0], im_size[1]))
#
#         seed = int(time.time() * 1000) % 100000
#         dsa_param = ParamDiffAug()
#         img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#         img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#
#         images_real_all.append(img_real)
#         images_syn_all.append(img_syn)
#
#         if image_server is not None:
#             img_server = image_server[c]
#             img_server = DiffAugment(img_server, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#             images_server_all.append()
#
#     images_real_all = torch.cat(images_real_all, dim=0)
#     images_syn_all = torch.cat(images_syn_all, dim=0)
#
#     output_real = embed(images_real_all).detach()
#     output_syn = embed(images_syn_all)
#
#     loss += torch.sum((torch.mean(output_real.reshape(num_classes, batch_real, -1), dim=1) - torch.mean(
#         output_syn.reshape(num_classes, ipc, -1), dim=1)) ** 2)
#
#     if image_server is not None:
#         images_server_all = torch.cat(images_server_all, dim=0)
#         output_server = embed(images_server_all).detach()
#         server_client_loss = torch.sum((torch.mean(output_server, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
#         loss += lambda_sim * server_client_loss
#
#     # # l2 and total variation loss
#     # loss += lambda_sim * l2_norm(img_syn)
#     # loss += lambda_sim * total_variation(img_syn)
#
#     optimizer_img.zero_grad()
#     loss.backward()
#     optimizer_img.step()
#
#     if image_server is not None:
#         return loss.item(), image_syn, server_client_loss.item()
#     else:
#         return loss.item(), image_syn
#
#
# def distribution_matching_DP(image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, minibatch_loader,
#                              microbatch_loader, net=None):
#     ''' update synthetic data '''
#     # loss = torch.tensor(0.0).to(device)
#     minibatch_loaders = []
#     for c in range(num_classes):
#         img_real = image_real[c]
#         label_syns_tmp = torch.tensor(np.array(np.ones(len(img_real)) * c), dtype=torch.long, requires_grad=False,
#                                       device=device).view(-1)
#         minibatch_loaders.append(minibatch_loader(TensorDataset(img_real, label_syns_tmp)))
#
#     train_iters = [iter(minibatch_loaders[i]) for i in range(len(minibatch_loaders))]
#     loss_all = 0
#     optimizer_img.zero_grad()
#     for c in range(num_classes):
#         img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, channel, im_size[0], im_size[1]))
#         for step in range(len(train_iters[c])):
#
#             # default we use ConvNet
#             if net == None:
#                 net = get_network('ConvNet', channel, num_classes, im_size).to(device)  # get a random model
#                 net.train()
#                 # for param in list(net.parameters()):
#                 #     param.requires_grad = False
#             else:
#                 net.train()
#                 # for param in list(net.parameters()):
#                 #     param.requires_grad = False
#
#             embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel
#             # get minibatch images for microbatch
#             X_minibatch, y_minibatch = next(train_iters[c])
#             for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
#                 # seed = int(time.time() * 1000) % 100000
#                 # dsa_param = ParamDiffAug()
#                 # X_microbatch = DiffAugment(X_microbatch, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#                 # img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#
#                 output_real = embed(X_microbatch).detach()
#                 output_syn = embed(img_syn)
#
#                 optimizer_img.zero_microbatch_grad()
#                 loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
#                 loss.backward()
#                 loss_all += loss.item()
#                 # total_norm = 0.
#                 # for group in optimizer_img.param_groups:
#                 #     for param in group['params']:
#                 #         if param.requires_grad:
#                 #             total_norm += param.grad.data.norm(2).item() ** 2.
#                 # total_norm = total_norm ** .5
#                 # print(total_norm)
#                 optimizer_img.microbatch_step()
#     optimizer_img.step()
#     # img_real = image_real[c]
#     # label_syns_tmp = torch.tensor(np.array(np.ones(len(img_real))*c), dtype=torch.long, requires_grad=False, device=device).view(-1)
#     # c_train_set = TensorDataset(img_real, label_syns_tmp)
#
#     # seed = int(time.time() * 1000) % 100000
#     # dsa_param = ParamDiffAug()
#     # img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#     # img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
#
#     # for X_minibatch, y_minibatch in minibatch_loader(c_train_set):
#     #     optimizer_img.zero_grad()
#     #     for X_microbatch, y_microbatch in microbatch_loader(TensorDataset(X_minibatch, y_minibatch)):
#
#     #         output_real = embed(X_microbatch).detach()
#     #         output_syn = embed(img_syn)
#
#     #         optimizer_img.zero_microbatch_grad()
#     #         loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
#     #         loss.backward()
#     #         optimizer_img.microbatch_step()
#
#     #     optimizer_img.step()
#
#     return loss.item(), image_syn





# def gradient_matching(args, net, criterion, gw_reals, image_syn, optimizer_img, channel, num_classes, im_size, ipc):
#     lambda_sim = 0.1
#
#     ''' get model info '''
#     net_parameters = list(net.parameters())
#
#     ''' update synthetic data '''
#     loss = torch.tensor(0.0).to(device)
#     for c in range(num_classes):
#         img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, channel, im_size[0], im_size[1]))
#         lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
#
#         output_syn = net(img_syn)
#         loss_syn = criterion(output_syn, lab_syn)
#         gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
#
#         loss += match_loss(gw_syn, gw_reals[c], args)
#
#     # l2 and total variation loss
#     l2_reg, tv_reg = 3e-8, 2.5e-5
#     loss += l2_reg * l2_norm(img_syn)
#     loss += tv_reg * total_variation(img_syn)
#
#     # # BN inversion
#     # if 'BN' in args.model:
#     #     loss_r_feature_layers = []
#     #     for module in net.modules():
#     #         if isinstance(module, torch.nn.BatchNorm2d):
#     #             loss_r_feature_layers.append(BNFeatureHook(module))
#     #     net(image_syn)
#     #     loss_r_feature = sum([
#     #         mod.r_feature
#     #         for (idx, mod) in enumerate(loss_r_feature_layers)
#     #     ])
#
#     #     loss += lambda_sim * loss_r_feature
#
#     optimizer_img.zero_grad()
#     loss.backward()
#     optimizer_img.step()
#
#     return loss.item(), image_syn
#
#
# # def gradient_matching(args, net, criterion, gw_reals, image_syn, optimizer_img, channel, num_classes, im_size, ipc):
#
# #     lambda_sim = 0.1
#
# #     ''' get model info '''
# #     net_parameters = list(net.parameters())
#
# #     ''' update synthetic data '''
# #     loss = torch.tensor(0.0).to(device)
# #     for c in range(num_classes):
# #         img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
# #         lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
#
# #         output_syn = net(img_syn)
# #         loss_syn = criterion(output_syn, lab_syn)
# #         gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
#
# #         loss += match_loss(gw_syn, gw_reals[c], args)
#
# #     # l2 and total variation loss
# #     l2_reg, tv_reg = 3e-8, 2.5e-5
# #     loss += l2_reg * l2_norm(img_syn)
# #     loss += tv_reg * total_variation(img_syn)
#
# #     # # BN inversion
# #     # if 'BN' in args.model:
# #     #     loss_r_feature_layers = []
# #     #     for module in net.modules():
# #     #         if isinstance(module, torch.nn.BatchNorm2d):
# #     #             loss_r_feature_layers.append(BNFeatureHook(module))
# #     #     net(image_syn)
# #     #     loss_r_feature = sum([
# #     #         mod.r_feature
# #     #         for (idx, mod) in enumerate(loss_r_feature_layers)
# #     #     ])
#
# #     #     loss += lambda_sim * loss_r_feature
#
# #     optimizer_img.zero_grad()
# #     loss.backward()
# #     optimizer_img.step()
#
# #     return loss.item(), image_syn
#
#
# def gradient_matching_all(args, net, criterion, gw_reals, image_syn, lab_syn, optimizer_img, channel, num_classes,
#                           im_size, ipc):
#     lambda_sim = 0.1
#
#     ''' get model info '''
#     net_parameters = list(net.parameters())
#
#     ''' update synthetic data '''
#     loss = torch.tensor(0.0).to(device)
#     img_syn = image_syn.reshape((ipc * num_classes, channel, im_size[0], im_size[1]))
#     # lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
#
#     output_feature, output_syn = net(img_syn)
#     loss_syn = criterion(output_syn, lab_syn)
#     gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
#
#     loss += match_loss(gw_syn, gw_reals, args)
#
#     # # l2 and total variation loss
#     # loss += lambda_sim * l2_norm(img_syn)
#     # loss += lambda_sim * total_variation(img_syn)
#
#     # # BN inversion
#     # if 'BN' in args.model:
#     #     loss_r_feature_layers = []
#     #     for module in net.modules():
#     #         if isinstance(module, torch.nn.BatchNorm2d):
#     #             loss_r_feature_layers.append(BNFeatureHook(module))
#     #     net(image_syn)
#     #     loss_r_feature = sum([
#     #         mod.r_feature
#     #         for (idx, mod) in enumerate(loss_r_feature_layers)
#     #     ])
#
#     #     loss += lambda_sim * loss_r_feature
#
#     optimizer_img.zero_grad()
#     loss.backward()
#     optimizer_img.step()
#
#     return loss.item()
#
#
# # def gradient_inversion(net, criterion, optimizer_img, image_syn, label_syn, num_classes, ipc, channel, im_size, loss_r_feature_layers):
#
# #     lambda_sim = 0.01
#
# #     net.train()
#
#
# #     loss = 0
# #     for c in range(num_classes):
# #         img_syn = image_syn[c*ipc:(c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
# #         # lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
# #         lab_syn = label_syn[c*ipc:(c+1)*ipc]
#
# #         output_syn = net(img_syn)
# #         loss += criterion(output_syn, lab_syn)
#
#
# #         # # l2 and total variation loss
# #         # loss += lambda_sim * l2_norm(img_syn)
# #         # loss += lambda_sim * total_variation(img_syn)
#
# #     optimizer_img.zero_grad()
# #     loss.backward()
# #     optimizer_img.step()
#
# #     return loss.item()
#
#
# def gradient_distribution_matching(args, net, criterion, gw_real, image_reals, image_syn, optimizer_img, channel,
#                                    num_classes, im_size, ipc):
#     lambda_sim = 0.5
#
#     ''' get model info '''
#     net_parameters = list(net.parameters())
#     embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel
#
#     # # for models contains BN
#     # for module in net.modules():
#     #     if 'BatchNorm' in module._get_name():  #BatchNorm
#     #         module.eval() # fix mu and sigma of every BatchNorm layer
#
#     ''' update synthetic data'''
#     loss = torch.tensor(0.0).to(device)
#     for c in range(num_classes):
#
#         # GM
#         img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, channel, im_size[0], im_size[1]))
#         lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
#
#         output_syn = net(img_syn)
#         loss_syn = criterion(output_syn, lab_syn)
#         gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
#
#         loss += match_loss(gw_syn, gw_real, args)
#
#         # DM
#         output_syn = embed(img_syn)
#         output_real = torch.zeros((ipc * 5, output_syn.size(1))).to(args.device)
#         for image_real in image_reals:
#             img_real = image_real[c * ipc * 5:(c + 1) * ipc * 5].reshape((ipc * 5, channel, im_size[0], im_size[1]))
#             output_real += embed(img_real).detach() / len(image_reals)
#
#         loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
#
#     # l2 and total variation loss
#     loss += lambda_sim * l2_norm(img_syn)
#     loss += lambda_sim * total_variation(img_syn)
#
#     optimizer_img.zero_grad()
#     loss.backward()
#     optimizer_img.step()
#
#     return loss.item(), image_syn
#
#
# def gradient_distribution_matching_bn(args, net, criterion, gw_real, image_reals, image_syn, optimizer_img, channel,
#                                       num_classes, im_size, ipc):
#     lambda_sim = 0.5
#
#     ''' get model info '''
#     net_parameters = list(net.parameters())
#     embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel
#
#     # # for models contains BN
#     # for module in net.modules():
#     #     if 'BatchNorm' in module._get_name():  #BatchNorm
#     #         module.eval() # fix mu and sigma of every BatchNorm layer
#
#     ''' update synthetic data'''
#     loss = torch.tensor(0.0).to(device)
#     images_real_all = []
#     images_syn_all = []
#     for c in range(num_classes):
#
#         # GM
#         img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, channel, im_size[0], im_size[1]))
#         lab_syn = torch.ones((ipc,), device=device, dtype=torch.long) * c
#
#         output_syn = net(img_syn)
#         loss_syn = criterion(output_syn, lab_syn)
#         gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
#
#         loss += match_loss(gw_syn, gw_real, args)
#
#         # DM
#         output_syn = embed(img_syn)
#         output_real = torch.zeros((ipc * 5, output_syn.size(1))).to(args.device)
#         for image_real in image_reals:
#             img_real = image_real[c * ipc * 5:(c + 1) * ipc * 5].reshape((ipc * 5, channel, im_size[0], im_size[1]))
#             output_real += embed(img_real).detach() / len(image_reals)
#
#         loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
#
#     # l2 and total variation loss
#     loss += lambda_sim * l2_norm(img_syn)
#     loss += lambda_sim * total_variation(img_syn)
#
#     optimizer_img.zero_grad()
#     loss.backward()
#     optimizer_img.step()
#
#     return loss.item(), image_syn