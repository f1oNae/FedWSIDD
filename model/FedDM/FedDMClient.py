from copy import deepcopy
from tqdm import tqdm
from model.Client import AgentBase
from model.condensation import distribution_matching, distribution_matching_woMIL, distribution_matching_woMIL_new
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim, random_pertube
import torch
import numpy as np
import gc

def get_images(images_all, indices_class, c, n): # get random n images from class c
    if n < len(indices_class[c]):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
    else:
        idx_shuffle_0 = np.random.permutation(indices_class[c])
        idx_shuffle_1 = np.random.permutation(indices_class[c])[:n-len(indices_class[c])]
        idx_shuffle = np.concatenate([idx_shuffle_0, idx_shuffle_1], axis=0)
    img_pth = images_all[idx_shuffle[0]]
    X = torch.load(img_pth).unsqueeze(0)
    return X

class FedDMAgent(AgentBase):
    def __init__(self, args, global_model, logger):
        super().__init__(args, global_model, logger)
        self.init_syn_data()

    def init_syn_data(self):
        self.feature_size = 1024 if 'ResNet50' in self.args.ft_model else 384
        self.ipc = self.args.ipc
        self.nps = self.args.nps
        self.rho = self.args.rho
        self.image_lr = self.args.image_lr
        self.dc_iterations = self.args.dc_iterations
        self.synthetic_images = torch.randn(
            size=(
                self.args.n_classes * self.ipc,
                self.nps,
                3, self.args.syn_size, self.args.syn_size
            ),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        # print('Size of synthetic images:', self.synthetic_images.size())

    def local_train(self, agent_idx):
        self.turn_on_training()
        # organize the real dataset
        indices_class = [[] for c in range(self.args.n_classes)]
        # images_all = [torch.unsqueeze(self.train_dataset[client_idx][i][0], dim=0) for i in
        #               range(len(self.train_dataset[client_idx]))]
        images_all = [self.train_dataset.__getitem__(i, path=True)[2] for i in
                      range(len(self.train_dataset))]
        labels_all = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        for idx, lab in enumerate(labels_all):
            lab_item = int(lab.item())
            indices_class[lab_item].append(idx)

        # initialize S_k from real examples and initialize optimizer
        for i, c in enumerate(range(self.args.n_classes)):
            self.synthetic_images.data[i * self.ipc: (i + 1) * self.ipc] = self.train_dataset.get_image(c, self.ipc, self.nps).detach().data
        print('Synthetic images initialized with size and buffer size:', self.synthetic_images.size(), torch.cuda.max_memory_allocated() / 1024 ** 2)
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)
        # optimizer_image = torch.optim.Adam([self.synthetic_images, ], lr=1.0, )
        optimizer_image.zero_grad()
        total_loss = 0.0
        image_batch = 1
        dm_loss_avg = 0
        # Check current memory usage
        pbar = tqdm(range(self.dc_iterations), postfix={'Max Allocated': torch.cuda.max_memory_allocated() / 1024 ** 2})
        for dc_iteration in pbar:
            # get real images for each class
            image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(self.args.n_classes)]
            # print([image_real[i].size(0) for i in range(len(image_real))])
            loss, self.synthetic_images = distribution_matching_woMIL(image_real,
                                                                       self.synthetic_images,
                                                                       optimizer_image,
                                                                       3,
                                                                       self.args.n_classes,
                                                                       self.args.syn_size,
                                                                       self.args.ipc,
                                                                       self.args.nps,
                                                                       args=self.args,
                                                                       loss_fn=self.mil_loss, )
            # # sample_model = sample_random_model(self.global_model, self.rho)
            # sample_model = random_pertube(self.local_model, self.rho)
            # sample_model.eval()
            # total_memory = 0
            # # syn_before_train = deepcopy(self.synthetic_images)
            # for i, c in enumerate(range(self.args.n_classes)):
            #     labels = torch.tensor([c] * self.ipc, dtype=torch.long, device=self.device)
            #     real_image = self.train_dataset.get_image(c, 1, 0)
            #     real_image = real_image.to(self.device)
            #     if 'CLAM' in self.args.mil_method:
            #         _, _, real_feature, real_logits = clam_runner(self.args,
            #                                                       sample_model,
            #                                                       real_image.squeeze(0),
            #                                                       labels[0].unsqueeze(0),
            #                                                       self.mil_loss,
            #                                                       return_lgt=True,
            #                                                       return_feature=True)
            #     else:
            #         self.logger.error(f'{self.args.mil_method} not implemented')
            #         raise NotImplementedError
            #
            #     # del real_image
            #     # torch.cuda.empty_cache()
            #
            #     for syn_i in range(self.ipc):
            #         synthetic_image = self.synthetic_images[i * self.ipc: (i + 1) * self.ipc][syn_i]
            #         if 'CLAM' in self.args.mil_method:
            #             _, _, synthetic_feature, synthetic_logits = clam_runner(self.args,
            #                                       sample_model,
            #                                       synthetic_image,
            #                                       labels[0].unsqueeze(0),
            #                                       self.mil_loss,
            #                                       return_lgt=True,
            #                                       return_feature=True,
            #                                       raw_image=True)
            #         else:
            #             self.logger.error(f'{self.args.mil_method} not implemented')
            #             raise NotImplementedError
            #         loss_ft = torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
            #         loss_lgt = torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0)) ** 2)
            #         loss = loss_ft + loss_lgt
            #         #
            #         optimizer_image.zero_grad()
            #         loss.backward(retain_graph=True)
            #         optimizer_image.step()

                # del real_feature, real_logits, synthetic_feature, synthetic_logits
            max_allocated_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            pbar.set_postfix({'Max Allocated': max_allocated_memory})

            # report_cuda_memory()
            # del sample_model
            # torch.cuda.empty_cache()

            # check if synthetic images are updated
            # if torch.allclose(syn_before_train, self.synthetic_images):
            #     self.logger.info(f'Agent: {agent_idx}, Iter: {dc_iteration}, Synthetic images are not updated')
            #     break
            # else:
            #     self.logger.info(f'Agent: {agent_idx}, Iter: {dc_iteration}, Synthetic images are updated')
            if dc_iteration % 100 == 0:
                self.logger.info(f'Agent: {agent_idx}, Iter: {dc_iteration}, DM Loss: {loss}')
            total_loss += loss

        # return S_k
        synthetic_labels = torch.cat([torch.ones(self.ipc) * c for c in range(self.args.n_classes)])
        return deepcopy(self.synthetic_images.detach()), synthetic_labels.long(), total_loss / self.dc_iterations

    def local_train_new(self, agent_idx):
        self.turn_on_training()
        # organize the real dataset
        indices_class = [[] for c in range(self.args.n_classes)]
        # images_all = [torch.unsqueeze(self.train_dataset[client_idx][i][0], dim=0) for i in
        #               range(len(self.train_dataset[client_idx]))]
        images_all = [self.train_dataset.__getitem__(i, path=True)[2] for i in
                      range(len(self.train_dataset))]
        labels_all = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        for idx, lab in enumerate(labels_all):
            lab_item = int(lab.item())
            indices_class[lab_item].append(idx)

        self.synthetic_images = torch.randn(
            size=(
                len(labels_all),
                self.nps,
                3, self.args.syn_size, self.args.syn_size
            ),
            dtype=torch.float,
            requires_grad=True,
            device=self.device,
        )
        # initialize S_k from real examples and initialize optimizer
        if self.args.init_real:
            for i in range(len(labels_all)):
                c = labels_all[i].item()
                self.synthetic_images.data[i] =  self.train_dataset.get_image(c, 1, self.nps).detach().data
        print('Synthetic images initialized with size and buffer size:', self.synthetic_images.size(),
              torch.cuda.max_memory_allocated() / 1024 ** 2)
        if self.args.image_opt == 'adam':
            optimizer_image = torch.optim.Adam([self.synthetic_images, ], lr=self.args.image_lr, )
        elif self.args.image_opt == 'sgd':
            optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.args.image_lr, momentum=0.5, weight_decay=0)
        else:
            raise NotImplementedError
        optimizer_image.zero_grad()
        total_loss = 0.0
        image_batch = 1

        pbar = tqdm(range(self.dc_iterations), postfix={'Max Allocated': torch.cuda.max_memory_allocated() / 1024 ** 2})
        for dc_iteration in pbar:
            # get real images for each class
            image_real = [get_images(images_all, indices_class, c, image_batch) for c in range(self.args.n_classes)]
            # print([image_real[i].size(0) for i in range(len(image_real))])
            loss, self.synthetic_images = distribution_matching_woMIL_new(image_real,
                                                                      self.synthetic_images,
                                                                      optimizer_image,
                                                                      3,
                                                                      self.args.n_classes,
                                                                      self.args.syn_size,
                                                                      self.args.ipc,
                                                                      self.args.nps,
                                                                      args=self.args,
                                                                      loss_fn=self.mil_loss,
                                                                      indices_class=indices_class)
            max_allocated_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            pbar.set_postfix({'Max Allocated': max_allocated_memory})
            if dc_iteration % 100 == 0:
                self.logger.info(f'Agent: {agent_idx}, Iter: {dc_iteration}, DM Loss: {loss}')
            total_loss += loss

        # return S_k
        synthetic_labels = torch.cat([torch.ones(len(indices_class[c])) * c for c in range(self.args.n_classes)])
        return deepcopy(self.synthetic_images.detach()), synthetic_labels.long(), total_loss / self.dc_iterations
def report_cuda_memory():
    total_memory = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            if obj.is_cuda:
                # Calculate memory of the tensor in MBzs
                tensor_memory = obj.numel() * obj.element_size() / 1024 ** 2
                total_memory += tensor_memory
                # get obj name

                print(f"{obj} ------------->  {tensor_memory:.2f} MB")

    print(f"\nTotal CUDA memory used by tensors: {total_memory:.2f} MB")