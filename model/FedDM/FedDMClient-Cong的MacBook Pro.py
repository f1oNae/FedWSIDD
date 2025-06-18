from copy import deepcopy
from tqdm import tqdm
from model.Client import AgentBase
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from utils.trainer_util import get_optim, random_pertube
import torch
import gc

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

    def local_train(self, agent_idx):
        self.turn_on_training()
        # initialize S_k from real examples and initialize optimizer
        for i, c in enumerate(range(self.args.n_classes)):
            self.synthetic_images.data[i * self.ipc: (i + 1) * self.ipc] = self.train_dataset.get_image(c, self.ipc, self.nps).detach().data
        print('Synthetic images initialized with size and buffer size:', self.synthetic_images.size(), torch.cuda.max_memory_allocated() / 1024 ** 2)
        optimizer_image = torch.optim.SGD([self.synthetic_images], lr=self.image_lr, momentum=0.5, weight_decay=0)
        optimizer_image.zero_grad()
        total_loss = 0.0

        # Check current memory usage
        pbar = tqdm(range(self.dc_iterations), postfix={'Max Allocated': torch.cuda.max_memory_allocated() / 1024 ** 2})
        for dc_iteration in pbar:
            # sample_model = sample_random_model(self.global_model, self.rho)
            sample_model = random_pertube(self.local_model, self.rho)
            sample_model.eval()
            total_memory = 0
            # syn_before_train = deepcopy(self.synthetic_images)
            for i, c in enumerate(range(self.args.n_classes)):
                labels = torch.tensor([c] * self.ipc, dtype=torch.long, device=self.device)
                real_image = self.train_dataset.get_image(c, 1, 0)
                real_image = real_image.to(self.device)
                if 'CLAM' in self.args.mil_method:
                    _, _, real_feature, real_logits = clam_runner(self.args,
                                                                  sample_model,
                                                                  real_image.squeeze(0),
                                                                  labels[0].unsqueeze(0),
                                                                  self.mil_loss,
                                                                  return_lgt=True,
                                                                  return_feature=True)
                else:
                    self.logger.error(f'{self.args.mil_method} not implemented')
                    raise NotImplementedError

                # del real_image
                # torch.cuda.empty_cache()

                for syn_i in range(self.ipc):
                    synthetic_image = self.synthetic_images[i * self.ipc: (i + 1) * self.ipc][syn_i]
                    if 'CLAM' in self.args.mil_method:
                        _, _, synthetic_feature, synthetic_logits = clam_runner(self.args,
                                                  sample_model,
                                                  synthetic_image,
                                                  labels[0].unsqueeze(0),
                                                  self.mil_loss,
                                                  return_lgt=True,
                                                  return_feature=True,
                                                  raw_image=True)
                    else:
                        self.logger.error(f'{self.args.mil_method} not implemented')
                        raise NotImplementedError
                    loss_ft = torch.sum((torch.mean(real_feature, dim=0) - torch.mean(synthetic_feature, dim=0)) ** 2)
                    loss_lgt = torch.sum((torch.mean(real_logits, dim=0) - torch.mean(synthetic_logits, dim=0)) ** 2)
                    loss = loss_ft + loss_lgt
                    #
                    optimizer_image.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer_image.step()

                # del real_feature, real_logits, synthetic_feature, synthetic_logits
            max_allocated_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
            pbar.set_postfix({'Max Allocated': max_allocated_memory})

            if dc_iteration % 100 == 0:
                self.logger.info(f'Agent: {agent_idx}, Iter: {dc_iteration}, DM Loss: {loss.item()}')
            total_loss += loss.cpu().item()

        # return S_k
        synthetic_labels = torch.cat([torch.ones(self.ipc) * c for c in range(self.args.n_classes)])
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