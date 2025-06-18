from copy import deepcopy
from model.Client import AgentBase
import torch.nn.functional as F
from utils.trainer_util import get_optim
import torch.nn as nn
import torch


class FedSOLAgent(AgentBase):
    def __init__(self, args, global_model, global_opt, rho, logger):
        super().__init__(args, global_model, logger)
        self.optimizer = get_optim(self.args, self.local_model)
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")

    def download_global(self, server_model, server_optimizer, rho):
        """Load model & Optimizer"""
        self.local_model.load_state_dict(server_model.state_dict())
        self.optimizer.load_state_dict(server_optimizer.state_dict())
        self._keep_global(server_model)
        self.sam_optimizer = self._get_sam_optimizer(self.optimizer, rho)

    def _keep_global(self, model_dg):
        """Keep distributed global model's weight"""
        self.dg_model = deepcopy(model_dg)
        self.dg_model.to(self.device)

        for params in self.dg_model.parameters():
            params.requires_grad = False

    def reset(self):
        """Clean existing setups"""
        self.datasize = None
        self.optimizer = get_optim(self.args, self.local_model)

    def _get_sam_optimizer(self, base_optimizer, rho):
        optim_params = base_optimizer.state_dict()
        lr = optim_params["param_groups"][0]["lr"]
        # momentum = optim_params["param_groups"][0]["momentum"]
        momentum = self.args.reg
        weight_decay = optim_params["param_groups"][0]["weight_decay"]
        sam_optimizer = ExpSAM(
            self.local_model.parameters(),
            self.dg_model.parameters(),
            base_optimizer=torch.optim.Adam,
            rho=rho,
            adaptive=False,
            lr=lr,
            # momentum=momentum,
            weight_decay=weight_decay,
        )

        return sam_optimizer

    def local_train(self, agent_idx):
        self.turn_on_training()
        # optimizer = get_optim(self.args, self.local_model)
        epoch_loss = 0.
        local_size = len(self.train_loader.dataset)
        for iter in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                _, error, y_prob, logits = self.mil_run(self.local_model,
                                                   images,
                                                   labels,
                                                   self.mil_loss,
                                                   return_lgt=True)
                _, _, _, dg_logits = self.mil_run(self.dg_model,
                                                           images,
                                                           labels,
                                                           self.mil_loss,
                                                           return_lgt=True)
                with torch.no_grad():
                    dg_probs = torch.softmax(dg_logits / 3, dim=1)
                pred_probs = F.log_softmax(logits / 3, dim=1)

                loss = self.KLDiv(
                    pred_probs, dg_probs
                )  # use this loss for any training statistics
                loss.backward()
                self.sam_optimizer.first_step(zero_grad=True)

                loss_task, error, y_prob = self.mil_run(self.local_model,
                                                   images,
                                                   labels,
                                                   self.mil_loss)
                loss_task.backward()
                self.sam_optimizer.second_step(zero_grad=True)
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                if batch_idx % 20 == 0:
                    print(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                    self.logger.info(f'Agent: {agent_idx}, Iter: {iter}, Batch: {batch_idx}, Loss: {loss.item()}')
                batch_loss += loss.item()
            batch_loss /= len(self.train_loader)
            batch_error /= len(self.train_loader)
            epoch_loss += batch_loss
        return self.local_model.state_dict(), epoch_loss / self.args.local_epochs, local_size

class ExpSAM(torch.optim.Optimizer):
    def __init__(
        self, params, ref_params, base_optimizer, rho=0.05, adaptive=False, **kwargs
    ):
        # assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(ExpSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

        self.ref_param_groups = []
        ref_param_groups = list(ref_params)

        if not isinstance(ref_param_groups[0], dict):
            ref_param_groups = [{"params": ref_param_groups}]

        for ref_param_group in ref_param_groups:
            self.add_ref_param_group(ref_param_group)

        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group, ref_group in zip(self.param_groups, self.ref_param_groups):
            scale = group["rho"] / (grad_norm + 1e-12)

            for p, ref_p in zip(group["params"], ref_group["params"]):
                if p.grad is None:
                    try:
                        self.state[p]["old_p"] = p.data.clone()
                    except:
                        pass

                    continue

                # avg_mag = torch.abs(p - ref_p).mean()

                self.state[p]["old_p"] = p.data.clone()
                e_w = F.normalize((p - ref_p).abs(), 2, dim=0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    (1.0 * p.grad).norm(p=2).to(shared_device)
                    for group, ref_group in zip(
                        self.param_groups, self.ref_param_groups
                    )
                    for p, ref_p in zip(group["params"], ref_group["params"])
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

    def add_ref_param_group(self, param_group):
        params = param_group["params"]

        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        params = param_group["params"]

        param_set = set()
        for group in self.ref_param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.ref_param_groups.append(param_group)
