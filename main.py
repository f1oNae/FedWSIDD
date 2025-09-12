from __future__ import print_function
import argparse
import os, sys
from datetime import datetime
import numpy as np
import logging
from utils.core_util import Meta
from model.FedBase.FedBaseServer import FedBase
from model.FedHisto.FedHistoServer import FedHisto
from model.FedAvg.FedAvgServer import FedAvg
from model.FedProx.FedProxServer import FedProx
from model.FedDyn.FedDynServer import FedDyn
from model.FedMOON.FedMOONServer import FedMOON
# from model.FedDM.FedDMServer import FedDM
from model.FedGen.FedGenServer import FedGen
from model.FedAF.FedAFServer import FedAF
from model.FedNoVa.FedNovaServer import FedNoVa
from model.FedScaffold.FedScaffoldServer import FedScaffold
from model.FedDESA.FedDESARunner import FedDESA
from model.FedProto.FedProtoServer import FedProto
from model.FedPrompt.FedPromptServer import FedPrompt
from model.FedWSIDD.FedWSIDDRunner import FedWSIDD
from model.FedHE.FedHEServer import FedHE
from model.SGPT.SGPTServer import SGPT
from model.FedImpro.FedImproServer import FedImpro
from model.FedMut.FedMutServer import FedMut
from model.FedSOL.FedSOLServer import FedSOL
from model.FedDDG.FedDDGRunner import FedDDG
import torch


# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for Fed + DD + WSI Training')
parser.add_argument('--repeat', type=int, default=5,
                    help='number of repeated experiments')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--ensemble_epochs', type=int, default=50)
parser.add_argument('--kd_iters', type=int, default=100)
parser.add_argument('--pretrained_dir', type=str, default='/scratch/iq24/cc0395/FedDDHist/model/FExtractor/vit_s_16_imagenet.npz')
parser.add_argument('--image_batch_size', type=int, default=128)
parser.add_argument('--pretrain_kd', action='store_true', default=False, help='pretrain with kd')
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--global_epochs', type=int, default=200,
                    help='maximum number of epochs to train globaly(default: 200)')
parser.add_argument('--global_epochs_dm', type=int, default=200,
                    help='maximum number of epochs to train globaly(default: 200)')
parser.add_argument('--local_epochs', type=int, default=200,
                    help='maximum number of epochs to train localy(default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--heter_model', action='store_true', default=False, help='heter model')
parser.add_argument('--ld_proto', type=float, default=0.1, help='proto loss weight')
parser.add_argument("--ensemble_lr", type=float, default=1e-4, help="Ensemble learning rate.")
parser.add_argument('--generative_alpha', type=int, default=10)
parser.add_argument('--generative_beta', type=int, default=10)
parser.add_argument('--ensemble_beta', type=int, default=0)
parser.add_argument('--ensemble_eta', type=int, default=1)
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--n_classes', type=int, default=2,
                    help='number of classes')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--best_run', type=int, default=0)
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce', 'mag'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['tiny', 'ultra_small', 'small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--ft_model', type=str, default='ResNet50',
                    choices=['ResNet50', 'ResNet50_prompt', 'ResNet50_deep_ft_prompt',
                             'ResNet50_simclr', 'ResNet50_simclr_prompt',
                             'ViT_S_16', 'ViT_S_16_prompt',
                             'ViT_S_16_dino', 'ViT_S_16_dino_prompt', 'ViT_S_16_dino_deep_ft_prompt',
                             'ViT_T_16', 'ViT_T_16_prompt', 'ViT_S_16_deep_ft_prompt', 'hipt'],)
parser.add_argument('--mil_method', type=str, default='CLAM_SB', help='mil method')
parser.add_argument('--feature_type', type=str, default='R50_features', help='feature type')
parser.add_argument('--fed_method', type=str, default='fed_avg',choices=['fed_base',
                                                                         'fed_histo',
                                                                         'fed_avg',
                                                                         'fed_prox',
                                                                         'fed_dyn',
                                                                         'scaffold',
                                                                         'moon',
                                                                         'fed_gen',
                                                                         'fed_nova',
                                                                         'fed_dm',
                                                                         'fed_af',
                                                                         'fed_desa',
                                                                         'fed_proto',
                                                                         'fed_he',
                                                                         'sgpt',
                                                                         'fed_impro',
                                                                         'fed_mut',
                                                                         'fed_sol',
                                                                         'fed_prompt',
                                                                         'fed_ddg'], help='fed method')
parser.add_argument('--fed_split', type=str, default='FeatureSynthesisLabel', help='fed split')
parser.add_argument('--fed_split_std_mode', type=str, default='update', help='fed split std mode')
parser.add_argument('--fed_split_noise_std', type=float, default=0.1, help='fed split noise std')
parser.add_argument('--fed_split_client_DP_degree', type=float, default=0.001, help='fed split client DP degree')
parser.add_argument('--mu', type=float, default=0.01, help='proximal term for fedprox')
parser.add_argument('--ensemble_alpha', type=float, default=0.2, help='The hypter parameter for the FedGKD')
parser.add_argument('--radius', type=float, default=4.0)
parser.add_argument('--min_radius', type=float, default=0.1)
parser.add_argument('--mut_acc_rate', type=float, default=0.3)
parser.add_argument('--mut_bound', type=int, default=50)
parser.add_argument('--alpha_coef', type=float, default=1e-2, help='alpha coefficient for feddyn')
parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss for moon')
parser.add_argument('--pool_option', type=str, default='FIFO', choices=['FIFO', 'LIFO'], help='pooling option for moon')
parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
parser.add_argument('--contrast_mu', type=float, default=1, help='the mu parameter for fedprox or moon')
parser.add_argument('--lambda_local', type=float, default=0.01, help='fed af local loss weight')
parser.add_argument('--lambda_global', type=float, default=0.1, help='fed af global loss weight')
parser.add_argument('--task', type=str)
parser.add_argument('--accumulate_grad_batches', type=int, default=1,)
parser.add_argument('--use_h5', action='store_true', default=False, help='use h5 files')
parser.add_argument('--syn_size', type=int,default=256, help='size of synthetic patch')
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False,
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
### DFTD specific options
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
### FRMIL specific options
parser.add_argument('--shift_feature', action='store_true', default=False, help='shift feature')
parser.add_argument('--drop_data', action='store_true', default=False, help='drop data')
parser.add_argument('--balanced_sample', action='store_true', default=False, help='balanced bag')
parser.add_argument('--n_heads', type=int, default=1, help='number of heads')
parser.add_argument('--mag', type=float, default=1.0, help='magnitude')
### DFP
parser.add_argument('--dfp', action='store_true', default=False, help='use dfp')
parser.add_argument('--prompt_initialisation', type=str, default='gaussian', help='prompt init')
parser.add_argument('--prompt_aggregation', type=str, default='multiply', choices=['multiply', 'add', 'prepend'], help='prompt aggregation method')
parser.add_argument('--number_prompts', type=int, default=1)
parser.add_argument("--key_prompt", type=int, default=0, help='cluster numbers')
parser.add_argument('--prompt_epoch', type=int, default=10)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--prompt_reg', type=float, default=1e-5)
parser.add_argument('--adaptive_prompt_lr', action='store_true', default=False, help='adaptive prompt lr')
parser.add_argument('--renew_train', action='store_true', default=False, help='adaptive prompt lr')
parser.add_argument('--share_blocks', nargs='+', type=int, default=[], help="shared transformer set 6 ")
parser.add_argument('--share_blocks_g', nargs='+', type=int,  default=[], help="shared transformer set 6 ")
parser.add_argument('--use_stain', action='store_true', default=False, help='use stain')
### HIPT
parser.add_argument('--top_k', type=int, default=-1, help='top k')
parser.add_argument('--pretrain_4k',    type=str, default='None', help='Whether to initialize the 4K Transformer in HIPT', choices=['None', 'vit4k_xs_dino'])
parser.add_argument('--pretrain_WSI',    type=str, default='None')
parser.add_argument('--freeze_4k',      action='store_true', default=False, help='Whether to freeze the 4K Transformer in HIPT')
parser.add_argument('--freeze_WSI',     action='store_true', default=False, help='Whether to freeze the WSI Transformer in HIPT')
### FedDM
parser.add_argument('--ipc', type=int, default=10, help='Number of syn slide per class')
parser.add_argument('--nps', type=int, default=10, help='Number of syn patchs per slide')
parser.add_argument('--image_lr', type=float, default=0.1, help='Learning rate for synthetic images')
parser.add_argument('--image_opt', type=str, default='adam', help='image_opt')
parser.add_argument('--init_real', action='store_true', default=False, help='init syn image with real images =')
parser.add_argument('--dc_iterations', type=int, default=1000, help='Number of iterations for synthetic images')
parser.add_argument('--rho', type=float, default=5.0, help='Perturbation strength for model perturbation')
### FedWSI
parser.add_argument('--sn',action='store_true', default=False, help='aug syn image with stain normalization')


parser.add_argument('--debug', action='store_true', default=False, help='debugging tool')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO,
                    filemode="w",
                    format="%(name)s: %(asctime)s | %(filename)s:%(lineno)s |  %(message)s",
                    filename=f"logs/{args.task}/{args.feature_type}/{args.fed_method}_{args.feature_type}_{args.mil_method}_{args.exp_code}_logs.txt")
logger = logging.getLogger(__name__)

args.results_dir = os.path.join(args.results_dir, f"{args.task}/{args.fed_method}_{args.feature_type}_{args.mil_method}_{args.exp_code}")
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)
print('Results will be saved in: {}'.format(args.results_dir))
logger.info('Results will be saved in: {}'.format(args.results_dir))
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    if args.debug:
        args.repeat = 1
        args.global_epochs = 1
        args.local_epochs = 1
        args.dc_iterations = 1
        args.global_epochs_dm = 1
        args.kd_iters = 1
        args.ipc = 10
        args.nps = 100
    overall_avg_acc, overall_acc_per_agent = [], {}
    logger.info(f'Performing experiments: {args.exp_code} {args.fed_method} {args.mil_method} {args.ft_model}')
    for rep in range(args.repeat):
        args.rep = rep
        logger.info(f'======================== Run {rep} Starts========================')
        seed = int(datetime.now().timestamp())
        seed_torch(seed)
        if args.fed_method == 'fed_base':
            runner = FedBase(args, logger=logger)
        elif args.fed_method == 'fed_histo':
            runner = FedHisto(args, logger=logger)
        elif args.fed_method == 'fed_avg':
            runner = FedAvg(args, logger=logger)
        elif args.fed_method == 'fed_prox':
            runner = FedProx(args, logger=logger)
        elif args.fed_method == 'fed_dyn':
            runner = FedDyn(args, logger=logger)
        elif args.fed_method == 'moon':
            runner = FedMOON(args, logger=logger)
        elif args.fed_method == 'fedgen':
            runner = FedGen(args, logger=logger)
        elif args.fed_method == 'fed_nova':
            runner = FedNoVa(args, logger=logger)
        # elif args.fed_method == 'fed_dm':
        #     runner = FedDM(args, logger=logger)
        elif args.fed_method == 'fed_af':
            runner = FedAF(args, logger=logger)
        elif args.fed_method == 'scaffold':
            runner = FedScaffold(args, logger=logger)
        elif args.fed_method == 'fed_desa':
            runner = FedDESA(args, logger=logger)
        elif args.fed_method == 'fed_proto':
            runner = FedProto(args, logger=logger)
        elif args.fed_method == 'fed_he':
            runner = FedHE(args, logger=logger)
        elif args.fed_method == 'fed_wsidd':
            runner = FedWSIDD(args, logger=logger)
        elif args.fed_method == 'sgpt':
            runner = SGPT(args, logger=logger)
        elif args.fed_method == 'fed_impro':
            runner = FedImpro(args, logger=logger)
        elif args.fed_method == 'fed_mut':
            runner = FedMut(args, logger=logger)
        elif args.fed_method == 'fed_sol':
            runner = FedSOL(args, logger=logger)
        elif args.fed_method == 'fed_prompt':
            runner = FedPrompt(args, logger=logger)
        elif args.fed_method == 'fed_ddg':
            runner = FedDDG(args, logger=logger)
        else:
            raise NotImplementedError
        best_accuracy, train_acc_wt, best_accuracy_per_agent = runner.run(rep)
        overall_avg_acc.append(best_accuracy)
        if len(overall_acc_per_agent)==0:
            for i in range(len(best_accuracy_per_agent)):
                overall_acc_per_agent[i] = [best_accuracy_per_agent[i]]
        else:
            for i in range(len(best_accuracy_per_agent)):
                overall_acc_per_agent[i].append(best_accuracy_per_agent[i])
        logger.info(f'======================== Run {rep} Ends========================')
    logger.info(f'Accuracies: avg: {np.mean(overall_avg_acc):.4f} std: {np.std(overall_avg_acc):.4f} best: {np.max(overall_avg_acc):.4f}')
    logger.info(f'Weighed avg: {np.mean(train_acc_wt):.4f} std: {np.std(train_acc_wt):.4f} best: {np.max(train_acc_wt):.4f}')
    logger.info('Accuracies per agent: ')
    for ag_idx in overall_acc_per_agent:
        ag_acc = overall_acc_per_agent[ag_idx]
        logger.info(f'Agent {ag_idx}: avg: {np.mean(ag_acc):.4f} std: {np.std(ag_acc):.4f} best: {np.max(ag_acc):.4f}')
    logger.info(f'Best run: {np.argmax(overall_avg_acc)}')


