import torch
import torch.nn as nn
from model.CLAM.model_clam import CLAM_MB, CLAM_SB
from model.ABMIL.model_abmil import Attention, GatedAttention
from model.TransMIL.model_transmil import TransMIL
from model.FRMIL.model_frmil import FRMIL
from model.HIPT.model_hierarchical_mil import HIPT_LGP_FC
from model.ACMIL.model_acmil import ACMIL_GA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def define_clam(args):
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.mil_method != 'mil':
        if 'ResNet50' in args.ft_model:
            model_size = 'small'
        elif 'ViT_S_16' in args.ft_model:
            model_size = 'ultra_small'
        elif 'ViT_T_16' in args.ft_model:
            model_size = 'tiny'
        else:
            raise NotImplementedError
        model_dict.update({"size_arg": model_size})

    if args.subtyping:
        model_dict.update({'subtyping': True})

    if args.B > 0:
        model_dict.update({'k_sample': args.B})

    if args.inst_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        instance_loss_fn = SmoothTop1SVM(n_classes=2)
        instance_loss_fn = instance_loss_fn.cuda()
    else:
        instance_loss_fn = nn.CrossEntropyLoss()

    if args.mil_method == 'CLAM_SB':
        model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
    elif args.mil_method == 'CLAM_MB':
        model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
    else:
        raise NotImplementedError
    model.relocate()
    return model

def define_hipt(args):
    model_dict = {'path_input_dim': 384, "dropout": args.drop_out, 'n_classes': args.n_classes}
    model = HIPT_LGP_FC(**model_dict, freeze_4k=args.freeze_4k, pretrain_4k=args.pretrain_4k,
                        freeze_WSI=args.freeze_WSI, pretrain_WSI=args.pretrain_WSI)
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    return model

def define_transmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError

    model = TransMIL(model_size, n_classes=args.n_classes)
    model = model.to(device)
    return model

def define_abmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    if args.mil_method == 'ABMIL_att':
        model = Attention(model_size, args.n_classes)
    elif args.mil_method == 'ABMIL_gatedatt':
        model = GatedAttention(model_size, args.n_classes)
    else:
        raise NotImplementedError
    model = model.to(device)
    return model

def define_frmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    size_dict = {"tiny": [192, 128, 128], "ultra_small": [384, 192, 128],
                      "small": [1024, 512, 256], "big": [2048, 512, 384]}

    model = FRMIL(args, size_dict[model_size])
    model = model.to(device)
    return model

def define_acmil(args):
    if 'ResNet50' in args.ft_model:
        model_size = 'small'
    elif 'ViT_S_16' in args.ft_model:
        model_size = 'ultra_small'
    elif 'ViT_T_16' in args.ft_model:
        model_size = 'tiny'
    else:
        raise NotImplementedError
    size_dict = {"tiny": [192, 128], "ultra_small": [384, 128],
                      "small": [1024, 512], "big": [2048, 512]}
    args.size = size_dict[model_size]
    n_token = 5
    n_masked_patch = 10
    mask_drop = 0.6
    model = ACMIL_GA(args, n_token=n_token, n_masked_patch=n_masked_patch, mask_drop=mask_drop)
    model = model.to(device)
    return model

def define_model(args):
    if 'CLAM' in args.mil_method:
        model = define_clam(args)
    elif 'TransMIL' in args.mil_method:
        model = define_transmil(args)
    elif 'ABMIL' in args.mil_method:
        model = define_abmil(args)
    elif 'FRMIL' in args.mil_method:
        model = define_frmil(args)
    elif 'HIPT' in args.mil_method:
        model = define_hipt(args)
    elif 'ACMIL' in args.mil_method:
        model = define_acmil(args)
    else:
        raise NotImplementedError
    return model
