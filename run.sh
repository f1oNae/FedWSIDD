#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=8GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=48:00:00
#PBS -l jobfs=10GB

PROJECT_ID=iq24
MODEL_NAME=ResNet50
FEATURE_TYPE=R50_features
MODEL_NAME_MIL=CLAM_SB
MODEL_NAME_FED=fed_desa
OPTIMIZER=adamw
OPTIMIZER_IMAGE=sgd
EXP_CODE=FedWSIDD_10_100_real_init_heter
N_DD=10
N_DD_PATCH=100
LOCAL_EPOCHS=50
G_EPOCHS=10
N_PROMPTS=1
REPEAT=3
LR=0.001
DATA_NAME=CAMELYON16 #CAMELYON16_IMAGE CAMELYON17
FT_ROOT=/g/data/$PROJECT_ID/CAMELYON16_patches
CODE_ROOT=/scratch/iq24/cc0395/FedDDHist

cd $CODE_ROOT
source /g/data/$PROJECT_ID/mmcv_env/bin/activate
echo "Current Working Directory: $(pwd)"

python3 main.py \
--heter_model \
--feature_type $FEATURE_TYPE \
--ft_model $MODEL_NAME \
--mil_method $MODEL_NAME_MIL \
--fed_method $MODEL_NAME_FED \
--opt $OPTIMIZER \
--contrast_mu 2 \
--mu 0.01 \
--repeat $REPEAT \
--n_classes 2 \
--drop_out \
--lr $LR \
--pretrain_kd \
--syn_size 64 \
--global_epochs_dm 50 \
--ipc $N_DD \
--nps $N_DD_PATCH \
--dc_iterations 1000 \
--image_lr 1.0 \
--image_opt $OPTIMIZER_IMAGE \
--B 8 \
--accumulate_grad_batches 1 \
--task $DATA_NAME \
--exp_code $EXP_CODE \
--global_epochs $G_EPOCHS \
--local_epochs $LOCAL_EPOCHS \
--bag_loss ce \
--inst_loss svm \
--results_dir $CODE_ROOT/exp \
--data_root_dir $FT_ROOT \
--prompt_lr 3e-3 \
--prompt_initialisation random \
--prompt_aggregation multiply \
--number_prompts $N_PROMPTS \
--key_prompt 4 \
--share_blocks 0 1 2 3 4 \
--share_blocks_g   5 6 \
--image_size 224 \
--init_real \
#--debug


