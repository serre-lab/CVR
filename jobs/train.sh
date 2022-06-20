#!/bin/bash

data_dir="<path to data>"

db_name=${10}
WORK_PATH="<path to experiments>/${db_name}"

task_pt=$1
model=${11}
backbone=$2
checkpoint=$3
n_samples_pt=$4
max_epochs_pt=$5
check_val_every_n_epoch=$6
es_patience=$7
lr=$8
wd=${12}
task_embedding=$9
ssl=${13}
mlp_hidden_dim=${14}
early_stopping=1
refresh_rate=50

seed=0
gpus=1
num_workers=4

if [ -z "$checkpoint" ]; then
checkpoint=""
elif [ "$checkpoint" = "none" ]; then
checkpoint=""
else
checkpoint="--checkpoint ${checkpoint} "
fi

mlp_dim=128
batch_size=64
ckpt_period=1

config="configs/cvrt.yaml"
path_db="../visr_db/${db_name}"

##############################################

max_epochs=${max_epochs_pt}
n_samples=${n_samples_pt}
task=${task_pt}

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="${NOW}_pretrain_task_${task}_nsamples_${n_samples}_model_${model}_bb_${backbone}_epochs_${max_epochs}_temb${task_embedding}_seed_${seed}"

if [ "${ssl}" = "true" ]; then
exp_name="${exp_name}_ftssl"
fi

exp_dir="${WORK_PATH}/${exp_name}"

mkdir $exp_dir
mkdir $path_db


echo $exp_dir

python main.py \
    --config $config \
    --exp_name $exp_name \
    --exp_dir $exp_dir \
    --data_dir $data_dir \
    --gpus $gpus \
    --seed $seed \
    --model $model \
    --lr $lr \
    --wd $wd \
    --mlp_dim $mlp_dim \
    --mlp_hidden_dim $mlp_hidden_dim \
    --task_embedding $task_embedding \
    --task $task \
    --batch_size $batch_size \
    --n_samples $n_samples \
    --ckpt_period $ckpt_period $checkpoint\
    --max_epochs $max_epochs \
    --backbone $backbone \
    --path_db $path_db \
    --num_workers $num_workers \
    --early_stopping $early_stopping \
    --refresh_rate $refresh_rate \
    --es_patience $es_patience \
    --check_val_every_n_epoch $check_val_every_n_epoch \
