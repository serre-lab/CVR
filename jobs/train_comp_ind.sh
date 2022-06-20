#!/bin/bash

data_dir="<path to data>"

WORK_PATH="<path to experiments>"

#########################

# experiment index from 1 to 42
# this parameters selects a pair of component tasks.
exp_index=1

protocol_setting="jobs/transfer_exp_config.txt"

lines=$(sed -n "${exp_index}p" ${protocol_setting})
declare -a all_tasks=($lines)
task_pt=${all_tasks[0]}
tasks_ft=("${all_tasks[@]:1}")

model_type='VIT'
# model_type='ResNet'
# model_type='SCL'
# model_type='WREN'
# model_type='SCLRN'

n_samples_pt=5000
n_samples_ft=(20 50 100 200 500 1000)

max_epochs_pt=100
early_stopping=1
refresh_rate=10

####################################################
####################################################
####################################################
####################################################
####################################################


if [ "$model_type" = "VIT" ]; then

    model="CNN"
    backbone='vit_small_ssl'

    mlp_hidden_dim=2048

    max_epochs_ft=(800 800 400 200 100 100 100 100 100)
    check_val_every_n_epoch_ft=(8 8 4 2 1 1 1 1 1)
    es_patience_ft=(20 20 20 20 20 20 20 20 20)
    lrs=(0.000003 0.000078 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001)
    # task_embedding=0
    
    lr=0.00001

    wd=0.0001

    checkpoint=""
    exp_type="cnn"

###########################
elif [ "$model_type" = "ResNet" ]; then
    model="CNN"
    backbone='resnet50'

    mlp_hidden_dim=2048

    max_epochs_ft=(800 800 400 200 100 100 100 100 100)
    check_val_every_n_epoch_ft=(8 8 4 2 1 1 1 1 1)
    es_patience_ft=(20 20 20 20 20 20 20 20 20)
    lrs=(0.00003 0.00078 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
    # task_embedding=0

    lr=0.0001

    wd=0.0001
    
    checkpoint=""
    exp_type="cnn"

###########################
elif [ "$model_type" = "SCL" ]; then

    model="SCN"
    backbone="scl"

    mlp_hidden_dim=2048

    max_epochs_ft=(800 800 400 200 100 100 100 100 100)
    check_val_every_n_epoch_ft=(8 8 4 2 1 1 1 1 1)
    es_patience_ft=(20 20 20 20 20 20 20 20 20)
    lrs=(0.0003 0.0078 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
    # task_embedding=0

    lr=0.001

    wd=0.0001
    
    checkpoint=""
    exp_type="cnn"

###########################
elif [ "$model_type" = "WREN" ]; then
    model="WREN"
    backbone="wren"

    mlp_hidden_dim=2048

    n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

    max_epochs_ft=(800 800 400 200 100 100 100 100 100)
    check_val_every_n_epoch_ft=(8 8 4 2 1 1 1 1 1)
    es_patience_ft=(20 20 20 20 20 20 20 20 20)
    lrs=(0.00003 0.00078 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
    # task_embedding=0

    lr=0.0001

    wd=0

    checkpoint=""
    exp_type="cnn"

###########################
elif [ "$model_type" = "SCLRN" ]; then
    model="SCNHead"
    # backbone="resnet50"
    backbone="resnet18"

    mlp_hidden_dim=128

    n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

    max_epochs_ft=(800 800 400 200 100 100 100 100 100)
    check_val_every_n_epoch_ft=(8 8 4 2 1 1 1 1 1)
    es_patience_ft=(20 20 20 20 20 20 20 20 20)
    lrs=(0.00015 0.0039 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005)
    # task_embedding=0

    lr=0.0005

    wd=0.0001

    checkpoint=""
    exp_type="cnn"

###########################
fi

####################################################
####################################################
####################################################
####################################################
####################################################

mlp_dim=128
task_embedding=0

seed=0

gpus=1
num_workers=4

batch_size=64 # 256

ckpt_period=1

config="configs/cvrt.yaml"
path_db="../visr_db/cnn_comp"

####################################################

max_epochs=${max_epochs_pt}
n_samples=${n_samples_pt}
task=${task_pt}

check_val_every_n_epoch=1
es_patience=30

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="${NOW}_pretrain_task_${task}_nsamples_${n_samples}_model_${model}_bb_${backbone}_epochs_${max_epochs}_seed_${seed}"
exp_dir="${WORK_PATH}/cnn_comp/${exp_name}"

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

####################################################

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

path_db="../visr_db/cnn_comp_ft"

checkpoint="--checkpoint ${exp_dir}"

finetune=1
freeze_pretrained=0

mkdir $path_db

for t in "${tasks_ft[@]}"
do

    task=$t

    for i in {0..5}
    do

        n_samples=${n_samples_ft[$i]}
        max_epochs=${max_epochs_ft[$i]}
        check_val_every_n_epoch=${check_val_every_n_epoch_ft[$i]}
        es_patience=${es_patience_ft[$i]}
        lr=${lrs[$i]}


        exp_name="${NOW}_finetune_pt_${task_pt}_task_${task}_nsamples_${n_samples}_model_${model}_bb_${backbone}_epochs_${max_epochs}_seed_${seed}"
        exp_dir="${WORK_PATH}/cnn_comp_ft/${exp_name}"

        mkdir $exp_dir
        
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
            --finetune $finetune \
            --refresh_rate $refresh_rate \
            --es_patience $es_patience \
            --check_val_every_n_epoch $check_val_every_n_epoch \

    done
done
