#!/bin/bash

# ssl="true"
ssl="false"


condition='ind' # individual training condition
# condition='all' # joint training condition over all tasks
# condition='elem' # joint training condition over elementary tasks
# condition='comp' # joint training condition over composition tasks
# condition='elem_comp' # joint training condition over composition tasks using an initialization trained on elementary tasks

model_type='ResNet'
# model_type='VIT'
# model_type='SCL'
# model_type='WREN'
# model_type='SCLRN'


## when condition is ind
n1=0
n2=102
task_list=$( seq $n1 $n2 )

cfg_list=(0 1 2 3 4 5)
# cfg_list=(7)
# cfg_list=(8)


ckpt_dir="<path to checkpoints>"

echo ${ckpt_dir}

if [ "$ssl" = "true" ]; then
    db_name_start="ssl_ft"
else
    db_name_start="cnn"
fi
echo ${db_name_start}


if [ "$condition" = "ind" ]; then
    db_name="${db_name_start}"
    # task_list=$( seq $n1 $n2 )
elif [ "$condition" = "all" ]; then
    task_list=("a")
    db_name="${db_name_start}_a"
elif [ "$condition" = "elem" ]; then
    task_list=("elem")
    db_name="${db_name_start}_a_elem"
elif [ "$condition" = "comp" ]; then
    task_list=("comp")
    db_name="${db_name_start}_a_comp"
elif [ "$condition" = "elem_comp" ]; then
    task_list=("comp")
    db_name="${db_name_start}_a_elem_comp"
fi
echo ${db_name}


###########################
if [ "$model_type" = "VIT" ]; then

    model="CNN"
    backbone='vit_small_ssl'

    mlp_hidden_dim=2048

    if [ "$condition" = "ind" ]; then
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(800 800 400 200 100 100 100 100 100)
        check_val_every_n_epoch=(8 8 4 2 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.000003 0.000078 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001)
        task_embedding=0

        wd=0.0001
    else
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(100 100 100 100 100 100 100 100 100)
        check_val_every_n_epoch=(1 1 1 1 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001)
        task_embedding=64

        wd=0.0001

    fi
    if [ "$ssl" = "true" ]; then
        checkpoint="${ckpt_dir}/ssl_checkpoint_vit_small_corrected.pth.tar"
    elif [ "$condition" = "elem_comp" ]; then
        checkpoint="<path to experiment dir>" # the experiment where the model is trained in the elem condition
    else
        checkpoint="none"
    fi

###########################
elif [ "$model_type" = "ResNet" ]; then
    model="CNN"
    backbone='resnet50'

    mlp_hidden_dim=2048

    if [ "$condition" = "ind" ]; then
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(800 800 400 200 100 100 100 100 100)
        check_val_every_n_epoch=(8 8 4 2 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.00003 0.00078 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
        task_embedding=0

        wd=0.0001
    else
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(100 100 100 100 100 100 100 100 100)
        check_val_every_n_epoch=(1 1 1 1 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
        task_embedding=64

        wd=0.0001
    fi
    if [ "$ssl" = "true" ]; then
        checkpoint="${ckpt_dir}/ssl_checkpoint_resnet50_corrected.pth.tar"
    elif [ "$condition" = "elem_comp" ]; then
        checkpoint="<path to experiment dir>" # the experiment where the model is trained in the elem condition
    else
        checkpoint="none"
    fi
    
###########################
elif [ "$model_type" = "SCL" ]; then

    model="SCN"
    backbone="scl"

    mlp_hidden_dim=2048

    if [ "$condition" = "ind" ]; then

        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(800 800 400 200 100 100 100 100 100)
        check_val_every_n_epoch=(8 8 4 2 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.0003 0.0078 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
        task_embedding=0

        wd=0.0001

    else

        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(100 100 100 100 100 100 100 100 100)
        check_val_every_n_epoch=(1 1 1 1 1 1 1 1 1)
        es_patience=(30 30 30 30 30 30 30 30 30)

        lrs=(0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
        task_embedding=64

        wd=0.0001

    fi
    if [ "$ssl" = "true" ]; then
        checkpoint="none"
    elif [ "$condition" = "elem_comp" ]; then
        checkpoint="<path to experiment dir>" # the experiment where the model is trained in the elem condition
    else
        checkpoint="none"
    fi
    
###########################
elif [ "$model_type" = "WREN" ]; then
    model="WREN"
    backbone="wren"

    mlp_hidden_dim=2048

    if [ "$condition" = "ind" ]; then
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(800 800 400 200 100 100 100 100 100)
        check_val_every_n_epoch=(8 8 4 2 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.00003 0.00078 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
        task_embedding=0

        wd=0
    else
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(100 100 100 100 100 100 100 100 100)
        check_val_every_n_epoch=(1 1 1 1 1 1 1 1 1)
        es_patience=(30 30 30 30 30 30 30 30 30)
        lrs=(0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)
        # lrs=(0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
        task_embedding=64

        wd=0
    fi
    if [ "$ssl" = "true" ]; then
        checkpoint="none"
    elif [ "$condition" = "elem_comp" ]; then
        checkpoint="<path to experiment dir>" # the experiment where the model is trained in the elem condition
    else
        checkpoint="none"
    fi

###########################
elif [ "$model_type" = "SCLRN" ]; then
    model="SCNHead"
    # backbone="resnet50"
    backbone="resnet18"

    mlp_hidden_dim=128

    if [ "$condition" = "ind" ]; then
        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(800 800 400 200 100 100 100 100 100)
        check_val_every_n_epoch=(8 8 4 2 1 1 1 1 1)
        es_patience=(20 20 20 20 20 20 20 20 20)
        lrs=(0.00015 0.0039 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005)
        task_embedding=0

        wd=0.0001

    else

        n_samples_pt_s=(20 50 100 200 500 1000 2000 5000 10000)

        max_epochs_pt=(100 100 100 100 100 100 100 100 100)
        check_val_every_n_epoch=(1 1 1 1 1 1 1 1 1)
        es_patience=(30 30 30 30 30 30 30 30 30)

        # lrs=(0.0008 0.0008 0.0008 0.0008 0.0008 0.0008 0.0008 0.0008 0.0008)
        lrs=(0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005 0.0005)
        task_embedding=64

        wd=0.0001
    fi
    if [ "$ssl" = "true" ]; then
        checkpoint="${ckpt_dir}/ssl_checkpoint_resnet50_corrected.pth.tar"
    elif [ "$condition" = "elem_comp" ]; then
        checkpoint="<path to experiment dir>" # the experiment where the model is trained in the elem condition
    else
        checkpoint="none"
    fi
    
###########################
fi

echo ${model} ${backbone} ${checkpoint}
    
###########################
###########################
###########################


for cfg in ${cfg_list[@]}
do
    for i in ${task_list[@]}
    do

        bash jobs/train.sh $i ${backbone} ${checkpoint} ${n_samples_pt_s[$cfg]} ${max_epochs_pt[$cfg]} ${check_val_every_n_epoch[$cfg]} ${es_patience[$cfg]} ${lrs[$cfg]} ${task_embedding} ${db_name} ${model} ${wd} ${ssl} ${mlp_hidden_dim}
        # sbatch -J VR_${db_name} --time=36:00:00 jobs/train.sh $i ${backbone} ${checkpoint} ${n_samples_pt_s[$cfg]} ${max_epochs_pt[$cfg]} ${check_val_every_n_epoch[$cfg]} ${es_patience[$cfg]} ${lrs[$cfg]} ${task_embedding} ${db_name} ${model} ${wd} ${ssl} ${mlp_hidden_dim}
    done
done



