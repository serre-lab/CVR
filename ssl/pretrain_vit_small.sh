
backbone="vit_small"

max_epochs=100

n_samples=10000

seed=0

data_dir="<path to data>"

WORK_PATH="<path to experiment directory>"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="${NOW}_ssl_task_${task}_nsamples_${n_samples}_model_${backbone}_epochs_${max_epochs}_seed_${seed}"
exp_dir="${WORK_PATH}/${exp_name}"

python main_moco.py \
  -a vit_small \
  -b 1024 \
  -j 9 \
  --optimizer=adamw \
  --lr=1.5e-4 \
  --weight-decay=.1 \
  --epochs=$max_epochs \
  --warmup-epochs=40 \
  --stop-grad-conv1 \
  --moco-m-cos \
  --moco-t=.2 \
  --seed $seed \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --exp_dir $exp_dir \
  $data_dir \
