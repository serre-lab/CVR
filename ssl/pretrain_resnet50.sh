
backbone="resnet50"

max_epochs=100

n_samples=10000

seed=0

data_dir="<path to data>"

WORK_PATH="<path to experiment directory>"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

exp_name="${NOW}_ssl_task_${task}_nsamples_${n_samples}_model_${backbone}_epochs_${max_epochs}_seed_${seed}"
exp_dir="${WORK_PATH}/${exp_name}"


python main_moco.py \
  -a resnet50 \
  -b 1536 \
  -j 16 \
  --epochs=$max_epochs \
  --moco-m-cos \
  --multiprocessing-distributed \
  --seed $seed \
  --rank 0 \
  --world-size 1 \
  --dist-url 'tcp://localhost:10001' \
  --exp_dir $exp_dir \
  $data_dir \
