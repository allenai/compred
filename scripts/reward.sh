contextualize=$1
subset=$2
model_name=$3

pip install -e /code/trl
pip install evaluate

torchrun --nnodes 1  --nproc_per_node 1 /code/reward_training.py \
    --dataset_name allenai/compred \
    --model_name $model_name \
    --output_dir /models/\
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2\
    --contextualize $contextualize\
    --subset $subset \
    --max_length 1024


