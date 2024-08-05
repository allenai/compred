contextualize=$1
subset=$2
model_name=$3

pip install trl

torchrun --nnodes 1  --nproc_per_node 1 /code/sft.py\
    --dataset_name allenai/compred\
    --model_name=$model_name\
    --use_lora\
    --streaming\
    --subset $subset\
    --no_gradient_checkpointing\
    --learning_rate 1e-5\
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2\
    --contextualize $contextualize\
    --output_dir /models/