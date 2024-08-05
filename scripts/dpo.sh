pip install trl

contextualize=$1
subset=$2
base_model=$3

torchrun --nnodes 1  --nproc_per_node 1 /code/dpo.py\
    --model_name=$base_model\
    --tokenizer_name=$base_model\
    --contextualize $contextualize\
    --model_dir="/models/"\
    --num_train_epochs 1\
    --learning_rate 2e-7\
    --output_dir="/dpo_models/"\
    --subset $subset\
    --dataset_name allenai/compred\
    --no_gradient_checkpointing\
    --use_lora
    # --load_lora

# sbatch dpo_llama2.sh plain all 12345
# sbatch dpo_llama2.sh subredditname all 12346
# sbatch dpo_llama2.sh subredditname askphysics 12347
# sbatch dpo_llama2.sh subredditname explainlikeimfive 12348
# sbatch dpo_llama2.sh contextualized all 12349
