
contextualize=$1
subset=$2
bs=$3
model_path=$4

pip install pysbd
pip install -U bitsandbytes

if [ -z "$tokenizer_name" ]
then 
    tokenizer_name="meta-llama/Llama-2-7b-hf"
fi

mkdir -p /outputs/
python /code/predict.py\
    --dataset_name allenai/compred \
    --output_dir /outputs/\
    --subset $subset\
    --batch_size $bs\
    --model_dir $model_path\
    --tokenizer_name $tokenizer_name\
    --contextualize $contextualize