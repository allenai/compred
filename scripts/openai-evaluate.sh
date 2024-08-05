
path1=/path1/outputs.jsonl
path2=/path2/outputs.jsonl
# evaluating_model=$1

pip install pysbd
pip install openai -U

python /code/openai_evaluate.py\
    --output_file1 $path1\
    --output_file2 $path2\
    --results_file /outputs/results.jsonl\
    --response_subkey "1"\
    --openai_key <key>\
    --org_id <org>\
    --max_tokens 10\
    --temperature 1.0
    # --model $evaluating_model\