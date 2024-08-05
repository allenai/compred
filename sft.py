# Fine-Tune causal LMs on reddit or stackoverflow datasets
import os
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments

import trl 

# print(trl.SFTTrainer)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from functools import partial

from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    output_dir: Optional[str] = field(default="trained_models/shp/sft", metadata={"help": "the directory where the trained model will be saved"})
    
    contextualize: Optional[bool] = field(default=False, metadata={"help": ""})

    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})

    dataset_name: Optional[str] = field(default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"})

    subset: Optional[str] = field(default=None, metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    max_steps: Optional[int] = field(default=500, metadata={"help": "the maximum number of sgd steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the maximum number of sgd steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=500, metadata={"help": "the evaluation frequency"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "the per device train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "the per device eval batch size"})
    gradient_accumulation_steps: Optional[int] = field(default=128, metadata={"help": "the gradient accumulation steps"})
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    use_lora: Optional[bool] = field(default=False, metadata={"help": ""})

    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "the learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # num_warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    warmup_ratio: Optional[float] = field(default=0.03, metadata={"help": "warmup ratio"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    #output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
script_args.output_dir = os.path.join(script_args.output_dir, "sft", script_args.model_name.replace("/", "-"))

os.makedirs(script_args.output_dir, exist_ok=True)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    return text


def prepare_sample_text_plain(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_plain"] + example["output"]
    return text


def prepare_sample_text_subredditname(example):
    """Prepare the text from a sample of the dataset."""
    # text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
    text = example["input_subredditname"] + example["output"]
    return text

prepare_sample_text_fns ={
    False: prepare_sample_text_plain,
    True: prepare_sample_text_subredditname,
}


def preprocess_function_subredditname(example):
    domain = example['domain']
    instruction = f"<|domain|>\nr/{domain}\n<|user|>\n"
    prompt = f"{example['title']}\n{example['history']}"
      
    preferred_output = example['human_ref_A']
    dispreferred_output = example['human_ref_B']
    if example['labels'] == 0:
        preferred_output, dispreferred_output = dispreferred_output, preferred_output

    example_text = instruction + prompt + "\n<|assistant|>\n" + preferred_output + tokenizer.eos_token
    return example_text


def preprocess_function_plain(example):
    """Prepare the text from a sample of the dataset."""
    instruction = f"<|domain|>\nReddit\n<|user|>\n"
    prompt = f"{example['title']}\n{example['history']}"

    preferred_output = example['human_ref_A']
    dispreferred_output = example['human_ref_B']
    if example['labels'] == 0:
        preferred_output, dispreferred_output = dispreferred_output, preferred_output

    example_text = instruction + prompt + "\n<|assistant|>\n" + preferred_output + tokenizer.eos_token

    return example_text


preprocess_functions ={
    False: preprocess_function_plain,
    True: preprocess_function_subredditname,
}


def chars_token_ratio(dataset, tokenizer, contextualize=False, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = preprocess_functions[contextualize](example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))
    print(total_characters, total_tokens)
    return total_characters / (total_tokens+1e-6)


def create_dataset(tokenizer, args):
    if getattr(tokenizer, "pad_token", None) is None:
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "The Tokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    
    data_path = args.dataset_name # will contain train, test, validation
    subset = args.subset
    contextualize = args.contextualize

    def subsample(dataset, ratio_thresh, examples_per_post):
        df = dataset.to_pandas()
        df = df[df["score_ratio"] >=  ratio_thresh]
        df = df.groupby("post_id").apply(
            lambda x: x.sample(n=min(examples_per_post, len(x)))
        )
        df = df.sample(n=len(df))
        return Dataset.from_pandas(df)
    
    dataset = load_dataset(data_path, subset)
    print(dataset)

    train_data = dataset["train_pref"]
    valid_data = dataset["validation_pref"]

    print(f"Original training data size: {len(train_data)}")
    train_data = subsample(train_data, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered training data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(train_data)}")

    print(f"Original validation data size: {len(valid_data)}, it's too large")
    valid_data = subsample(valid_data, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered validation data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer, args.contextualize)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    preprocess_function = preprocess_functions[contextualize]
    
    train_dataset = train_data.map(lambda x: {'text': preprocess_function(x)})
    valid_dataset = valid_data.map(lambda x: {'text': preprocess_function(x)})

    return train_dataset, valid_dataset


tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, token=HF_TOKEN)
tokenizer.padding_side = "left"

train_dataset, eval_dataset = create_dataset(tokenizer, script_args)
print("Datasets loaded")

if script_args.use_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
        quantization_config=bnb_config,
        device_map={'':0},
        trust_remote_code=True,
        token=HF_TOKEN
    )
    base_model.config.use_cache = False
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        device_map={'':0},
        trust_remote_code=True,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16
    )
    base_model.config.use_cache = False

if script_args.use_lora:
    peft_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )

training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    eval_steps=script_args.eval_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit = 1,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_ratio=script_args.warmup_ratio,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft",
    ddp_find_unused_parameters=False,
)

response_template_with_context = "x<|assistant|>\n" 
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[1:] #`x`` will take one token
print("response_template_ids", response_template_ids)

data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

base_model.resize_token_embeddings(len(tokenizer))

if script_args.use_lora:
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
else:
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        data_collator=data_collator,
        max_seq_length=script_args.seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

trainer.train()
trainer.save_model(script_args.output_dir)

if script_args.use_lora:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del base_model
    torch.cuda.empty_cache()

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map={'':0}, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
