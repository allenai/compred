# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from functools import partial

import torch
from datasets import Dataset, load_dataset, load_from_disk
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser, TrainingArguments

from trl import DPOTrainer

from constants import HF_TOKEN as TOKEN
from huggingface_hub.commands.user import login; login(token=TOKEN)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    contextualize: Optional[str] = field(default=None, metadata={"help": "plain, subredditname, or contextualized"})
    subset: Optional[str] = field(default="all", metadata={"help": "all or subredditname"})

    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name: Optional[str] = field(
        default="huggyllama",
        metadata={"help": "the name of the SFT base model"},
    )
    tokenizer_name: Optional[str] = field(
        default="huggyllama",
        metadata={"help": "the name of the SFT base model"},
    )
    learning_rate: Optional[float] = field(default=3e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "warmup ratio"})
    warmup_steps: Optional[int] = field(default=1000, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "max number of training epochs"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=10000, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    dataset_name: Optional[str] = field(default=None, metadata={"help": "the output directory"})
    
    model_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

    use_4bit: Optional[bool] = field(default=False, metadata={"help": ""})
    use_lora: Optional[bool] = field(default=False, metadata={"help": ""})
    load_lora: Optional[bool] = field(default=False, metadata={"help": ""})

def subsample(dataset, ratio_thresh, examples_per_post):
    df = dataset.to_pandas()
    df = df[df["score_ratio"] >=  ratio_thresh]
    df = df.groupby("post_id").apply(
        lambda x: x.sample(n=min(examples_per_post, len(x)))
    )
    df = df.sample(n=len(df))
    return Dataset.from_pandas(df)

def get_paired_dataset(
    data_path: str,
    split: str = "train_pref",
    subset: str = "history",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    tokenizer=None,
) -> Dataset:

    dataset = load_dataset(data_path, subset, split=split)
    print(dataset)
   
    cols_to_remove = dataset.column_names
    cols_to_remove.remove("history")
    cols_to_remove.remove("human_ref_A")
    cols_to_remove.remove("human_ref_B")
    cols_to_remove.remove("labels")
    cols_to_remove.remove("score_ratio")

    dataset.remove_columns(cols_to_remove)

    print(f"Original {split} data size: {len(dataset)}")
    dataset = subsample(dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
    print(f"Filtered {split} data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(dataset)}")

    original_columns = dataset.column_names

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return_object = {"prompt": [], "chosen": [], "rejected": []}
        
        for subreddit, title, question, response_j, response_k, label in zip(samples['domain'], samples["title"], samples["history"], samples["human_ref_A"], samples["human_ref_B"], samples['labels']):
            subreddit = subreddit#.split("_")[0]
            
            response_j += tokenizer.eos_token
            response_k += tokenizer.eos_token

            instruction = "<|domain|>\n{domain}\n<|user|>{title}\n{post}\n<|assistant|>\n"
            if script_args.contextualize:
                domain = "r/"+subreddit
            else:
                domain = "Reddit"
                
            if label == 0:
                response_j, response_k = response_k, response_j
            
            # prompt = instruction + question + " \n\n COMMENT: "
            prompt = instruction.format(domain=domain, title=title, post=question)
            return_object['prompt'].append(prompt)
            return_object['chosen'].append(response_j)
            return_object['rejected'].append(response_k)

        return return_object
    
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
  
    script_args.model_name_or_path = os.path.join(script_args.model_dir, "sft", script_args.model_name.replace("/", "-"))
    script_args.output_dir = os.path.join(script_args.output_dir, "dpo", script_args.model_name.replace("/", "-"))
    
    if script_args.load_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            load_in_4bit=script_args.use_4bit,
            token=TOKEN,
            is_trainable=True
        )
        print("load lora")
        print(model.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=script_args.use_4bit,
            token=TOKEN,
            device_map="auto"#{"": 0}
        )
    model.config.use_cache = False
    print("embedding size", model.get_input_embeddings())
    
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.load_lora:
        model_ref = AutoPeftModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=script_args.use_4bit,
            token=TOKEN
        )
        print("load lora ref")
        model_ref.eval()
    else:
        model_ref = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            load_in_4bit=script_args.use_4bit,
            token=TOKEN,
            device_map="auto"
        )
        model_ref.eval()
    print("embedding size", model.get_input_embeddings())
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, token=TOKEN)
    if getattr(tokenizer, "pad_token", None) is None:
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "The Tokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    tokenizer.padding_side = "left"

    if script_args.load_lora:
        if len(tokenizer) != model.get_base_model().get_input_embeddings().weight.size(0):
            model.get_base_model().resize_token_embeddings(len(tokenizer))
            model_ref.get_base_model().resize_token_embeddings(len(tokenizer))
    else:
        if len(tokenizer) != model.get_input_embeddings().weight.size(0):
            model.resize_token_embeddings(len(tokenizer))
            model_ref.resize_token_embeddings(len(tokenizer))
    
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

    # def print_trainable_parameters(model):
    #     """
    #     Prints the number of trainable parameters in the model.
    #     """
    #     trainable_params = 0
    #     all_param = 0
    #     for _, param in model.named_parameters():
    #         all_param += param.numel()
    #         if param.requires_grad:
    #             trainable_params += param.numel()
    #     print(
    #         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    #     )
    # print_trainable_parameters(model)
    # print_trainable_parameters(model_ref)

    # 2. Load train dataset
    train_dataset = get_paired_dataset(data_path=script_args.dataset_name, split="train_pref", subset=script_args.subset, sanity_check=script_args.sanity_check, tokenizer=tokenizer)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 3. Load evaluation dataset
    eval_dataset = get_paired_dataset(data_path=script_args.dataset_name, split="validation_pref", subset=script_args.subset, sanity_check=True, tokenizer=tokenizer)
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    )

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        # max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        save_total_limit = 1,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="",
        ddp_find_unused_parameters=False,
    )

    if script_args.use_lora:
        # 5. initialize the DPO trainer
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
        )
    else:
        # 5. initialize the DPO trainer
        dpo_trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            beta=script_args.beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_prompt_length=script_args.max_prompt_length,
            max_length=script_args.max_length,
        )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    if script_args.use_lora:
        output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
        dpo_trainer.model.save_pretrained(output_dir)

        # Free memory for merging weights
        del model
        torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(script_args.output_dir, device_map={'':0}, torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
