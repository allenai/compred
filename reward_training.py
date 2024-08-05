from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import json 

import os

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForSequenceClassification
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)

from functools import partial

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    dataset_name: Optional[str] = field(default="stanfordnlp/shp", metadata={"help": "the model name, write all for all domains combined"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": "contextualized or plain"})
    subset: Optional[str] = field(default="all", metadata={"help": "all or subredditname"})

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    
    score_ratio_threshold: Optional[float] = field(default=2.0)
    num_examples_per_post: Optional[float] = field(default=5)

    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=5e-6)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    output_dir: Optional[str] = field(
        default=".",
        metadata={
            "help": ""
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )

    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run eval after the first step"},
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

def subsample(dataset, ratio_thresh, examples_per_post):
    df = dataset.to_pandas()
    df = df[df["score_ratio"] >=  ratio_thresh]
    df = df.groupby("post_id").apply(
        lambda x: x.sample(n=min(examples_per_post, len(x)), random_state=42)
    )
    df = df.sample(n=len(df), random_state=42)
    return Dataset.from_pandas(df)


dataset = load_dataset(script_args.dataset_name, script_args.subset)
print(dataset)
train_dataset = dataset["train_pref"]
eval_dataset = dataset["test_pref"]

print(f"Original training data size: {len(train_dataset)}")
train_dataset = subsample(train_dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
print(train_dataset[0])
print(f"Filtered training data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(train_dataset)}")

print(f"Original validation data size: {len(eval_dataset)}, it's too large")
eval_dataset = subsample(eval_dataset, script_args.score_ratio_threshold, script_args.num_examples_per_post)
print(eval_dataset[0])
print(f"Filtered validation data with >{script_args.score_ratio_threshold} score ratio and {script_args.num_examples_per_post} comment pairs per post: {len(eval_dataset)}")

output_name = os.path.join(script_args.output_dir, "reward_model", script_args.model_name.replace("/", "-"))

training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    # max_steps=10,
    evaluation_strategy="steps",
    eval_steps=10000,
    save_strategy="steps",
    save_steps=10000,
    save_total_limit = 1,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    # deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)
# Load the value-head model and tokenizer.
tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token


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
        task_type=TaskType.SEQ_CLS,
    )

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name, num_labels=1, torch_dtype=torch.bfloat16
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
train_original_columns = train_dataset.column_names
eval_original_columns = eval_dataset.column_names

# Turn the dataset into pairs of post + answers, where text_j is the preferred question + answer and text_k is the other.
# Then tokenize the dataset.
def preprocess_function(examples, contextualize=False):
    new_examples = {
        "domain": [],
        "input_ids_j": [],
        "attention_mask_j": [],
        "input_ids_k": [],
        "attention_mask_k": [],
    }
    for subreddit, title, question, response_j, response_k, label in zip(examples['domain'], examples['title'], examples["history"], examples["human_ref_A"], examples["human_ref_B"], examples['labels']):
        subreddit = subreddit

        instruction = "<|domain|>\n{domain}\n<|user|>{post}\n<|assistant|>{comment}\n"

        if contextualize:
            domain = "r/"+subreddit
        else:
            domain = "Reddit"
        
        if label == 0:
            response_j, response_k = response_k, response_j

        tokenized_j = tokenizer(instruction.format(domain=domain, post=f"{title}\n{question}", comment=response_j), truncation=True, max_length=script_args.max_length)
        tokenized_k = tokenizer(instruction.format(domain=domain, post=f"{title}\n{question}", comment=response_k), truncation=True, max_length=script_args.max_length)

        new_examples["domain"].append(subreddit)
        new_examples["input_ids_j"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_j"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_k"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_k"].append(tokenized_k["attention_mask"])

    return new_examples

preprocess_function_instr = partial(preprocess_function, contextualize=script_args.contextualize)
# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function_instr,
    batched=True,
    num_proc=num_proc,
    remove_columns=train_original_columns,
)

eval_dataset = eval_dataset.map(
    preprocess_function_instr,
    batched=True,
    num_proc=num_proc,
    remove_columns=eval_original_columns,
)

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    print("eval_pred", eval_pred, flush=True)
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss. We use the InstructGPT pairwise logloss: https://arxiv.org/abs/2203.02155
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=script_args.max_length),
)


if script_args.eval_first_step:

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

trainer.train()#script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
trainer.model.save_pretrained(output_name + "_peft_last_checkpoint")

# Free memory for merging weights
# del model
# torch.cuda.empty_cache()

# model = AutoPeftModelForSequenceClassification.from_pretrained(output_name + "_peft_last_checkpoint", num_labels=1, device_map="auto", torch_dtype=torch.bfloat16)
# model = model.merge_and_unload()

# output_merged_dir = os.path.join(output_name + "_final_merged_checkpoint")
# model.save_pretrained(output_merged_dir, safe_serialization=True)

#########

def predict(model, inputs):
    score_j = model(input_ids=torch.LongTensor([inputs["input_ids_j"]]).to("cuda"), attention_mask=torch.LongTensor([inputs["attention_mask_j"]]).to("cuda"))
    score_k = model(input_ids=torch.LongTensor([inputs["input_ids_k"]]).to("cuda"), attention_mask=torch.LongTensor([inputs["attention_mask_k"]]).to("cuda"))
    
    print(score_j.logits, score_k.logits)

    return (score_j.logits[0] > score_k.logits[0]).long().tolist() #(max_value, max_id)

all_predictions = []
all_prediction_strings = []
print(eval_dataset)
correct = 0
domain2predictions = {}

with torch.no_grad():
    for i in range(len(eval_dataset)):
        print(i)
        prediction = predict(model, eval_dataset[i]) #make this batched?
        gold_label = 0#eval_dataset[i]['labels']
        # print(prediction, gold_label)
        correct += int(prediction[0] == gold_label)
        all_predictions.append(prediction[0])
        
        domain = eval_dataset[i]['domain']
        if domain not in domain2predictions:
            domain2predictions[domain] = []
        domain2predictions[domain].append(prediction[0])
        # all_prediction_strings.append(label_feature.int2str(prediction[0].item()))
        
print(correct, correct/len(all_predictions))
domain2accuracy = {domain: sum(predictions)/len(predictions) for domain, predictions in domain2predictions.items()}
print(domain2accuracy)
for domain, accuracy in domain2accuracy.items():
    print(f"{domain}\t{accuracy}\n")

with open(os.path.join(script_args.output_dir, "test_outputs.jsonl"), "w") as fres:
    for example, prediction, prediction_id in zip(eval_dataset, all_predictions, all_predictions):
        example['prediction'] = prediction
        example['prediction_id'] = prediction_id
        fres.write(json.dumps(example)+"\n")
        # fres.write(f"{r1}\t{l1}\t{r2}\t{l2}\n")
