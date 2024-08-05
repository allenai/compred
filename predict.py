from typing import Optional

import torch
from datasets import load_dataset

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, StoppingCriteria

from peft import AutoPeftModelForCausalLM, PeftModel

from constants import HF_TOKEN
from huggingface_hub.commands.user import login; login(token=HF_TOKEN)

import vllm 

from dataclasses import dataclass, field
from typing import Optional

import logging
import random
import pysbd

from tqdm import tqdm

import time
import os
import json

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences
        self.all_done = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # sequences_should_be_stopped = []
        if self.all_done is None:
            self.all_done = [False for _ in range(input_ids.shape[0])]
        for i in range(input_ids.shape[0]):
            if not self.all_done[i]:
                for stop_sequence in self.stop_sequences:
                    #print(stop_sequence, input_ids[i][-len(stop_sequence):].tolist())
                    if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                        self.all_done[i]
                        break
            # sequences_should_be_stopped.append(sequence_should_be_stopped)
            # print(sequence_should_be_stopped)
        return all(self.all_done)

@dataclass
class ScriptArguments:
    subset: str = field(default=None, metadata={"help": "which subset to use"})
    tokenizer_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model tokenizer"})
    output_dir: Optional[str] = field(default=None, metadata={"help": "the reward model name"})
    contextualize: Optional[bool] = field(default=False, metadata={"help": "whether to add subreddit context"})
    use_vllm: Optional[bool] = field(default=False, metadata={"help": "use vllm for inference (it adds nondeterminism even with greedy decoding)"})
    model_dir: Optional[str] = field(default="/models/", metadata={"help": "the reward model name"})
    dataset_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "decoding batch size"})
    randomize_context: Optional[bool] = field(default=False, metadata={"help": "add a random context to the model instead of the provided context"})


parser =  HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device=0
if not torch.cuda.is_available():
    device="cpu"  

subset = script_args.subset
model_path = script_args.model_dir

vllm_model = False
if script_args.use_vllm:
    try:
        model = vllm.LLM(
            model=model_path,
            tokenizer=script_args.tokenizer_name,
            tokenizer_mode="auto",
            trust_remote_code=True
        )
        vllm_model=True
    except Exception as e:
        print("vLLM failed, loading hf models with error", e)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)#,
    model.eval()
    model.to(device)
    print(f"device={model.device}")

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
if getattr(tokenizer, "pad_token", None) is None:
    num_added_tokens = tokenizer.add_special_tokens({
        "pad_token": "<pad>",
    })
    assert num_added_tokens in [0, 1], "The Tokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.unk_token
    model.resize_token_embeddings(len(tokenizer))
    
stop_id_sequences = []
for stopstring in ["\n<|", " <|", "<|"]:
    stop_id_sequences.append(tokenizer.encode(stopstring, add_special_tokens=False)[2:])
    #print(stopstring, stop_id_sequences[-1])

print("Loading dataset...")
ds = load_dataset(script_args.dataset_name, script_args.subset+"_test_prompts", split="test_prompts")

print(f"number of test examples: {len(ds)}")
cols_to_remove = ds.column_names
cols_to_remove.remove("title")
cols_to_remove.remove("history")
cols_to_remove.remove("post_id")
cols_to_remove.remove("domain")

ds = ds.remove_columns(cols_to_remove)
df = ds.to_pandas()

coveredposts = set()

logging.info(f"Done loading data with {ds.shape[0]} entries!")
c = 0

batchsize = script_args.batch_size
x = 0
st=time.time()
emptylines = []
text_batch = []
posttext_batch = []
title_batch = []
domains = []

os.makedirs(script_args.output_dir, exist_ok=True)

output_file = f"{script_args.output_dir}/outputs.jsonl"
print(output_file)
fout = open(output_file, "w")

y = 0

all_domains = set()
for i, row in df.iterrows():
    # domain = "_".join(row['domain'].split("_")[:-1])
    domain = row['domain']
    all_domains.add(domain)
all_domains = list(all_domains)

segmenter = pysbd.Segmenter(language="en", clean=False)
with torch.no_grad():
    for i, row in tqdm(df.iterrows(), total=df.shape[0]): 
        if (i == 0 or len(text_batch) < batchsize):
            # print(i, len(text_batch), batchsize)
            if row['post_id'] not in coveredposts:
                # cleanpost = clean_text(row["history
                # "])
                posttext_batch.append(row["history"])
                title_batch.append(row["title"])
                subreddit = row['domain']#.split("_")[0]
                if script_args.randomize_context:
                    new_subreddit = random.choice(all_domains)
                    dcount = 0
                    while new_subreddit == subreddit and dcount < 20:
                        new_subreddit = random.choice(all_domains)
                        dcount += 1
                    subreddit = new_subreddit
                domains.append(subreddit)
                instruction = "<|domain|>\n{domain}\n<|user|>\n{title_and_post}\n<|assistant|>\n"
                post=row["history"]
                title=row["title"]
                title_and_post=f"{title}\n{post}"
                
                sentences = []
                slack = 1024
                for s in segmenter.segment(title_and_post):
                    l = len(tokenizer(s).input_ids)
                    slack -= l

                    if slack > 0:
                        sentences.append(s)
                title_and_post = "".join(sentences)

                if script_args.contextualize:
                    domain = "r/"+subreddit
                else:
                    domain = "Reddit"
                text_batch.append(instruction.format(domain=domain, title_and_post=title_and_post))                    
                coveredposts.add(row['post_id'])
            if (i < df.shape[0]-1):
                continue    
        
        # print(i, len(text_batch), flush=True)
        # input()
        st = time.time()
        # print(text_batch)
        # tokenizer.padding = "left"
        tokens = tokenizer(text_batch, return_tensors="pt", padding=True).to(device) # check

        # for k in tokens.keys():
        #     tokens[k] = tokens[k].cuda()
        #     torch.cuda.empty_cache()
        # print(time.time() - st)
        st=time.time()
        
        text_outputs = [{} for _ in range(len(text_batch))]
        # for top_p in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        for beam_size in [1]:

            if vllm_model:
                if beam_size > 1:
                    sampling_params = vllm.SamplingParams(
                        use_beam_search=True,
                        temperature=0,
                        best_of=beam_size,
                        max_tokens=1024,
                    )
                else:
                    sampling_params = vllm.SamplingParams(
                        temperature=0,
                        max_tokens=1024,
                    )
                outputs = model.generate(text_batch, sampling_params, stopping_criteria=[KeyWordsCriteria(stop_id_sequences)])
                
                full_text_outputs = [it.outputs[0].text for it in outputs]

                for k, full_text_output in enumerate(full_text_outputs):
                    text_outputs[k][f'{beam_size}'] = full_text_output
            else:
                generation_kwargs = {"num_beams": beam_size, "do_sample": False, "repetition_penalty": 1.1}
                outputs = model.generate(**tokens, max_new_tokens=1024, stopping_criteria=[KeyWordsCriteria(stop_id_sequences)], **generation_kwargs) # check

                for k, output in enumerate(outputs):
                    full_text_output = tokenizer.decode(output, skip_special_tokens=True)
                    text_outputs[k][f'{beam_size}'] = full_text_output[len(text_batch[k]):]
            
        filewrite = [json.dumps({"title": title, "post": post, "domain": domain,"response": response}) for title, post, domain, response in zip(title_batch, posttext_batch, domains, text_outputs)]

        # print(filewrite)
        # input()
        fout.write("\n".join(filewrite) + "\n")
        fout.flush()

        # input("see")

        x += len(text_outputs)
        y += len(filewrite)
        # print(y)
        text_batch = []
        posttext_batch = []
        domains = []
        title_batch = []

        if i % 50 == 0:
            print(f"done total {y} inputs, took {(time.time()-st)/(x)} seconds per sentence since last log", flush=True)
            x = 0

print(f"wrote {y} lines")

fout.close()