# pairwise evaluate using openai-gpt models
import argparse
import json

import random

from openai import OpenAI
from tqdm import tqdm

import pysbd

from transformers import AutoTokenizer

segmenter = pysbd.Segmenter(language="en", clean=False)

# from huggingface_hub.commands.user import login; login(token="access_token")
# print("loggedin!")

# llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

system_instruction="You are a highly efficient assistant, who evaluates and selects the best large language model (LLMs) based on the quality of their responses to a given instruction. This process will be used to create a leaderboard reflecting the most accurate and human-preferred answers."
full_instruction="""I require a leaderboard for various Reddit comment generator models. I'll provide you with posts selected from Reddit given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the output that will be upvoted more in the subreddit it was asked in.

## Subreddit

{domain}

## Instruction

{instruction}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

"model_identifier": "m",
"output": "{output_1}"

"model_identifier": "M",
"output": "{output_2}"

## Task

Evaluate the models based on the quality and relevance of their outputs, and select the model that generated the best output. Answer by providing the model identifier of the best model. We will use your output as the name of the best model, so make sure your output only contains one of the following model identifiers and nothing else (no quotes, no spaces, no new lines, ...): m or M.   

## Best Model Identifier
"""
# . If both are acceptable, then answer B. If neither are acceptable, then answer N.

parser = argparse.ArgumentParser()
parser.add_argument("--openai_key", required=True)
parser.add_argument("--org_id", required=True)
parser.add_argument("--model", default="gpt-4-1106-preview", choices=["gpt-4", "gpt-4o", "gpt-3.5-turbo", "gpt-4-1106-preview"])
parser.add_argument("--max_tokens", type=int, default=10)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--output_file1", type=str, required=True)
parser.add_argument("--output_file2", type=str, required=True)
parser.add_argument("--response_subkey", type=str, required=True)
parser.add_argument("--results_file", type=str, required=True)

args = parser.parse_args()

client = client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=args.openai_key,
    organization = args.org_id
)
# openai.api_key = args.openai_key
# openai.organization = args.org_id

MODEL_COSTS = {"gpt-4o": [0.000005, 0.000015], "gpt-4": [0.00003, 0.00006], "gpt-3.5-turbo": [0.0000015, 0.000002], "gpt-4-1106-preview": [0.00001, 0.00003]}
in_cost_per_token, out_cost_per_token = MODEL_COSTS[args.model]

total_input_tokens = 0
total_output_tokens = 0
coveredposts = set()
count = 0
left, right, mid = 0, 0, 0
domain2wins = {}

with open(args.output_file1) as fout1, open(args.output_file2) as fout2, open(args.results_file, "w") as fres:
    for idx, line1 in tqdm(enumerate(fout1)):
        line2 = fout2.readline()
        requestdata = json.loads(line1)
        requestdata2 = json.loads(line2)

        # if requestdata['post_id'] in coveredposts:
        #     continue
        count += 1
        # coveredposts.add(requestdata['post_id'])

        responsedata = {}
        responsedata['evaluating_model'] = args.model
        
        domain = requestdata['domain'].split("_")[0]
        post = f"{requestdata['title']}\n{requestdata['post']}"

        responsedata['domain'] = domain
        
        prompt1 = requestdata["response"][args.response_subkey]
        prompt2 = requestdata2["response"][args.response_subkey]

        flipped = False 
        if random.randint(0, 1) == 1:
            flipped = True
            prompt2, prompt1 = prompt1, prompt2

        model_input = full_instruction.format(domain=domain, instruction=post, output_1=prompt1, output_2=prompt2)
        responsedata['instruction'] = model_input

        response = client.chat.completions.create(
            model=args.model,
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": model_input}
                ],
            max_tokens=args.max_tokens,
            n=1,
            temperature=args.temperature
        )

        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens

        if count % 20 == 0:
            print(f"estimated cost so far= ${in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens}")
        
        response = response.choices[0].message.content
        
        responsedata['result'] = response.strip()
        responsedata['flipped'] = flipped
        if not flipped:
            if response == "m":
                left += 1
                winner = "m"
            elif response == "M":
                right += 1
                winner = "M"
            else:
                mid += 1
                winner = "neither"
        else:
            if response == "m":
                right += 1
                winner = "M"
            elif response == "M":
                left += 1
                winner = "m"
            else:
                mid += 1
                winner = "neither"

        if responsedata["domain"] not in domain2wins:
            domain2wins[responsedata["domain"]] = {"m": 0, "M": 0, "neither": 0}

        domain2wins[responsedata["domain"]][winner] += 1
        
        fres.write(json.dumps(responsedata)+"\n")
        fres.flush()

        if "gpt-4" in args.model and count > 100:
            break

print("only doing 2000 for now to estimate cost")
total = left + mid + right 
print (left/total, mid/total, right/total)

print(f"{total_input_tokens=}")
print(f"{total_output_tokens=}")
print(f"estimated cost= ${in_cost_per_token * total_input_tokens + out_cost_per_token * total_output_tokens}")

print(domain2wins)

with open(args.results_file+".wins", "w") as fwins:
    fwins.write(json.dumps(domain2wins))