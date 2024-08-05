## Personalized LMs: Aligning Language Models with Diverse Human Preferences
<p align="left">
<!--   <a href='https://arxiv.org/abs/2407.12043'>
    <img src='https://img.shields.io/badge/Arxiv-2308.16905-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <!-- <a href=''>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
  <a href='https://nbviewer.org/github/allenai/noncompliance/blob/main/paper.pdf' class="image fit">
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow' alt="">
  </a> -->
  <a href="https://huggingface.co/datasets/allenai/compred">
    <img src="https://img.shields.io/badge/🤗-Data-orange">
  </a>

This repository provides the code to reproduce the results from the paper "Personalized LMs: Aligning Language Models with Diverse Human Preferences" which introduces a dataset **ComPreD** to study diverse preferences and personalization in language models using Reddit communities.

### 📄 Data
ComPreD contains five subsets divided based on factors driving diverging user preferences (we followed a similar process as SHP to create this dataset). 

| Subset(s)    | Factor |
| -------- | ------- |
| politics  | Ideologies    |
| gender_and_sexuality | Demographics     |
| finance, history    | Community Norms    |
| science | Level of expertise / Community Norms | 

You can also view and download the dataset on the [🤗 Huggingface Hub](https://huggingface.co/datasets/allenai/compred). And download them by:

```python
from datasets import load_dataset


# load finance train set
finance_train_pref = load_dataset("allenai/compred", "finance", split="train_pref")

# load finance test prompts
finance_test_prompts = load_dataset("allenai/coconot", "finance_test_prompts", split="test_prompts")
```

### 📦 Installing Packages
For evaluation, please first install [trl](https://github.com/huggingface/trl) module on top on which we build our finetuning and inference code. Please follow the installation available in trl.


### Training pipeline
We follow a two-stage process for finetuning models with and without community context: we first conduct supervised finetuning on the preferred responses from the train set followed by direct preference optimization (DPO) using both preferred and dispreferred responses from the train set. 

```
bash scripts/sft.sh <contextualize=True/False> <subset> <base_model_name_or_path>
bash scripts/dpo.sh <contextualize=True/False> <subset> <base_model_name_or_path> trained_models/<subset>-sft
```

Check out `scripts/sft.sh` and `scripts/dpo.sh` for more details

## Inference

Once the models are trained you can run inference on the test prompts as follows

```
bash predict.sh <contextualize=True/False> <subset> <batch_size> <base_model_name_or_path> trained_models/<subset>-sft
```

### Citation
Coming soon
