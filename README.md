# Continual Dialog State Tracking via Task Skill Localization and Consolidation
Thank you for your interest in our work, and this is the original implementation of "TaSL: Continual Dialog State Tracking via Task Skill Localization and Consolidation", accepted to ACL 2024.

## Local Setup
```
conda create -n CDST python=3.8
conda activate CDST
pip install -r requirements.txt
```

## Step 1. Preliminary Preparation
The preprocessed SGD dataset for Continual DST is provided in the "/data" folder. If you are interested in the pre-processing, please check `utils/preprocess.py` and `utils/dataloader.py` at [here](https://github.com/thu-coai/CPT4DST).
For the four different backbone models, you can download they from the following links at huggingface:
* [T5-small](https://huggingface.co/google-t5/t5-small)
* [T5-base](https://huggingface.co/google-t5/t5-base)
* [Flan-T5-large](https://huggingface.co/google/flan-t5-large)
* [LLaMA-7B](https://huggingface.co/yahma/llama-7b-hf)


Then replace the corresponding files in the Transformers package with `trainer.py` and `trainer_seq2seq.py`, which have modified the source code to add our importance-aware skill localization method.


## Step 2. Training
We conducted experiments on four different student models:
### LLaMA-7B (`finetune_ContinualDST_LLaMA7B.py`)
```ruby
./scripts/run_train_TaSL_LLaMA7B.sh
```
### T5 Series Models (`finetune_ContinualDST_T5XL.py`)
```ruby
./scripts/run_train_TaSL_t5.sh
```
* --model_path: replace the position of various t5 models.

For LLaMA-7B, we use [LoRA](https://github.com/microsoft/LoRA) to accelerate the speed of fine-tuning process. At the end of training, the fine-tuned weights will be stored in `$checkpoint_files`. And the importance distribution of skill units will be stored in `$ipt_file`.

The code then automatically implements the fine-grained skill consolidation strategy (`skill_consolidation.py`).


## Step 3. Inference
Three metrics are used to measure the performance of our model for Continual Learning:

### **Avg.JGA**
```ruby
./scripts/run_generate_avgJGA.sh
```
### Forward Transfer (**FWT**)
```ruby
./scripts/run_generate_fwt.sh
```
### Backward Transfer (**BWT**)
```ruby
./scripts/run_generate_bwt.sh
```
After inference, the generated prediction results will be stored at `\output` folder. 


## Step 4. Evaluation
Then you can calculate three metrics by running
```ruby
./eval_avgJGA.py
./eval_fwt.py
./eval_bwt.py
```


