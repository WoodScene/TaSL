# new: apply a regularization term + second order localization
import os
os.environ['TRANSFORMERS_CACHE'] = '/data_8T2/yujie/cache'
import sys
from typing import List
from peft import PeftModel
import fire
import torch
import transformers
import shutil
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import gc
import numpy as np
import pandas as pd
from utils.lora_tailored_importance import RankAllocator
AutoConfig.default_cache_dir = '/data_8T2/yujie/cache'
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import time

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from transformers import set_seed

set_seed(42)
from utils.dataset_order import get_dataset_order
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("add a regularization term")

    def compute_orth_regu(self, model, regu_weight=0.1):
        # The function to compute orthongonal regularization for SVDLinear in `model`. 
        regu_loss, num_param = 0., 0
        for n,p in model.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                para_cov = p @ p.T if "lora_A" in n else p.T @ p 
                I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
                I.requires_grad = False
                regu_loss += torch.norm(para_cov-I, p="fro")
                num_param += 1
        return regu_weight*regu_loss/num_param

    def compute_loss(self, model, inputs, return_outputs=False):

        outputs = model(**inputs)
        loss = outputs.loss
        
        reg_term = self.compute_orth_regu(model)
        loss += reg_term

        return (loss, outputs) if return_outputs else loss


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 8,
    micro_batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_input_length: int = 1024,
    max_target_length: int = 128,
    val_set_size: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q", "v"
    ],
    # llm hyperparams
    ignore_pad_token_for_loss: bool = True,
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    dataset_id: int = 1, # 1 - 5  5次实验
    service_begin_id: int = 0, # 这个表示从哪个service开始训练，默认从头开始训练
    with_replay: bool = False, # 这个表示是否用有memory的数据集进行训练
    beta1: float = 0.85, 
    beta2: float = 0.85,
    hyper_para: int = 0, # 超参数的组合
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training T5 model with params:\n"
            f"base_model: {base_model}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"max_input_length: {max_input_length}\n"
            f"max_target_length: {max_target_length}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"ignore_pad_token_for_loss: {ignore_pad_token_for_loss}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"with_replay: {with_replay}\n"
            f"hyper_para: {hyper_para}\n"
            f"beta1: {beta1}\n"
            f"beta2: {beta2}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    
    dataset_order = get_dataset_order(dataset_id)


    service_id = service_begin_id
    #os.environ['CUDA_VISIBLE_DEVICES']='4,6,7'

    print(f"current service name: {dataset_order[service_id]}... begin fine tuning!")
    #model_name = base_model.split("/")[-1] + "lora" + str(hyper_para)
    model_name = base_model.split("/")[-1] + "lora" + str(hyper_para)
    
    
    if with_replay:
        data_path = "./data2/train_with_MemoryReplay_dataset_id_"+ str(dataset_id) + "/" + dataset_order[service_id] + "_T5.json"
        output_dir = os.path.join("./checkpoint_files", model_name +"_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id)+"-"+dataset_order[service_id])
        log_dir = os.path.join("./training_loss_log", model_name +"_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id)+"-"+dataset_order[service_id])
    else:
        data_path = "./data2/train/" + dataset_order[service_id] + "_T5.json"
        output_dir = os.path.join("./checkpoint_files", model_name +"_TaSL_dataset_id_"+str(dataset_id), str(service_id)+"-"+dataset_order[service_id])
        log_dir = os.path.join("./training_loss_log", model_name +"_dataset_id_"+str(dataset_id), str(service_id)+"-"+dataset_order[service_id])
    print(f"data path: {data_path}")
    if not os.path.exists(data_path):
        print(f"data_path {data_path} not find!")
        sys.exit(1)
    print(f"output_dir: {output_dir}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"log_dir: {log_dir}")
    # 首先需要检查一下上一个service的checkpoint文件是否存在
    if service_id == 0:
        lora_weights = ""
    else:
        last_service_name = dataset_order[service_id - 1]
        if with_replay:
            last_checkpoint_dir = os.path.join("./checkpoint_files", model_name + "_TaSL_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id-1)+"-"+last_service_name)
        else:    
            last_checkpoint_dir = os.path.join("./checkpoint_files", model_name + "_TaSL_dataset_id_"+str(dataset_id), str(service_id-1)+"-"+last_service_name+"-averaging")
        lora_weights = last_checkpoint_dir
        if not os.path.exists(lora_weights):
            print(f"lora_weights dir {lora_weights} not find!")
            sys.exit(1)
        
    print(f"lora_weights: {lora_weights}\n")

    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model, 
        # torch_dtype=torch.bfloat16, 
        device_map=device_map,
        )

    #tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, device_map="auto")
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")


    padding = False # "max_length"
    prefix = ""
    def preprocess_function(examples):
        inputs = examples['input'] 
        targets = examples['output']
        inputs = [prefix + inp for inp in inputs]
        #print(f"inputs : {inputs}")
        #sys.exit(1)
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        #print(f"model_inputs : {model_inputs}")
        #sys.exit(1)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        #print(labels["input_ids"])
        #print("laile")
        #print()
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    
    if service_id == 0:
        model = get_peft_model(model, config)
        print("fine tune lora from scratch!")
    # https://github.com/tloen/alpaca-lora/issues/44
    else:
        model = PeftModel.from_pretrained(model, lora_weights, is_trainable=True)
        print("continual fine tune lora!")
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path,cache_dir="/data_8T2/yujie/cache")
    else:
        data = load_dataset(data_path,cache_dir="/data_8T2/yujie/cache")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        )
        val_data = (
            train_val["test"].shuffle().map(preprocess_function,remove_columns=['input', 'output'], batched=True)
        )
    else:
        train_data = data["train"].shuffle().map(preprocess_function)
        val_data = None
    #train_data = train_data.remove_columns(['input', 'output'])
    #val_data = val_data.remove_columns(['input', 'output'])
    print(f"train_data: {train_data}")
    print(f"val_data: {val_data}")  
    # print(f"val_data: {val_data['input_ids']}")  
    # print(f"val_data: {val_data['attention_mask']}")  
    # print(f"val_data: {val_data['labels']}")  
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    rankallocator = RankAllocator(
        model,
        init_warmup=50,
        beta1=beta1, 
        beta2=beta2, 
        rank=lora_r,
        taylor="param_mix",
    )

    # update
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        ipt_score = rankallocator,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=40 if val_set_size > 0 else None,
            save_steps=400,
            output_dir=output_dir,
            save_total_limit=2,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    ipt_name_list, ipt_score_list = rankallocator.calculate_score(metric="ipt")    
    print(ipt_name_list)
    print(ipt_score_list)
    data = {'Module_Name': ipt_name_list, 'Importance_Score': ipt_score_list}
    df = pd.DataFrame(data)

    ipt_file = "./ipt_file"
    if not os.path.exists(ipt_file):
        os.makedirs(ipt_file)

    if service_id == 0:
        csv_file_path = "./ipt_file/" + model_name+ "_TaSL_Score_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    else:
        csv_file_path = "./ipt_file/" + model_name +"_TaSL_Score_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    print(f"csv_file_path is {csv_file_path}")
    df.to_csv(csv_file_path, index=False)


    model.save_pretrained(output_dir)
    df_log = pd.DataFrame(trainer.state.log_history)
    df_log.to_csv(os.path.join(log_dir,"train_log.csv"))
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    


if __name__ == "__main__":
    fire.Fire(train)
