
import os

import sys
from typing import List
from peft import PeftModel
import fire
import torch
import transformers
import shutil
from datasets import load_dataset
from transformers import AutoConfig
import gc
import pandas as pd

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
from utils.lora_importance import RankAllocator
from transformers import set_seed
from transformers import TrainerCallback
set_seed(42)
from utils.dataset_order import get_dataset_order



def train(
    # model/data params
    base_model: str = "/llama-7b-hf",  # the only required argument
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 20,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    # llm hyperparams
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
    dataset_id: int = 2, # 1 - 5  
    service_begin_id: int = 0, # 
    with_replay: bool = False, #
    beta1: float = 0.85, 
    beta2: float = 0.85,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"with_replay: {with_replay}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    
    dataset_order = get_dataset_order(dataset_id)
    


    service_id = service_begin_id
    #os.environ['CUDA_VISIBLE_DEVICES']='4,6,7'

    print(f"current service name: {dataset_order[service_id]}... begin fine tuning!")
    
    
    if with_replay:
        data_path = "./data/SGD_single_service_train_with_MemoryReplay_dataset_id_"+ str(dataset_id) + "/" + dataset_order[service_id] + "-train-LLM.json"
        output_dir = os.path.join("./checkpoint_files", "dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id)+"-"+dataset_order[service_id])
    else:
        data_path = "./data/SGD_single_service_train/" + dataset_order[service_id] + "-train-LLM.json"
        output_dir = os.path.join("./checkpoint_files", "importance4_dataset_id_"+str(dataset_id)+"_averaging", str(service_id)+"-"+dataset_order[service_id])
    print(f"data path: {data_path}")
    if not os.path.exists(data_path):
        print(f"data_path {data_path} not find!")
        sys.exit(1)
    # if service_id == 0:
    #     output_dir = output_dir + "-averaging"
    print(f"output_dir: {output_dir}")
    

    if service_id == 0:
        lora_weights = ""
    else:
        last_service_name = dataset_order[service_id - 1]
        if with_replay:
            last_checkpoint_dir = os.path.join("./checkpoint_files", "importance4_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id-1)+"-"+last_service_name)
        else:    
            last_checkpoint_dir = os.path.join("./checkpoint_files", "importance4_dataset_id_"+str(dataset_id)+"_averaging", str(service_id-1)+"-"+last_service_name+"-averaging")
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

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    print("pre-trained model's BOS EOS and PAD token id:",bos,eos,pad," => It should be 1,2,none")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if service_id == 0:
        model = get_peft_model(model, config)
        print("fine tune lora from scratch!")
    # https://github.com/tloen/alpaca-lora/issues/44
    else:
        model = PeftModel.from_pretrained(model, lora_weights, is_trainable=True)
        print("continual fine tune lora!")
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

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
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True


    rankallocator = RankAllocator(
        model,
        init_warmup=50,
        beta1=beta1, 
        beta2=beta2, 
    )

    trainer = transformers.Trainer(
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
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=40 if val_set_size > 0 else None,
            save_steps=40,
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


    if service_id == 0:
        csv_file_path = "./ipt_file/Importance4_Score_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    else:
        csv_file_path = "./ipt_file/Importance4_Score_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    df.to_csv(csv_file_path, index=False)

    model.save_pretrained(output_dir, safe_serialization=False)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

      
if __name__ == "__main__":
    fire.Fire(train)
