from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import torch
import time
#import evaluate
import pandas as pd
import numpy as np
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import sys
from peft import PeftModel
import fire
from utils.prompter import Prompter
from utils.dataset_order import get_dataset_order
import argparse
import nltk
from transformers import set_seed
from utils.lora_importance_T5 import RankAllocator
set_seed(42)


def main(args):
    print("laile")
    with_replay = args.with_replay
    dataset_id = args.dataset_id
    dataset_order = get_dataset_order(dataset_id)
    service_id = args.service_begin_id
    model_name = args.model_path.split("/")[-1]
    if with_replay:
        data_path = "./data/SGD_single_service_train__T5_with_MemoryReplay_dataset_id_"+ str(dataset_id) + "/" + dataset_order[service_id] + "-train-LLM_T5.json"
        output_dir = os.path.join("./checkpoint_files", model_name +"_importance_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id)+"-"+dataset_order[service_id])
        # log_dir = os.path.join("./training_loss_log", model_name +"_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id)+"-"+dataset_order[service_id])
    else:
        data_path = "./data/SGD_single_service_train_T5/" + dataset_order[service_id] + "-train-LLM_T5.json"
        output_dir = os.path.join("./checkpoint_files", model_name +"_importance_dataset_id_"+str(dataset_id)+"_averaging", str(service_id)+"-"+dataset_order[service_id])
        # log_dir = os.path.join("./training_loss_log", model_name +"_dataset_id_"+str(dataset_id), str(service_id)+"-"+dataset_order[service_id])
    print(f"data path: {data_path}")
    if not os.path.exists(data_path):
        print(f"data_path {data_path} not find!")
        sys.exit(1)
    print(f"output_dir: {output_dir}")
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # print(f"log_dir: {log_dir}")

    if service_id == 0:
        resume_from_checkpoint = None
    else:
        last_service_name = dataset_order[service_id - 1]
        if with_replay:
            last_checkpoint_dir = os.path.join("./checkpoint_files", model_name +"_importance_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_id-1)+"-"+last_service_name)
        else:    
            last_checkpoint_dir = os.path.join("./checkpoint_files", model_name +"_importance_dataset_id_"+str(dataset_id)+"_averaging", str(service_id-1)+"-"+last_service_name+"-averaging")
        resume_from_checkpoint = last_checkpoint_dir

        if os.path.exists(resume_from_checkpoint):
            print(f"Restarting from {resume_from_checkpoint}")
        else:
            print(f"resume_from_checkpoint dir {resume_from_checkpoint} not find!")
            sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    max_input_length = args.max_input_length
    max_target_length = args.max_target_length
    padding = False #"max_length"
    ignore_pad_token_for_loss = args.ignore_pad_token_for_loss

    prefix = ""
    def preprocess_function(examples):
        inputs = examples['input'] 
        targets = examples['output']
        inputs = [prefix + inp for inp in inputs]
        #print(f"inputs : {inputs}")
        model_inputs = tokenizer(inputs, max_length=max_input_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        #print(f"model_inputs : {model_inputs}")
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data_files = {}
        data_files["train"] = data_path
        raw_datasets = load_dataset("json", data_files=data_files)

    else:
        print("error")
        sys.exit(1)
        
    val_set_size = 100
    if val_set_size > 0:
        train_val = raw_datasets["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(preprocess_function, batched=True)
        )
        val_data = (
            train_val["test"].shuffle().map(preprocess_function, batched=True)
        )
    else:
        train_data = raw_datasets["train"].shuffle().map(preprocess_function, batched=True)
        val_data = None
    print(f"train_data: {train_data}")
    print(f"val_data: {val_data}")    

    
    metric = load_metric("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result


    # print(preprocess_function(data["train"][:2]))
    # sys.exit(1)
    #tokenized_datasets = data
    tokenized_datasets = raw_datasets.shuffle().map(preprocess_function, batched=True,desc="Running tokenizer on train dataset")


    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    # print(model)
    # sys.exit(1)
    
    batch_size = args.batch_size
    model_name = args.model_path.split("/")[-1]
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy = "steps",
        save_strategy="steps",
        learning_rate = 3e-4,
        warmup_steps=50,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        weight_decay = 0.01,
        save_total_limit =2,
        load_best_model_at_end=True,
        eval_steps=500,
        save_steps=500,
        output_dir=output_dir,
        num_train_epochs = args.num_epochs,
        predict_with_generate = True,
        fp16 = True,
        push_to_hub = False,
        #logging_dir=log_dir,
    )
    
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
    )
    
    # new
    rankallocator = RankAllocator(
        model,
        init_warmup=50,
        beta1=args.beta1, 
        beta2=args.beta2, 
    )
    
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        ipt_score = rankallocator,
        train_dataset = train_data,
        eval_dataset = val_data,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    
    #resume_from_checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    ipt_name_list, ipt_score_list = rankallocator.calculate_score(metric="ipt")    
    print(ipt_name_list)
    print(ipt_score_list)
    
    if np.isnan(ipt_score_list).any():
        raise ValueError("important score NaN ")
    
    data = {'Module_Name': ipt_name_list, 'Importance_Score': ipt_score_list}
    df = pd.DataFrame(data)


    if service_id == 0:
        csv_file_path = "./ipt_file/"+ model_name+ "_Importance_Score_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    else:
        csv_file_path = "./ipt_file/"+ model_name +"_Importance_Score_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    df.to_csv(csv_file_path, index=False)

    model.save_pretrained(output_dir)
    
    # df_log = pd.DataFrame(trainer.state.log_history)
    #df_log.to_csv(os.path.join(log_dir,"train_log.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--with_replay", default=False, type=bool)
    parser.add_argument("--ignore_pad_token_for_loss", default=True, type=bool)

    parser.add_argument("--model_path", type=str, default="", help = "")
    parser.add_argument("--dataset_id", type=int, default=1, help = "")
    parser.add_argument("--service_begin_id", type=int, default=0, help = "")
    parser.add_argument("--batch_size", type=int, default=8, help = "")
    parser.add_argument("--num_epochs", type=int, default=2, help = "")

    parser.add_argument("--beta1", type=float, default=0.85, help = "")
    parser.add_argument("--beta2", type=float, default=0.85, help = "")


    parser.add_argument("--max_input_length", type=int, default=512, help = "")
    parser.add_argument("--max_target_length", type=int, default=128, help = "")

    args = parser.parse_args()
    print(
        f"Training T5 model with params:\n"
        f"dataset_id: {args.dataset_id}\n"
        f"service_begin_id: {args.service_begin_id}\n"
        f"base_model: {args.model_path}\n"
        f"beta1: {args.beta1}\n"
        f"beta2: {args.beta2}\n"
        f"batch_size: {args.batch_size}\n"
        f"num_epochs: {args.num_epochs}\n"
        f"max_input_length: {args.max_input_length}\n"
        f"max_target_length: {args.max_target_length}\n"
        f"with_replay: {args.with_replay}\n"
    )
    main(args)
