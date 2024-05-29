from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import torch
import time
import json
#import evaluate
import pandas as pd
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"
import sys
from peft import PeftModel
import fire
from utils.prompter import Prompter
import argparse
import nltk
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from utils.dataset_order import get_dataset_order


def main(args):
    dataset_order = get_dataset_order(args.dataset_id)
    service_name = dataset_order[args.service_begin_id]

    with_replay = args.with_replay
    dataset_id = args.dataset_id
    service_begin_id = args.service_begin_id
    lora_type = args.lora_type
    if with_replay:
        if lora_type == "vanilla":
            model_path = os.path.join("./checkpoint_files", args.model_name + "_dataset_id_"+str(dataset_id)+"_with_memoryreplay", str(service_begin_id)+"-"+service_name)
        else:
            model_path = os.path.join("./checkpoint_files", args.model_name + "_importance_dataset_id_"+str(dataset_id)+"_averaging_with_memoryreplay", str(service_begin_id)+"-"+service_name+ "" + lora_type)
    else:
        if lora_type == "vanilla":
            model_path = os.path.join("./checkpoint_files", args.model_name + "_dataset_id_"+str(dataset_id), str(service_begin_id)+"-"+service_name)
        else:
            model_path = os.path.join("./checkpoint_files", args.model_name + "_importance_dataset_id_"+str(dataset_id)+"_averaging", str(service_begin_id)+"-"+service_name+ "" + lora_type)
    
    if args.model_path != "":
        model_path = os.path.join("./checkpoint_files", args.model_path + "_dataset_id_"+str(dataset_id)+"_averaging", str(service_begin_id)+"-"+service_name+ "" + lora_type)
    
    
    if not os.path.exists(model_path):
        print(f"fine tuned mode path {model_path} not find!")
        sys.exit(1)   
    assert (
        model_path
    ), "Please specify a --model_path, e.g. --model_path='xxx'"

    if with_replay:
        if lora_type == "vanilla":
            output_dir = os.path.join("./output", args.model_name + "_dataset_id_"+str(dataset_id)+"_bwt_with_memoryreplay")
        else:
            output_dir = os.path.join("./output", args.model_name + "_importance_dataset_id_"+str(dataset_id)+"_bwt_with_memoryreplay"+ lora_type)  
    else:
        if lora_type == "vanilla":
            output_dir = os.path.join("./output", args.model_name + "_dataset_id_"+str(dataset_id)+"_bwt")
        else:
            output_dir = os.path.join("./output", args.model_name + "_importance_dataset_id_"+str(dataset_id)+"_bwt"+ lora_type)
    
    if args.output_dir != "":
        output_dir = os.path.join("./output", args.output_dir + "_dataset_id_"+str(dataset_id)+"_bwt"+ lora_type)
      
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"model_path: {model_path}")
    print(f"output_dir: {output_dir}")


    tokenizer = AutoTokenizer.from_pretrained( args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    service_id = service_begin_id
    
    print(f"current service name: {dataset_order[service_id]}... begin generating!")
    output_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")
    print(f"output filename: {output_file}")
    
    testfile_idx = "./data/SGD_single_service_test_T5/" + dataset_order[service_id] + "-test.idx"
    testfile_name = "./data/SGD_single_service_test_T5/" + dataset_order[service_id] + "-test-LLM_T5.json"
    
    print(f"test filename: {testfile_name}")

    if not os.path.isfile(output_file): 
        result_out = open(output_file, "w", encoding='utf-8')
        begin_id = 0 #

    else:
        with open(output_file, "r") as f:
            lines = f.readlines()
            begin_id = len(lines)
            f.close()
        result_out = open(output_file, "a", encoding='utf-8')
    
    idx_lines = open(testfile_idx).readlines()
    data = json.load(open(testfile_name)) 
    for idx_ in range(begin_id, len(data)):
        sample = data[idx_]
        idx_line = idx_lines[idx_].strip()

        Response_list = []

        #Response = evaluate(instruction = sample['instruction'], input = sample['input'])
        
        input_ids = tokenizer(sample['input'], return_tensors='pt').input_ids.cuda()
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        answer = tokenizer.decode(output[0])

        #print(answer)
        if "</s>" in answer:
            answer = answer.replace("</s>","")
        if "<pad>" in answer:
            answer = answer.replace("<pad>","")
                
        Response_list.append(answer.strip())

        #print("Input:", input2)
        print("Response list:", Response_list)
        print("Ground truth:", sample['output'])
        print()
        #sys.exit(1)
        # if "NONE" not in Response:
        #     break
        # if sample['output'] != "NONE":
        #     break
        result_out.write(idx_line + "|||" + str(Response_list))
        result_out.write("\n")

        #break
    result_out.close()
    print(f"current service name: {dataset_order[service_id]}... Generate End!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--with_replay", default=False, type=bool)

    parser.add_argument("--model_name", type=str, default="t5small", help = "")
    parser.add_argument("--lora_type", type=str, default="-averaging", help = "")
    
    parser.add_argument("--dataset_id", type=int, default=1, help = "")
    parser.add_argument("--service_begin_id", type=int, default=0, help = "")
    parser.add_argument("--max_new_tokens", type=int, default=128, help = "")

    parser.add_argument("--model_path", type=str, default="", help = "")
    parser.add_argument("--output_dir", type=str, default="", help = "")
            
    args = parser.parse_args()
    print(
        f"Testing T5 model with params:\n"
        f"dataset_id: {args.dataset_id}\n"
        f"service_begin_id: {args.service_begin_id}\n"
        f"with_replay: {args.with_replay}\n"
        f"model_name: {args.model_name}\n"
        f"lora_type: {args.lora_type}\n"
        f"max_new_tokens: {args.max_new_tokens}\n"
    )
    main(args)
