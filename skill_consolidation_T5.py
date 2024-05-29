
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
# import nltk
# nltk.download('averaged_perceptron_tagger')


"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
import time
import shutil
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from utils.dataset_order import get_dataset_order
from transformers import set_seed
import pandas as pd
import numpy as np
set_seed(42)

def get_common_threshold(ipt_scores, target_percentage=20):
    threshold = np.percentile(ipt_scores, 100 - target_percentage)
    return threshold


def get_lora_ipt(dataset_id, service_id, ipt_file_name, model_name):
    ipt_present_dic = {}
    ipt_previous_dic = {}
    
    dataset_order = get_dataset_order(dataset_id)
    ipt_file_present = "./ipt_file/"+ ipt_file_name +"_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    ipt_file_previous = "./ipt_file/"+ ipt_file_name +"_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_id-1)+"-"+dataset_order[service_id-1]+".csv"

    print(f"ipt_file_present: {ipt_file_present}")
    print(f"ipt_file_previous: {ipt_file_previous}")
    df1 = pd.read_csv(ipt_file_present)
    df2 = pd.read_csv(ipt_file_previous)
    
    
    min_value = df1['Importance_Score'].min()
    max_value = df1['Importance_Score'].max()
    df1['Importance_Score'] = (df1['Importance_Score'] - min_value) / (max_value - min_value)
    module_list = list(df1['Module_Name'])
    ipt_list = list(df1['Importance_Score'])
    #print(ipt_list)
    present_thresholds = get_common_threshold(ipt_list)
    for mm in range(len(module_list)):
        ipt_present_dic[module_list[mm]] = float(ipt_list[mm])

    
    min_value = df2['Importance_Score'].min()
    max_value = df2['Importance_Score'].max()
    df2['Importance_Score'] = (df2['Importance_Score'] - min_value) / (max_value - min_value)
    module_list = list(df2['Module_Name'])
    ipt_list = list(df2['Importance_Score'])
    #print(ipt_list)
    previous_thresholds = get_common_threshold(ipt_list)
    for mm in range(len(module_list)):
        ipt_previous_dic[module_list[mm]] = float(ipt_list[mm])


    assert set(ipt_present_dic.keys()) == set(ipt_previous_dic.keys())


    ipt_list1 = list(df1['Importance_Score'])
    ipt_list2 = list(df2['Importance_Score'])
    
    module_list1 = list(df1['Module_Name'])
    module_list2 = list(df2['Module_Name'])
    assert module_list1 == module_list2
    
    result_list = [0.3 * x + 0.7 * y for x, y in zip(ipt_list1, ipt_list2)]
    min_value = min(result_list)
    max_value = max(result_list)

    normalized_result = [(x - min_value) / (max_value - min_value) for x in result_list]

    data = {'Module_Name': module_list, 'Importance_Score': normalized_result}
    df = pd.DataFrame(data)

    csv_file_path = "./ipt_file/"+ ipt_file_name +"_averaging_dataset_id_"+ str(dataset_id) + "_" + str(service_id)+"-"+dataset_order[service_id]+".csv"
    
    df.to_csv(csv_file_path, index=False)
    
    
    return ipt_present_dic, ipt_previous_dic, present_thresholds, previous_thresholds
    #sys.exit(1)


def lora_averaging(checkpoint_name, present_thresholds, previous_thresholds, dataset_id, service_id, ipt_present_dic, ipt_previous_dic, lora_present_path, lora_previous_path):
    checkpoint_present = torch.load(os.path.join(lora_present_path,"pytorch_model.bin"), map_location=torch.device('cuda:0'))
    checkpoint_previous = torch.load(os.path.join(lora_previous_path,"pytorch_model.bin"), map_location=torch.device('cuda:0'))
    dataset_order = get_dataset_order(dataset_id)
    weighted_weights = {}


    count = 0
    for key in checkpoint_present:
        
        tensor_present = checkpoint_present[key].to('cuda:0')
        tensor_previous = checkpoint_previous[key].to('cuda:0')
        #print(checkpoint_present[key])
        #print(checkpoint_previous[key])
        #print(0.5*tensor1+0.5*tensor2)
        # print(key)
        # print(ipt_present_dic.keys())
        if key not in ipt_present_dic:
            weighted_weights[key] = tensor_present
            count += 1
            continue
        assert key in ipt_present_dic
        assert key in ipt_previous_dic
        ipt_score_present = ipt_present_dic[key]
        ipt_score_previous = ipt_previous_dic[key]
        #print(f"present ipt score is {ipt_score_present}, previous ipt score is {ipt_score_previous}")
        #sys.exit(1)

        if ipt_score_previous > previous_thresholds:
            if ipt_score_present > present_thresholds:

                weighted_weights[key] = 0.3*tensor_present + 0.7*tensor_previous
            else:

                weighted_weights[key] = tensor_previous
        else:
            if ipt_score_present > present_thresholds:
                weighted_weights[key] = tensor_present
            else:      
                weighted_weights[key] = 0.5*tensor_present + 0.5*tensor_previous
        
        # weighted_weights[key] = 0.5*tensor1+0.5*tensor2
        
    save_path = os.path.join("./checkpoint_files", checkpoint_name, str(service_id) + "-" + dataset_order[service_id]+"-averaging")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert len(weighted_weights) == len(checkpoint_present)
    torch.save(weighted_weights, os.path.join(save_path,"pytorch_model.bin"))


    source_path = os.path.join(lora_present_path, "config.json")
    destination_path = os.path.join(save_path, "config.json")
    shutil.copy(source_path, destination_path)
    
    source_path = os.path.join(lora_present_path, "generation_config.json")
    destination_path = os.path.join(save_path, "generation_config.json")
    shutil.copy(source_path, destination_path)

    assert count == 3
    
def train(
    # model/data params
    dataset_id: int = 2, # 1 - 5 
    service_begin_id: int = 1, #
    with_replay: bool = False, # 
    checkpoint_name: str = "t5small_importance_dataset_id_2_averaging",
    select_thresholds: float = -1, # 
    ipt_file_name: str = "t5small_Importance_Score",
    model_name: str = "t5small",
):
    checkpoint_name = checkpoint_name + "_dataset_id_" + str(dataset_id) + "_averaging"
    assert str(dataset_id) in checkpoint_name, "dataset_id"
    assert model_name != ""
    print("fine-grained model averaging begin!")
    print(f"model_name:{model_name}")
    print(f"select_thresholds:{select_thresholds}")
    print(f"dataset_id:{dataset_id}")
    print(f"checkpoint_name:{checkpoint_name}")
    print(f"service_begin_id:{service_begin_id}")
    
    service_id = service_begin_id
    dataset_order = get_dataset_order(dataset_id)
    if service_begin_id == 0:
        print("begin id = 0, just copy service id 0's weight.")
        save_path = os.path.join("./checkpoint_files", checkpoint_name, str(service_id) + "-" + dataset_order[service_id]+"-averaging")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        lora_present_path = os.path.join("./checkpoint_files", checkpoint_name, str(service_id) + "-" + dataset_order[service_id])
        source_path = os.path.join(lora_present_path, "config.json")
        destination_path = os.path.join(save_path, "config.json")
        shutil.copy(source_path, destination_path)
        
        source_path = os.path.join(lora_present_path, "generation_config.json")
        destination_path = os.path.join(save_path, "generation_config.json")
        shutil.copy(source_path, destination_path)
        
        source_path = os.path.join(lora_present_path, "pytorch_model.bin")
        destination_path = os.path.join(save_path, "pytorch_model.bin")
        shutil.copy(source_path, destination_path)
        
        sys.exit(1)
    

    previous_service_id = service_begin_id - 1 
    lora_present_path = os.path.join("./checkpoint_files", checkpoint_name, str(service_id) + "-" + dataset_order[service_id])
    lora_previous_path = os.path.join("./checkpoint_files", checkpoint_name, str(previous_service_id) + "-" + dataset_order[previous_service_id] + "-averaging")
    
    print(f"lora_present_path: {lora_present_path}")
    print(f"lora_previous_path: {lora_previous_path}")
    assert os.path.exists(lora_present_path), f"present lora_weights dir {lora_present_path} not find!"
    assert os.path.exists(lora_previous_path), f"previous lora_weights dir {lora_previous_path} not find!"

    assert os.path.exists(os.path.join(lora_present_path,"pytorch_model.bin")), f"present pytorch_model.bin not find!"
    assert os.path.exists(os.path.join(lora_previous_path,"pytorch_model.bin")), f"previous pytorch_model.bin not find!"

    output_dir = os.path.join("./checkpoint_files", checkpoint_name)

    ipt_present_dic, ipt_previous_dic, present_thresholds, previous_thresholds = get_lora_ipt(dataset_id, service_id, ipt_file_name, model_name)
    print(f"present_thresholds is {present_thresholds}")
    print(f"previous_thresholds is {previous_thresholds}")
    
    lora_averaging(checkpoint_name, present_thresholds, previous_thresholds, dataset_id, service_id, ipt_present_dic, ipt_previous_dic, lora_present_path, lora_previous_path)
        
    print("Model averaging success!")

if __name__ == "__main__":
    fire.Fire(train)
