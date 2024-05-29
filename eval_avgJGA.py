
import os
import sys
import json
from glob import glob
import argparse
import pandas as pd
from utils.dataset_order import get_dataset_order

def main(args):
    dataset_order = get_dataset_order(args.dataset_id)

    output_dir = os.path.join("./output", args.test_data_name)
    if not os.path.exists(output_dir):
        print(f"results dir {output_dir} not find!")
        sys.exit(1)
    
    JGA_list = []
    service_name_list = []
    #print(f"predict result dir: {output_dir}")
    #print("Calculating JGA score for each service.....")
    for service_id in range(0, len(dataset_order)):
        
        result_file = os.path.join(output_dir, str(service_id)+"-"+dataset_order[service_id] +"_result.txt")        
        model_results = open(result_file, "r").readlines()
        
        testfile_idx = "./data/SGD_single_service_test/" + dataset_order[service_id] + "-test.idx"
        testfile_name = "./data/SGD_single_service_test/" + dataset_order[service_id] + "-test-LLM.json"
        idx_lines = open(testfile_idx).readlines()
        test_lines = json.load(open(testfile_name))
        
        assert len(model_results) == len(idx_lines) == len(test_lines), "line number error!" + str(len(idx_lines)) + " " + str(len(model_results))
        
        dial_dic = {}
        # dia_dic['state'] = {}
        # dia_dic['pred_state'] = {}
        for idx_ in range(0, len(idx_lines)):
            true_state = test_lines[idx_]['output']
            result_line = model_results[idx_].strip().lower()
            idx_line = idx_lines[idx_].strip()
            if idx_line not in result_line:

                print(idx_line,result_line )
                sys.exit(1)
            pred_state = result_line.split("|||")[-1]
            # pred_state 
            pred_state = eval(pred_state)[0]
            if "</s>" in pred_state:
                pred_state = pred_state.replace("</s>","")
            
            dial_json_n, dial_idx, turn_idx, frame_idx, d_name, s_name = idx_line.split("|||") 
            dic_key_name = dial_idx + "-" + turn_idx
            if dic_key_name not in dial_dic:
                dial_dic[dic_key_name] = {}
                dial_dic[dic_key_name]['state'] = {}
                dial_dic[dic_key_name]['pred_state'] = {}
                dial_dic[dic_key_name]['state'][d_name + "-" + s_name] = true_state
                dial_dic[dic_key_name]['pred_state'][d_name + "-" + s_name] = pred_state
            else:
                dial_dic[dic_key_name]['state'][d_name + "-" + s_name] = true_state
                dial_dic[dic_key_name]['pred_state'][d_name + "-" + s_name] = pred_state
        # with open("pred.json", 'w') as f:
        #         json.dump(dial_dic, f, indent=4)   
        

        joint_total = 0
        joint_acc = 0
        for turn_id in dial_dic:
            joint_total += 1
            true_state_dic = dial_dic[turn_id]['state']
            pred_state_dic = dial_dic[turn_id]['pred_state']
            if set(true_state_dic.items()) == set(pred_state_dic.items()):
                joint_acc += 1
        joint_accuracy = joint_acc / joint_total
        #print('{} JGA: {}'.format(dataset_order[service_id], joint_accuracy))
        JGA_list.append(joint_accuracy)
        #break    
    print(f"average JGA is {sum(JGA_list) / len(JGA_list)}")
    print()
    average_JGA = sum(JGA_list) / len(JGA_list)
    JGA_list.append(average_JGA)
    service_name_list = dataset_order
    service_name_list.append("Average")
    dataframe = pd.DataFrame({'service_name':service_name_list,'JGA score':JGA_list})
    
    return average_JGA
            
if __name__=='__main__':
    mean_list = []
    for data_id in range(1):
        
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset_id", default=data_id+1, type=int)
        parser.add_argument("--test_data_name", type=str, default="t5small_importance_dataset_id_1_avgJGA-averaging", help = "_with_memoryreplay")


        args = parser.parse_args()
        average_JGA = main(args)
        mean_list.append(average_JGA)
    import numpy as np
    print(np.mean(mean_list))