#!/bin/bash


begin_id=0

for data_id in 1 2 3 4 5
do
    for ((ORDER=$begin_id; ORDER<15; ORDER++))
    do

        CUDA_VISIBLE_DEVICES=0 python finetune_continualDST_T5.py \
            --model_path '/home/data2/yujie/t5small' \
            --num_epochs=5 \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            
        wait

        CUDA_VISIBLE_DEVICES=0 python skill_consolidation_T5.py \
            --checkpoint_name 't5small_importance' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            --ipt_file_name 't5small_Importance_Score' \
            --model_name 't5small' \

    done
done