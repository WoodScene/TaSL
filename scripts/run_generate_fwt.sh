#!/bin/bash


begin_id=1

for data_id in 1 2 3 4 5
do

    for ((ORDER=$begin_id; ORDER<15; ORDER++))
    do

        CUDA_VISIBLE_DEVICES=1 python generate_fwt.py \
            --load_8bit \
            --base_model 'decapoda-research/llama-7b-hf' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \
            
    done
done