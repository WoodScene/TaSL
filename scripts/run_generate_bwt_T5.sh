#!/bin/bash


begin_id=0

for data_id in 1 2 3 4 5
do

    for ((ORDER=$begin_id; ORDER<14; ORDER++))
    do

        CUDA_VISIBLE_DEVICES=1 python generate_bwt_T5.py \
            --model_name 't5small' \
            --model_path 't5small_importance' \
            --output_dir 't5small_importance' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDER} \

    done
done