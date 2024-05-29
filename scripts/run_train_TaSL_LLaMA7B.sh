#!/bin/bash
begin_id=0

for data_id in 1 2 3 4 5
do
    for ((ORDERR=$begin_id; ORDERR<15; ORDERR++))
    do
        WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun \
            --nproc_per_node=2 \
            --master_port=1239 \
            finetune_continualDST.py \
            --base_model 'decapoda-research/llama-7b-hf' \
            --num_epochs=5 \
            --cutoff_len=512 \
            --group_by_length \
            --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
            --micro_batch_size=16 \
            --batch_size=128 \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDERR} 

        wait

        CUDA_VISIBLE_DEVICES=1 python skill_consolidation.py \
            --checkpoint_name 'importance' \
            --dataset_id=${data_id} \
            --service_begin_id=${ORDERR} \
            --ipt_file_name 'Importance_Score' \

    done
done