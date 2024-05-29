
for data_id in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=1 python generate_avgJGA.py \
        --load_8bit \
        --base_model 'decapoda-research/llama-7b-hf' \
        --dataset_id=${data_id}
        
done