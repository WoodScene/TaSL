

for data_id in 1 2 3 4 5
do
    CUDA_VISIBLE_DEVICES=0 python generate_avgJGA_T5.py \
        --model_name 't5small' \
        --model_path 't5small_importance' \
        --output_dir 't5small_importance' \
        --dataset_id=${data_id}
        
done