cd /home/comp/24483737/fc/sft/scripts
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/pyarrow
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/PIL
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/transformers/models
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/encodings
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/pygments/lexers
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10

wandb_run_name=qwen-3-rawfc-cv-256-ep1
model_name_or_path=/home/comp/24483737/model/Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
model_max_length=512
train_data_path=/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cv.json
eval_data_path=/home/comp/24483737/fc/data/rawfc-post/small-original-half/val_rawfc_cv.json
output_dir=/home/comp/24483737/fc/model/qwen-3-rawfc-cv-256-ep1

# CUDA_VISIBLE_DEVICES=4,3,6,7 bash /home/comp/24483737/fc/sft/scripts/train_7b_rawfc_qwen3_cv.shx

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=29057 \
    ../train.py \
    --wandb_run_name ${wandb_run_name}\
    --model_name_or_path ${model_name_or_path} \
    --model_max_length ${model_max_length} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --output_dir ${output_dir} \
    --eval_strategy "epoch" \
    --save_strategy "no" \
    --logging_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Qwen3DecoderLayer' \