cd /home/comp/24483737/fc/sft/scripts
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/pyarrow
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/PIL
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/transformers/models
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/encodings
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/pygments/lexers
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10
touch /home/comp/24483737/.conda/envs/persona/lib/python3.10/site-packages/torch

wandb_run_name=mistral-1-rawfc-cvi-ep2
model_name_or_path=/home/comp/24483737/model/Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da
model_max_length=1024
train_data_path=/home/comp/24483737/fc/data/rawfc-post/small-original-half/train_rawfc_cvi.json
eval_data_path=/home/comp/24483737/fc/data/rawfc-post/small-original-half/val_rawfc_cvi.json
output_dir=/home/comp/24483737/fc/model/mistral-1-rawfc-cvi-ep2

# CUDA_VISIBLE_DEVICES=0,1,7 nohup bash /home/comp/24483737/fc/sft/scripts/train_7b_rawfc_mistral_cvi.sh > train_log.txt 2>&1 &

torchrun \
    --nnodes=1 \
    --nproc_per_node=3 \
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
    --num_train_epochs 2 \
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
    --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
