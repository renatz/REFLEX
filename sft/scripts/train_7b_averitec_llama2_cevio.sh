wandb_run_name=llama-2-averitec-cevio-ep2-final_2802
model_name_or_path=/home/comp/24483737/model/llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
model_max_length=512
train_data_path=/home/comp/24483737/fc/data/post-averitec/final_2802_nolabelerror/train_averitec_cevi_ordinal.json
eval_data_path=/home/comp/24483737/fc/data/post-averitec/lalast/dev_averitec_cevi_ordinal_remove.json
output_dir=/home/comp/24483737/fc/model/llama-2-averitec-cevio-ep2-final_2802

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=29051 \
    ../train.py \
    --wandb_run_name ${wandb_run_name}\
    --model_name_or_path ${model_name_or_path} \
    --model_max_length ${model_max_length} \
    --train_data_path ${train_data_path} \
    --eval_data_path ${eval_data_path} \
    --output_dir ${output_dir} \
    --evaluation_strategy "epoch" \
    --save_strategy "no" \
    --logging_steps 1 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \

