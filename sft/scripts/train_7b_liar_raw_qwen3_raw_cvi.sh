# CUDA_VISIBLE_DEVICES=0,1,3,7 nohup bash /home/comp/24483737/fc/sft/scripts/train_7b_liar_raw_qwen3_raw_cvi_big_sep.sh > train_log_2.txt 2>&1 &

cd /home/comp/24483737/fc/sft/scripts

wandb_run_name=qwen-3-liar-raw-cvi-ep2
model_name_or_path=/home/comp/24483737/model/Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4
model_max_length=512
train_data_path=/home/comp/24483737/fc/data/liar-raw-post/cvi/train_liar_raw_cvi.json
eval_data_path=/home/comp/24483737/fc/data/liar-raw-post/cvi/val_liar_raw_cvi.json
output_dir=/home/comp/24483737/fc/model/qwen-3-liar-raw-cvi-ep2

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
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "epoch" \
  --save_only_model true \
  --save_strategy "no" \
  --save_total_limit 3 \
  --logging_steps 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap 'Qwen3DecoderLayer' \
