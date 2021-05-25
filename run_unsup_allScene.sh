#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
export CUDA_VISIBLE_DEVICES=1
nohup python train.py \
    --model_name_or_path /search/odin/guobk/data/model/bert-base-chinese/ \
    --train_file /search/odin/guobk/data/simcse/pretrainData.txt \
    --output_dir /search/odin/guobk/data/simcse/simcse_roberta_zh_l12 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 48 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 >> log/train.log 2>&1 &
