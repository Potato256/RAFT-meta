#!/bin/bash
mkdir -p checkpoints
python -u train.py --name meta_finetune --stage meta --validation meta --gpus 0 --num_steps 50000 \
    --batch_size 2 --lr 0.00025 --image_size 1024 768 --wdecay 0.0001 --mixed_precision \
    --dataset /home/lbx/code/metasurface-depth/hyperSim/hypersim-1k \
    --eval_freq 4 --cosine_loss_weight 0.2 #\
    # --fine_tune --pretrained /home/lbx/code/RAFT-meta/models/raft-sintel.pth
