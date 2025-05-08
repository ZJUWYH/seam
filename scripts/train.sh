#!/bin/bash
# screen -dmS train bash -c "bash scripts/train.sh"
# bash scripts/train.sh

DEFENSE_SIZE=8000  
LEARNING_RATE=2e-5   
EPSILON=0.001 # Epsilon value for the attack (perturbation radius)
BETA=0.01 # Beta value for the attack (weight of L_sd)
ALPHA=1  # Alpha value for the attack (weight of L_up)

OUTPUT_DIR="ckpt/seam_${DEFENSE_SIZE}_lr${LEARNING_RATE}_eps${EPSILON}_beta${BETA}_alpha${ALPHA}"
WANDB_RUN_NAME="seam_${DEFENSE_SIZE}_lr${LEARNING_RATE}_eps${EPSILON}_beta${BETA}_alpha${ALPHA}"


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train \
    --output_dir "$OUTPUT_DIR" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --defense_size "$DEFENSE_SIZE" \
    --epsilon "$EPSILON" 