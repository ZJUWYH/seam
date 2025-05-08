#!/bin/bash
# screen -dmS eval bash -c "bash scripts/eval.sh"
# bash scripts/eval.sh
 
EVAL_SIZES=(1000)  
EVAL_LEARNING_RATES=(2e-5 5e-5 8e-5 1e-4 2e-4) 

OUTPUT_DIR=" " # Specify the output directory where the model is saved
for EVAL_SIZE in "${EVAL_SIZES[@]}"; do
    for EVAL_LEARNING_RATE in "${EVAL_LEARNING_RATES[@]}"; do
        WANDB_RUN_NAME="seam_eval_sample${EVAL_SIZE}_lr${EVAL_LEARNING_RATE}"
        CUDA_VISIBLE_DEVICES=0 python -m src.eval \
            --attack_size "$EVAL_SIZE" \
            --learning_rate "$EVAL_LEARNING_RATE" \
            --model_name "$OUTPUT_DIR" \
            --wandb_run_name "$WANDB_RUN_NAME"
    done
done