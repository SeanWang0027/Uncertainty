#!/bin/bash

#SBATCH --qos=m
#SBATCH -p a40
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --output=test_python_%j.out

MODEL_NAME='meta-llama/Meta-Llama-3-8B'
DATASET='webquestions'
START=1000
END=4000
NUM_RESPONSES=15
K_SHOT=32
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start) START="$2"; shift ;;
        --end) END="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
CUDA_VISIBLE_DEVICES=3 python ../sample.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --start $START \
    --end $END \
    --num_responses $NUM_RESPONSES \
    --k_shot $K_SHOT