#!/bin/bash

#SBATCH --qos=m
#SBATCH -p a40
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --time=6:00:00
#SBATCH --output=test_python_%j.out

MODEL_NAME='mistralai/Mistral-7B-v0.1'
DATASET='webquestions'
START=0
END=1000
NUM_RESPONSES=30
K_SHOT=32
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start) START="$2"; shift ;;
        --end) END="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
CUDA_VISIBLE_DEVICES=2 python ../sample.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --start $START \
    --end $END \
    --num_responses $NUM_RESPONSES \
    --k_shot $K_SHOT