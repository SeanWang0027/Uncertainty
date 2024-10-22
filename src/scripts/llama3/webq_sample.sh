MODEL_NAME='google/gemma-2-9b-it'
DATASET='webquestions'
START=0
END=4000
NUM_RESPONSES=5
K_SHOT=32
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start) START="$2"; shift ;;
        --end) END="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done
CUDA_VISIBLE_DEVICES=3 python ../../sample.py \
    --model_name $MODEL_NAME \
    --dataset $DATASET \
    --start $START \
    --end $END \
    --num_responses $NUM_RESPONSES \
    --k_shot $K_SHOT