#!/bin/bash

CURRENT_DIR=`pwd`
NCCL_DEBUG=INFO
GPU_ID=0
PRETRAINED_MODEL_DIR="$CURRENT_DIR/models/pre-training/Cls"
#MODEL_PATH="$CURRENT_DIR/models/pre-training/Cls/pytorch_model.bin"  # default codet5_base + CCT5 dict + CCT5 classifier
MODEL_PATH="$CURRENT_DIR/models/codet5_base/pytorch_model.bin"   # codet5_base + codet5_base dict + CCT5 classifier
# FINETUNED_MODEL_PATH="$CURRENT_DIR/models/fine-tuning/JITDefectPrediction/SF/pytorch_model.bin"
FINETUNED_MODEL_PATH="/home/zangtao/Liang/DP/cct5/outputs/models/fine-tuning/JITDefectPrediction/SF/LApredict_preprocessed/qt/all_metrics/bs32_gas4/checkpoint-700-0.7742-0.3602-0.2857-0.4872-0.748-0.6522-0.4199/pytorch_model.bin"
EVAL_FLAG=false

usage() {
  echo "Usage: ${0} [-g] [-e]" 1>&2
  exit 1 
}

while getopts ":g:e:" opt; do
    case $opt in
        g)  GPU_ID="$OPTARG"
            ;;
        e)  MODEL_PATH="$OPTARG"
            EVAL_FLAG=true
          ;;
        \?)
          # if invalid option is provided, print error message and exit
          echo "Invalid option: -$OPTARG" >&2
          exit 1
          ;;
        :)
        # if -e flag is provided without a parameter, set eval variable to true
        EVAL_FLAG=true
        MODEL_PATH=$FINETUNED_MODEL_PATH
        ;;
    esac
done

function finetune() {
    local project=$1
    SCRIPT_PATH="src/fine_tuning/finetune_JIT_SF.py"
    if [[ $EVAL_FLAG == false ]]; then
      python $SCRIPT_PATH \
          --do_train \
          --do_test \
          --train_filename ${CURRENT_DIR}/Dataset/fine-tuning/JITDefectPrediction/LApredict_filtered_inconsistent_preprocessed/inherit_clone_with_IL_modified0/${project}/changes_train.jsonl \
          --dev_filename ${CURRENT_DIR}/Dataset/fine-tuning/JITDefectPrediction/LApredict_filtered_inconsistent_preprocessed/inherit_clone_with_IL_modified0/${project}/changes_valid.jsonl \
          --test_filename ${CURRENT_DIR}/Dataset/fine-tuning/JITDefectPrediction/LApredict_filtered_inconsistent_preprocessed/inherit_clone_with_IL_modified0/${project}/changes_test.jsonl \
          --model_type codet5_CC \
          --warmup_steps 0 \
          --learning_rate 2e-5 \
          --model_name_or_path "$CURRENT_DIR/models/codet5_base" \
          --load_model_path $MODEL_PATH \
          --output_dir ${CURRENT_DIR}/outputs/models/fine-tuning/JITDefectPrediction/SF/codet5/inherit_clone_with_IL_modified0/${project}/all_metrics/bs32_gas4_1500step \
          --train_batch_size 32 \
          --gradient_accumulation_steps 4 \
          --eval_batch_size 32 \
          --max_source_length 512 \
          --max_target_length 128 \
          --gpu_id ${GPU_ID} \
          --save_steps 100 \
          --log_steps 5 \
          --train_steps 1500 \
          --evaluate_sample_size -1
    else
      python $SCRIPT_PATH \
          --do_test \
          --train_filename ${CURRENT_DIR}/Dataset/fine-tuning/JITDefectPrediction/LApredict_preprocessed/${project}/changes_train.jsonl \
          --dev_filename ${CURRENT_DIR}/Dataset/fine-tuning/JITDefectPrediction/LApredict_preprocessed/${project}/changes_valid.jsonl \
          --test_filename ${CURRENT_DIR}/Dataset/fine-tuning/JITDefectPrediction/LApredict_preprocessed/${project}/changes_test.jsonl \
          --model_type codet5_CC \
          --warmup_steps 0 \
          --learning_rate 2e-5 \
          --model_name_or_path "$CURRENT_DIR/models/codet5_base" \
          --load_model_path $MODEL_PATH \
          --output_dir ${CURRENT_DIR}/outputs/models/fine-tuning/JITDefectPrediction/SF/LApredict_preprocessed/${project}/all_metrics/bs32_gas4 \
          --train_batch_size 32 \
          --gradient_accumulation_steps 4 \
          --eval_batch_size 32 \
          --max_source_length 512 \
          --max_target_length 128 \
          --gpu_id ${GPU_ID} \
          --save_steps 100 \
          --log_steps 5 \
          --train_steps 1500 \
          --evaluate_sample_size -1
    fi
}

#PROJECTS=("go" "openstack" "qt")
#PROJECTS=("gerrit" "go" "jdt" "openstack" "platform" "qt")
#PROJECTS=("gerrit")
PROJECTS=("go" "jdt" "openstack" "platform" "qt")


for PROJECT in "${PROJECTS[@]}"; do
    echo "Fintuning CodeT5 for project: $PROJECT"
    finetune "$PROJECT"
done


# finetune;