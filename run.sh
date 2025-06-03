#!/bin/bash
# bash run.sh /data/zhuyan/exp_log/test7 xxx/PASTIS24
# Configuration variables
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
workspace=$1
cls_log_path="${workspace}/cls_log"
cams_path="${workspace}/cams"
pseudo_labels_path="${workspace}/pseudo_labels"
seg_log_path="${workspace}/seg_log"

# xxx/PASTIS24
dataset_path=$2
sample_list="${dataset_path}/fold-paths/folds_1_123_paths.csv"


# Log function with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Step 1: Train classification model
log "Starting training classification model..."
python ./tools/train_cls.py \
    --save_path "${cls_log_path}" \
    --dataset_path "${dataset_path}" || {
    log "Error: Training classification model failed"
    exit 1
}

# Step 2: Generate CAMs
log "Generating CAMs..."
python ./tools/generate_cams.py \
    --save_path "${cls_log_path}" \
    --dataset_path "${dataset_path}" \
    --save_cams_path "${cams_path}" || {
    log "Error: CAM generation failed"
    exit 1
}

# Step 3: Evaluate CAMs and generate pseudo labels
log "Evaluating CAMs and generating pseudo labels..."
python ./tools/evaluate_cams_and_generate_pseudo_labels.py \
    --sample_list "${sample_list}" \
    --save_cams_path "${cams_path}" \
    --dataset_path "${dataset_path}" \
    --save_pseudo_label_path "${pseudo_labels_path}" || {
    log "Error: CAM evaluation failed"
    exit 1
}

# Step 4: Train segmentation model use pseudo labels
log "Training segmentation model using pseudo labels..."
python ./tools/train_seg.py \
    --save_path "${seg_log_path}" \
    --pseudo_path "${pseudo_labels_path}" \
    --dataset_path "${dataset_path}"  || {
    log "Error: Segmentation training failed"
    exit 1
}


log "All steps completed successfully"
