#!/bin/bash

CKPT_PATH=/data/cad-recruit-02_814/kilee/NextChat/save_dir/model/NextChat_epoch_2_iter_30000.pth
VAL_FILE=/data/cad-recruit-02_814/kilee/NextChat/data/REC_refcocog_umd_val.jsonl
DET_ANN_PATH=/data/datasets_802/coco/annotations/instances_val2014.json
VAL_IMG_DIR=/data/datasets_802/coco/images/train2014

python ../evaluation.py \
       --model_ckpt $CKPT_PATH \
       --refcoco_valfile $VAL_FILE \
       --task detection \
       --ann_path $DET_ANN_PATH \
       --val_img_dir $VAL_IMG_DIR \
       --batch_size 64 \
       
