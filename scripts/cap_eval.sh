#!/bin/bash

CKPT_PATH=/data/cad-recruit-02_814/kilee/NextChat/save_dir/model/NextChat_epoch_2_iter_30000.pth
VAL_IMG_DIR=/data/datasets_802/coco/images/val2017
CAPTION_RESULT_DIR=/data/cad-recruit-02_814/kilee/NextChat/temporal_results/caption_dir
CAPTION_ANN_PATH=/data/datasets_802/coco/annotations/captions_val2017.json

python ../evaluation.py \
       --model_ckpt $CKPT_PATH \
       --val_img_dir $VAL_IMG_DIR \
       --task caption \
       --result_dir $CAPTION_RESULT_DIR \
       --ann_path $CAPTION_ANN_PATH \
       --max_new_tkn 20 \
       --batch_size 64
