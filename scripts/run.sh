#!/bin/bash

NAME="yolov5x6"
PROJECT_NAME="protector"
python train.py --img 1280 --batch 4 --epochs 100 --data data/$PROJECT_NAME.yaml --hyp data/hyps/hyp.scratch-high.yaml --weights yolov5x6.pt --wandb_run_name $NAME --wandb_project_name $PROJECT_NAME