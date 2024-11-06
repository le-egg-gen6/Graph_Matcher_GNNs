#!/bin/bash

python run.py \
  --train_data_file dataset/train_100.txt \
  --do_train \
  --output_dir saved_training_model
