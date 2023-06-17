#!/bin/bash
echo "Input video test file path:"
read VIDEO_TEST

echo "Input model file path:"
read MODEL_PATH

python video_summary.py --input $VIDEO_TEST --model $MODEL_PATH --extract-method his --bin 32 --input-size 96



# python create_data.py --input videos/ --output datasets/my_dataset.h5 --extract-method his --bin 32
# python add_user_summary.py --input datasets/my_dataset.h5 --output datasets/summe_his.h5 --dataset-name summe
# python create_split.py -d datasets/summe_his.h5 --save-dir datasets --save-name summe_splits  --num-splits 5
# python main.py -d datasets/summe_his.h5 -s datasets/summe_splits.json -vt summe --gpu 0 --split-id 0 --verbose --input-size 96
# python video_summary.py --input videos/test.mp4 --model Summaries/summe/model_epoch60.pth.tar --extract-method his --bin 32 --input-size 96