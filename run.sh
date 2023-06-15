python create_data.py --input videos\\temp --output datasets\\SUMME_HIS_32_BINS.h5 --extract_method his --bin 32
python add_user_summary.py --input datasets/SUMME_HIS_32_BINS.h5 --output datasets/user_summe_his.h5 --dataset_name summe
python create_split.py -d datasets\\SUMME_HIS_32_BINS.h5 --save-dir datasets --save-name summe_splits --num-splits 3 --train-percent 0.5
python main.py -d datasets\\user_summe_his.h5 -s datasets/summe_splits.json -vt tvsum --gpu 0 --split_id 0 --verbose --input_size 96 --save_time_training SUMME_32