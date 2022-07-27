#! /bin/bash
python train.py --epochs=100 --checkpoint_dir="../results/checkpoint/" --save_test_dir="../results/test/" --log_dir="../results/log/" --drop_ratio=0.15 --virtual --residual 