PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=4 python downstream/train-downstream.py \
--gnn gin \
--batch_size 256 \
--log_dir checkpoint/downstream/batch256/ --checkpoint_dir checkpoint/downstream/batch256/model/ --save_test_dir checkpoint/downstream/batch256/testresult/