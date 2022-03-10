Batch=512
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=0 python pretrain/train-gan.py \
--gnn gin \
--batch_size $Batch \
--log_dir checkpoint/downstream/batch$Batch/ \
--checkpoint_dir checkpoint/downstream/batch$Batch/model/ \
--save_test_dir checkpoint/downstream/batch$Batch/testresult/ 