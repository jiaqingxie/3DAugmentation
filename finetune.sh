Batch=256
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=4 python pretrain/train-downstream.py \
--gnn gin \
--batch_size $Batch \
--log_dir checkpoint/downstream/batch$Batch-zero/ \
--checkpoint_dir checkpoint/downstream/batch$Batch-zero/model/ \
--save_test_dir checkpoint/downstream/batch$Batch-zero/testresult/ \
--pretrainmodel zero