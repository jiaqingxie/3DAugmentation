Batch=256
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=2 python pretrain/train-downstream.py \
--gnn gin \
--batch_size $Batch \
--log_dir checkpoint/downstream/batch$Batch/ \
--checkpoint_dir checkpoint/downstream/batch$Batch/model/ \
--save_test_dir checkpoint/downstream/batch$Batch/testresult/ \
--pretrainmodel /remote-home/yxwang/Graph/3DInjection/checkpoint/batch256/lr/model/checkpointG3.pt