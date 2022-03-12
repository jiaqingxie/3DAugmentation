Batch=256
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=3 python pretrain/train-downstream_cyclegan.py \
--gnn gin \
--batch_size $Batch \
--log_dir checkpoint/cyclegan/downstream/batch$Batch-G19/ \
--checkpoint_dir checkpoint/cyclegan/downstream/batch$Batch-G19/model/ \
--save_test_dir checkpoint/cyclegan/downstream/batch$Batch-G19/testresult/ \
--pretrainmodel /remote-home/yxwang/Graph/3DInjection/checkpoint/cyclegan/batch256-G_lr-0.001-D_lr-0.001/model/checkpointG_A19.pt