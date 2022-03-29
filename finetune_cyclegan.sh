Batch=256
PretrainModel=/remote-home/yxwang/Graph/3DInjection/checkpoint/cyclegan/batch256-G_lr-0.001-D_lr-0.001-drop_ratio-0.2/model/checkpointG_A1.pt
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=2 python pretrain/train-downstream_cyclegan.py \
--gnn gin \
--batch_size $Batch \
--log_dir checkpoint/cyclegan/downstream/batch$Batch-PretrainModel$PretrainModel/ \
--checkpoint_dir checkpoint/cyclegan/downstream/batch$Batch-PretrainModel$PretrainModel/model/ \
--save_test_dir checkpoint/cyclegan/downstream/batch$Batch-PretrainModel$PretrainModel/testresult/ \
--pretrainmodel $PretrainModel