Batch=256
Gl=0.001
Dl=0.001
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=5 python pretrain/train-cyclegan.py \
--gnn gin \
--batch_size $Batch \
--G_lr $Gl \
--D_lr $Dl \
--log_dir checkpoint/cyclegan/batch$Batch-G_lr-$Gl-D_lr-$Dl/ \
--checkpoint_dir checkpoint/cyclegan/batch$Batch-G_lr-$Gl-D_lr-$Dl/model/ \
--save_test_dir checkpoint/cyclegan/batch$Batch-G_lr-$Gl-D_lr-$Dl/testresult/ 