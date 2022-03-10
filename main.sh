Batch=512
Gl=0.002
Dl=0.002
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=0 python pretrain/train-gan.py \
--gnn gin \
--batch_size $Batch \
--G_lr $Gl \
--D_lr $Dl \
--log_dir checkpoint/batch$Batch-G_lr-$Gl-D_lr-$Dl/ \
--checkpoint_dir checkpoint/batch$Batch-G_lr-$Gl-D_lr-$Dl/model/ \
--save_test_dir checkpoint/batch$Batch-G_lr-$Gl-D_lr-$Dl/testresult/ 