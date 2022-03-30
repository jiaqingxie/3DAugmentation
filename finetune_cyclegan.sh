Batch=256
PretrainModel=zero
PYTHONPATH="$(pwd)":"$PYTHON_PATH" CUDA_VISIBLE_DEVICES=2 python pretrain/train-downstream_cyclegan.py \
--gnn gin \
--batch_size $Batch \
--log_dir checkpoint/cyclegan/downstream/batch$Batch-PretrainModel$PretrainModel/ \
--checkpoint_dir checkpoint/cyclegan/downstream/batch$Batch-PretrainModel$PretrainModel/model/ \
--save_test_dir checkpoint/cyclegan/downstream/batch$Batch-PretrainModel$PretrainModel/testresult/ \
--pretrainmodel $PretrainModel