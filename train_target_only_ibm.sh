EXP_NAME=$1
GPUs=$2

#mkdir $EXP_NAME
#cp -r name_lists $EXP_NAME
#cp -r model $EXP_NAME
#cp -r data $EXP_NAME
#cp -r utils $EXP_NAME
#cp -r options $EXP_NAME
#cp *.py $EXP_NAME
#cp *.sh $EXP_NAME
#
#cd $EXP_NAME
#ln -s `pwd`/../pretrained .


BATCH_SIZE=16
CUDA_VISIBLE_DEVICES=$GPUs python train_target_only.py --snapshot-dir ./snapshots/$EXP_NAME \
    --train-list /data/yzhang/IBM_data/umd_gt_train_6class.txt \
    --test-list /data/yzhang/IBM_data/umd_gt_test_6class.txt \
    --data_root /data/yzhang/IBM_data \
    --learning-rate 0.0001 --weight-decay 2.5e-4 --batch-size $BATCH_SIZE \
    --init-weights pretrained/i3d_inception.pth --num-classes 38 \
    --num-epochs-stop 130 --save-pred-every 1 --milestones 100,120 \
    --model I3D-inception | tee $EXP_NAME.log
