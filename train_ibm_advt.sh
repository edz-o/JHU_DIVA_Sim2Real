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
CUDA_VISIBLE_DEVICES=$GPUs python train_sim_target_only.py --snapshot-dir ./snapshots/$EXP_NAME \
    --train-list name_lists/umd_gt_train.txt \
    --test-list name_lists/umd_gt_test.txt \
    --data_root /data/yzhang/IBM_data \
    --sim_data_root /data/yzhang/ \
    --sim-list name_lists/sim_meva_train_6class.txt \
    --learning-rate 0.0001 --weight-decay 2.5e-4 --batch-size $BATCH_SIZE \
    --init-weights pretrained/i3d_inception.pth --num-classes 38 \
    --learning-rate-D 0.0001 --lambda-adv-target 0.001 \
    --num-epochs-stop 200 --save-pred-every 1 --milestones 170,190 --milestones_D 170,190 \
    --model I3D-inception | tee $EXP_NAME.log

#--weight-decay 2.5e-4
