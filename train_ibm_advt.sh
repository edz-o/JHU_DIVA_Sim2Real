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


BATCH_SIZE=8
CUDA_VISIBLE_DEVICES=$GPUs python train_sim_target_only.py --snapshot-dir ./snapshots/$EXP_NAME \
    --train-list name_lists/umd_gt_train_6class.txt \
    --test-list name_lists/umd_gt_test_6class.txt \
    --data_root /home/tk/dev/data/DIVA \
    --sim_data_root /home/tk/dev/data/ue_data \
    --sim-list name_lists/sim_meva_train_6classes_depth.txt \
    --learning-rate 0.0001 --weight-decay 2.5e-4 --batch-size $BATCH_SIZE \
    --init-weights snapshots/tk_depth_2/134.pth  --num-classes 6 \
    --learning-rate-D 0.0001 --lambda-adv-target 0.001 \
    --num-epochs-stop 200 --save-pred-every 2 --milestones 170,190 --milestones_D 170,190 \
    --model I3D-inception --depth 1 | tee $EXP_NAME.log

#--weight-decay 2.5e-4
