EXP_NAME=$1
GPUs=$2

#mkdir $EXP_NAME
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
CUDA_VISIBLE_DEVICES=$GPUs python train_sim_target_only.py --snapshot-dir ./snapshots/ue2diva \
    --train-list name_lists/training.txt \
    --test-list name_lists/validate.txt \
    --sim-list name_lists/sim_training.txt \
    --learning-rate 0.01 --weight-decay 2.5e-4 --batch-size $BATCH_SIZE \
    --init-weights pretrained/i3d_inception.pth --num-classes 6 \
    --learning-rate-D 0.0001 --lambda-adv-target 0.001 \
    --num-steps-stop 15000 --save-pred-every 100 \
    --model I3D-inception | tee $EXP_NAME.log

