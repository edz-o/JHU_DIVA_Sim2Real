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
CUDA_VISIBLE_DEVICES=$GPUs python train_target_only.py --snapshot-dir ./snapshots/$EXP_NAME \
    --train-list name_lists/training.txt \
    --test-list name_lists/validate.txt \
    --learning-rate 0.01 --weight-decay 2.5e-4 --batch-size $BATCH_SIZE \
    --init-weights pretrained/i3d_inception.pth --num-classes 6 \
    --num-steps-stop 8000 --save-pred-every 200 \
    --model I3D-inception | tee $EXP_NAME.log
