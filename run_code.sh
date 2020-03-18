wget https://www.cs.jhu.edu/~yzh/20191026_diva_sim_data.tar
wget https://www.cs.jhu.edu/~yzh/190207_DIVA_Union_CGT_images.tar
wget https://www.cs.jhu.edu/~yzh/i3d_inception.pth

mkdir pretrained
mv i3d_inception.pth pretrained

tar -xf 20191026_diva_sim_data.tar
tar -xf 190207_DIVA_Union_CGT_images.tar

DATE=$(date '+%Y-%m-%d_%H:%M:%S')
EXP=exp_advT_$DATE
bash train_advt.sh $EXP 0,1
bash test_advt.sh $EXP


DATE=$(date '+%Y-%m-%d_%H:%M:%S')
EXP=exp_realonly_$DATE
bash train_target_only.sh $EXP 0,1
bash test_target_only.sh $EXP
