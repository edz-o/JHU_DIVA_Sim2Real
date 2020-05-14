## real only
#for rep in {1..5}
#do
#    EXP=ibm_umd_gt_real_$rep
#    bash train_target_only_ibm.sh $EXP 0,1,2,3
#    bash test_target_only_ibm.sh $EXP
#done

# real + sim
for rep in {4..5}
do
    EXP=ibm_umd_gt_advt_$rep
    bash train_ibm_advt.sh $EXP 0,1,2,3
    bash test_ibm_advt.sh $EXP
done
