## real only
#for rep in {1..5}
#do
#    EXP=ibm_umd_gt_real_$rep
#    #bash train_target_only_ibm.sh $EXP 0,1,2,3
#    bash test_target_only_ibm.sh $EXP 0
#done

# real + sim
for rep in {1..5}
do
    EXP=ibm_umd_gt_advt_annealing_$rep
    bash train_ibm_advt.sh $EXP 0,1,2,3
    bash test_ibm_advt.sh $EXP 0
done

## final finetune
#for rep in {2..5}
#do
#    EXP=ibm_umd_gt_advt_${rep}_ft
#    WEIGHTS=snapshots/ibm_umd_gt_advt_${rep}/190.pth
#    bash finetune_target_ibm.sh $EXP 0,1,2,3 $WEIGHTS
#done
#
#for rep in {1..5}
#do
#    EXP=ibm_umd_gt_advt_${rep}_ft
#    bash test_finetune_ibm.sh $EXP 0
#    #bash test_target_only_ibm.sh $EXP $(($rep-2)) &
#done
#
#wait
#echo "Finished Testing"
