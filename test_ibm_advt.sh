EXP=$1
for ITER in {190..200..1}
do
    model_weights=/data/tk/diva_snapshots/$EXP/$ITER.pth
    out_root=outputs/$EXP/$ITER
    batch_size=2
    python test_only.py --iter $ITER --model-weights $model_weights --batch-size $batch_size \
                                --out-root $out_root --num-workers 4 --data_root /data/diva \
                                --test-list name_lists/umd_gt_test_6class.txt | tee -a /data/tk/diva_snapshots/$EXP/test_log
done
#
