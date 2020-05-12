EXP=$1
for ITER in {121..130..1}
do
    model_weights=snapshots/$EXP/$ITER.pth
    out_root=outputs/$EXP/$ITER
    batch_size=2
    python test_only.py --iter $ITER --model-weights $model_weights --batch-size $batch_size \
                                --out-root $out_root --num-workers 4 --data_root /data/yzhang/IBM_data \
                                --test-list /data/yzhang/IBM_data/umd_gt_test_6class.txt | tee -a snapshots/$EXP/test_log
done
#
