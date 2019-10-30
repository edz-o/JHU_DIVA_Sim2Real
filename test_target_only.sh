
#for ITER in 5000 10000 15000 20000 25000 30000 35000 40000
for ITER in {100..100..1000}
do
    model_weights=20191029_test_target/snapshots/ue2diva/$ITER.pth
    out_root=outputs/20191008_target_only_$ITER
    batch_size=2
    python test_only.py --iter $ITER --model-weights $model_weights --batch-size $batch_size \
                                --out-root $out_root --num-workers 4 \
                                --test-list /data/wxy/diva_i3d/name_lists/validate.txt
done

