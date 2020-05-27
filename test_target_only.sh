EXP=$1
GPUs=$2
for ITER in {1000..8000..200}
do
    model_weights=snapshots/$EXP/$ITER.pth
    out_root=outputs/$EXP/$ITER
    batch_size=2
    CUDA_VISIBLE_DEVICES=$GPUs python test_only.py --iter $ITER --model-weights $model_weights --batch-size $batch_size \
                                --out-root $out_root --num-workers 4 \
                                --test-list name_lists/validate.txt
done

