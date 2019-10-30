# Instructions

This code contains two experiments: ***real***, ***real+sim***; train on diva dataset only or train on diva dataset plus synthetic dataset.

The way for domain adaptation is adversarial training.



## Train on diva dataset only

Modify configs in `train_target_only.sh` and run

You should put image paths in `.txt` file; modify `--train-list` and `--test-list`

The paths format should be:

```
frames_folder_path start_frame end_frame class

eg. DIVA_Union_CGT_images/training/Closing/VIRAT_S_040005_07_001026_001223_963 3011 3071 Closing
```



## Train on diva dataset plus synthetic dataset

Modify configs in `train_ue_diva.sh` and run

Similarly modify `--train-list`, `--test-list` and `--sim-list`



## Test on diva dataset

Modify configs in `test_target_only.sh` and run