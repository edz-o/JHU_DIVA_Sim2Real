# Requirements

- Python 3.5+
- PyTorch 1.0+

# Data preparation

## Download Sim-DIVA data

Sim-DIVA data download [here](https://www.cs.jhu.edu/~yzh/20191026_diva_sim_data.tar).

Currently, we have six DIVA activities: Closing, Closing_Trunk, Entering, Exiting, Opening, Open_Trunk.

## Crop DIVA data

We crop out the activities using groundtruth from DIVA. You need to set the `IMAGE_DIR` variable accordingly in `processing/image_crop_with_cgt.py`.
```bash
cd processing
python json_gt_to_cgt_oroginal_v2.py
python image_crop_with_cgt.py 
```

**The cropped frames can be downloaded from [here](https://www.cs.jhu.edu/~yzh/190207_DIVA_Union_CGT_images.tar).**

## Generate lists

**We provide examples in `name_lists/`**
The paths format should be:

```
frames_folder_path start_frame end_frame class

eg. DIVA_Union_CGT_images/training/Closing/VIRAT_S_040005_07_001026_001223_963 3011 3071 Closing
```

## Download pretrained model

I3D weights pretrained on Kinetics [here](https://www.cs.jhu.edu/~yzh/i3d_inception.pth). Put it under `pretrained/`.

# Instructions

This code contains two experiments: ***real***, ***real+sim***; train on diva dataset only or train on diva dataset plus synthetic dataset.

The way for domain adaptation is adversarial training.



## Train on diva dataset only

Modify configs in `train_target_only.sh` and run

You should put image paths in `.txt` file; modify `--train-list` and `--test-list`



## Train on diva dataset plus synthetic dataset

Modify configs in `train_ue_diva.sh` and run

Similarly modify `--train-list`, `--test-list` and `--sim-list`



## Test on diva dataset

Modify configs in `test_target_only.sh` and run
