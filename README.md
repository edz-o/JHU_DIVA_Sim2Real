# Requirements

- Python 3.5+
- PyTorch 1.0+

# One script to run

We provide a script to reproduce the results, see `run_code.sh`.

# Data preparation

## Download Sim-DIVA data

Sim-DIVA data download [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yzhan286_jh_edu/Ee9qNRPjdWtJmJ7M1riiv24B0QGq4o3PWLrLMok4KTfBLQ?e=hM2l6f).

Currently, we have six DIVA activities: Closing, Closing_Trunk, Entering, Exiting, Opening, Open_Trunk.

Make symlink to PWD,

```
ln -s PATH_TO_THE_UNARCHIVED_FOLDER .
```

## Crop DIVA data

We crop out the activities using groundtruth from DIVA. You need to set the `IMAGE_DIR` variable accordingly in `processing/image_crop_with_cgt.py`.
```bash
cd processing
python json_gt_to_cgt_oroginal_v2.py
python image_crop_with_cgt.py 
```

**The cropped frames can be downloaded from [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/yzhan286_jh_edu/ERJ-l7JMVSdKj4e7AFx7k5IBoi1mjqNBdo21AvqmGqQw1A?e=iKvZD9).**

Make symlink to PWD,

```
ln -s PATH_TO_THE_UNARCHIVED_FOLDER .
```

## Generate lists

**We provide examples in `name_lists/`**
The paths format should be:

```
frames_folder_path start_frame end_frame class

eg. DIVA_Union_CGT_images/training/Closing/VIRAT_S_040005_07_001026_001223_963 3011 3071 Closing
```

## Download pretrained model

I3D weights pretrained on Kinetics [here](https://www.cs.jhu.edu/~yzh/i3d_inception.pth). Put it under `pretrained/`.

Pretrained weights using synthetic data [here](https://www.cs.jhu.edu/~yzh/sim_pretrain_10000.pth).

# Instructions

This code contains two experiments: ***real***, ***real+sim***; train on diva dataset only or train on diva dataset plus synthetic dataset.

The way for domain adaptation is adversarial training.



## Train on diva dataset only

Modify configs in `train_target_only.sh`. You should put image paths in `.txt` file; modify `--train-list` and `--test-list`

Run the following e.g.,

```bash
bash train_target_only.sh exp_20200101 0,1
```


## Train on diva dataset plus synthetic dataset using adversarail training

Modify configs in `train_advt.sh`. Similarly modify `--train-list`, `--test-list` and `--sim-list`

Run the following e.g.,

```bash
bash train_advt.sh exp_20200101 0,1
```

## Test on diva dataset

Test real only model: modify configs in `test_target_only.sh` and run
Test adversarial training model: modify configs in `test_advt.sh` and run
For example,

```bash
bash test_target_only.sh exp_20200101
```
