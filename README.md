# Depth-related instructions (TK)
## 1. Prep Dataset
Please change lines 17-20 and 39-40 accordingly in `convert_dataset.py`. Then run 
```
python convert_dataset.py
```
## 2. Compute Depth Flow
Please edit lines 32 and 33 accordingly in `utils/compute_depth_flow.py`. Then run
```
python utils/compute_depth_flow.py
```
## 3. Pretrain on simulation
In `train_ibm_depth.sh`, check whether input argument `sim-list` points to appropriate simulation name-list (eg. sim_meva_train_classes_depth_*.txt). Also make sure to change `snapshot-dir`. Then to start pre-training on sim with experiment name `NAME1` on GPU 1:
```
./train_ibm_depth.sh NAME1 1
```

## 4. Finetune on real
In `finetune_ibm_depth.sh`, check `train-list` and `test-list` name-lists. Also, change `restore-from` argument to point to appropriate pretrained model checkpoint. Then to finetune,
```
./finetune_ibm_depth.sh FINETUNE_1 1
```
## 5. Test
Edit `test_ibm_advt.sh` accordingly. To run finetuned model from the above experiment (experiment-name = FINETUNE_1), run:
```
./test_ibm_advt.sh FINETUNE_1
```



# Requirements

- Python 3.5+
- PyTorch 1.0+

# One script to run

We provide a script to reproduce the results, see `run_code.sh`.

# Data preparation

## Download Sim-DIVA data

Sim-DIVA data download [here](https://www.cs.jhu.edu/~yzh/20191026_diva_sim_data.tar).

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

**The cropped frames can be downloaded from [here](https://www.cs.jhu.edu/~yzh/190207_DIVA_Union_CGT_images.tar).**

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

Modify configs in `train_target_only_ibm.sh`. You should put image paths in `.txt` file; modify `--train-list`, `--test-list`, `--data_root`

Run the following e.g.,

```bash
bash train_target_only_ibm.sh exp_20200101 0,1
```


## Train on diva dataset plus synthetic dataset using adversarail training

Modify configs in `train_ibm_advt.sh`. Similarly modify `--train-list`, `--test-list`, `--sim-list`, `--data_root`, `--sim_data_root`

Run the following e.g.,

```bash
bash train_ibm_advt.sh exp_20200101 0,1
```

## Test on diva dataset

Test real only model: modify configs in `test_target_only_ibm.sh` and run
Test adversarial training model: modify configs in `test_ibm_advt.sh` and run
For example,

```bash
bash test_target_only_ibm.sh exp_20200101
```
