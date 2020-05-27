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

This code contains three experiments: ***real***, ***real+sim adversarial training*** and ***pretrain on sim then finetune on real***; train on diva dataset only or train on diva dataset plus synthetic dataset.

The way for domain adaptation is adversarial training or finetuning.

## (NEW) Process new synthetic data

Use the script `process_sim_data.sh` to process newly generated synthetic videos. Set `Folder` to be the path to the video folder. At the same directory, the RGB frames will be extracted to a folder ends with `_image` and a JSON annotation file will be generated.

Use the script `generate_sim_list.py` to generate the name list of synthetic data. Modify the dict `list` to include the synthetic data you want to use.

## (NEW) Pretrain on synthetic data

Use `train_source_ibm.sh`. Set `--train-list` to be the list of synthetic data you want to pretrain on and `--data_root` to be the root directory of the data folder. 
```bash
bash train_source_ibm.sh EXP_NAME 0,1,2,3
```

## (NEW) Finetuning a pretrained network on MEVA

Use `finetune_target_ibm.sh`. Set `--train-list` to be the list of real data you want to finetune on and `--data_root` to be the root directory of the data folder. 
```bash
bash train_source_ibm.sh EXP_NAME 0,1,2,3
```
Testing is using the same script `test_target_only_ibm.sh` as in [Test on diva dataset](#test-on-diva-dataset).

## Train on meva dataset only

Modify configs in `train_target_only_ibm.sh`. You should put image paths in `.txt` file; modify `--train-list`, `--test-list`, `--data_root`

Run the following e.g.,

```bash
bash train_target_only_ibm.sh exp_20200101 0,1
```


## Train on meva dataset plus synthetic dataset using adversarail training

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
