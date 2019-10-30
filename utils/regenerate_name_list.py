import os
import numpy as np
from tqdm import tqdm

def regenerate_name_list():
    train_dir = "/data/wxy/diva_i3d/training.txt"
    val_dir = "/data/wxy/diva_i3d/validate.txt"
    new_train_dir = "/data/wxy/diva_i3d/name_lists/training_loso.txt"
    new_val_dir = "/data/wxy/diva_i3d/name_lists/validate_loso.txt"

    train_ = open(train_dir, 'r')
    val_ = open(val_dir, 'r')
    train_files = train_.readlines()
    val_files = val_.readlines()
    train_.close()
    val_.close()

    train_new_file = open(new_train_dir, 'w')
    val_new_file = open(new_val_dir, 'w')

    for i, name in tqdm(enumerate(train_files)):
        scene = name.split('/')[-1].split('_')[2][:4]
        if scene != "0000":
            train_new_file.write(name)
        else:
            val_new_file.write(name)

    for i, name in tqdm(enumerate(val_files)):
        scene = name.split('/')[-1].split('_')[2][:4]
        if scene != "0000":
            train_new_file.write(name)
        else:
            val_new_file.write(name)

    train_new_file.close()
    val_new_file.close()

if __name__ == '__main__':
    regenerate_name_list()

