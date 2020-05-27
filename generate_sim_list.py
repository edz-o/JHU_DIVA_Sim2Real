import json
import os, cv2, shutil
import torch
import os.path as osp
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from PIL import Image
from tqdm import tqdm
from glob import glob

from data.carhuman_proposal_dataset import DIVA_carhuman_rgb_1005
from data.carhuman_proposal_dataset import classes, VideoRecord

labels=[
 'person_opens_vehicle_door',
 'Closing_Trunk',
 'person_exits_vehicle',
 'Open_Trunk',
 'person_closes_vehicle_door',
 'person_enters_vehicle',
]

labels=['person_opens_vehicle_door',
 'Closing_Trunk',
 'person_exits_vehicle',
 'Open_Trunk',
 'person_closes_vehicle_door',
 'person_enters_vehicle',
 'Transport_HeavyCarry',
 'person_sets_down_object',
 'person_closes_facility_door',
 'person_sitting_down',
 'person_person_embrace',
 'person_standing_up',
 'person_opens_facility_door',
 'specialized_talking_phone',
 'person_picks_up_object',
 #'object_transfer'
       ]
for l in labels:
    if l not in classes:
        print(l)

# the synthetic data used
lists = {
    'meva_sim_data_20200413_image': '/data/yzhang/meva_sim_data_20200413.json',
    'meva_sim_data_20200422_image': '/data/yzhang/meva_sim_data_20200422.json',
    'meva_sim_data_20200423_image': '/data/yzhang/meva_sim_data_20200423.json',
    'meva_sim_data_20200425_image': '/data/yzhang/meva_sim_data_20200425.json',
    'meva_sim_data_20200427_image': '/data/yzhang/meva_sim_data_20200427.json',
    'meva_sim_data_20200505_image': '/data/yzhang/meva_sim_data_20200505.json',
    'meva_sim_data_20200510_image': '/data/yzhang/meva_sim_data_20200510.json',
    # Add more here
}

data_root = '/data/yzhang'
sim_lines = []
for path, l in lists.items():
    js = json.load(open(l))

    for name, item in js['database'].items():
        name = name.replace(path[:-6]+'_', '')
        label = classes.index(item['annotations']['label'])
        if classes[label] not in labels:
            continue
        frames = glob(osp.join(data_root, path, name, '*.jpg'))
        start = 0
        end = len(frames) - 1
        vid = osp.join(path, name)

        ln = '{} {} {} {}\n'.format(vid, start, end, label)
        sim_lines.append(ln)

with open('sim_meva_train_0510_no_object_transfer1.txt', 'w') as f:
    f.writelines(sim_lines)
