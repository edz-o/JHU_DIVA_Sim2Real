from utils.data_utils import train_seqs, val_seqs
import os.path as osp
import os
import cv2
import json
import numpy as np

split = 'validate' # training or validate
IMAGE_DIR = "/data/diva/diva_data" # directory of the dumped DIVA images, modify this accordingly
DUMP_DIR = '190207_DIVA_Union_CGT_images/{}'
CGT_DIR = '190207_DIVA_Union_CGT'

imread = lambda img: cv2.imread(img)[:, :, ::-1]
imwrite = lambda filepath, img: cv2.imwrite(filepath, img[:, :, ::-1])

def get_image(seq, frame):
    path = osp.join(IMAGE_DIR, seq.split('_')[2][:4], seq, '%05d.png' % frame)
    if osp.isfile(path):
        image = imread(path)
        return image
    else:
        return np.array([])

def load_json(filename):
    with open(filename, 'r') as f:
        anno = json.load(f)
    return anno

def crop_via_bbox(img, bbox):
    '''bbox: x1, y1, x2, y2'''
    return img[int(bbox[1]):int(bbox[3] + 1), int(bbox[0]):int(bbox[2] + 1)]

seqs_split = {
        'training' : train_seqs,
        'validate' : val_seqs
        }
# Already expanded
expanded = False
DUMP_DIR_SPLIT = DUMP_DIR.format(split)

seqs = seqs_split[split]

ac_name_list = []
for sequence in seqs:
    folder = sequence[8:12]
    print(sequence)
    if sequence == 'VIRAT_S_000205_01_000197_000342':
        continue

    imagefolder = osp.join(IMAGE_DIR, folder, sequence)
    gt_activities = load_json('{}/{}/{}.json'.format(CGT_DIR, split, sequence))
    for activity_index in gt_activities:
        activity = gt_activities[activity_index]
        ac_name = activity[0]
        if True:
            dump_path = osp.join(DUMP_DIR_SPLIT, ac_name, sequence + '_' + activity_index)

            activty_with_T = activity[1]
            if len(activty_with_T) == 0:
                continue
            if not osp.isdir(dump_path):
                os.makedirs(dump_path)
            # split annotation file

            for activity_frame in activty_with_T:
                frame = activity_frame[0]
                bbox = activity_frame[1]
                img = get_image(sequence, frame)
                if expanded:
                    x1, y1, x2, y2 = bbox
                    x_delta = int(0.25 * abs(x2 - x1))
                    y_delta = int(0.25 * abs(y2 - y1))
                    x1_expanded = max(0, x1 - x_delta)
                    y1_expanded = max(0, y1 - y_delta)
                    x2_expanded = min(img.shape[1], x2 + x_delta)
                    y2_expanded = min(img.shape[0], y2 + y_delta)
                    bbox = [x1_expanded, y1_expanded, x2_expanded, y2_expanded]
                if img.size != 0:
                    crop_img = crop_via_bbox(img, bbox)
                    imwrite(osp.join(dump_path, '%05d.png' % frame), crop_img)

