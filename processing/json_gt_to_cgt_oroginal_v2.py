from utils.data_utils import train_seqs, val_seqs
import os.path as osp
import os
import json
import numpy as np
import pdb

def roiImg2roiW(roi_img, W_tl):
    return np.array(roi_img) - np.hstack((W_tl, W_tl))

def save_json(anno, filename):
    with open(filename, 'w') as f:
        json.dump(anno, f, indent=2, sort_keys=True)

split = 'validate' # training or validate
GT_DIR = 'DIVA_anno_0614'
CGT_DIR = '190207_DIVA_Union_CGT'





seqs_split = {
        'training' : train_seqs,
        'validate' : val_seqs
        }

expanded = True
seqs = seqs_split[split]

for sequence in seqs:
    print(sequence)
    folder = sequence[8:12]
    if sequence == 'VIRAT_S_000205_01_000197_000342':
        continue
    if not osp.isfile('{}/{}/{}.json'.format(GT_DIR, split, sequence)):
        continue
    gt_activities = {}
    annotation = json.load(open('{}/{}/{}.json'.format(GT_DIR, split, sequence)))
    annotation_all_activities = annotation['activities']
    for activity in annotation_all_activities:
        ID = activity['activityID']
        gt_activities[ID] = []
        activity_name = activity['activity']
        gt_activities[ID].append(activity_name)
        objects = activity['objects']
        time_bbox_list = []
        time_bbox_list_with_name = []
        boxes_in_frame = {}

        boxes_in_frame_with_name = {}
        # Object -> Frame
        for obj in objects:
            obj_loc = [v for k, v in obj['localization'].items()][0]
            previous_frame = None
            previous_bbox = []
            sorted_locs = sorted(obj_loc.items(), key=lambda x: int(x[0]))
            for frame, bboxdict in sorted_locs:
                frame_int = int(frame)
                # Interpolate with previous bbox
                if previous_frame is not None and frame_int - previous_frame > 1:
                    for interpolate_frame in range(previous_frame + 1, frame_int):
                        boxes_in_frame.setdefault(interpolate_frame, []).append(previous_bbox)
                    for interpolate_frame in range(previous_frame + 1, frame_int):
                        boxes_in_frame_with_name.setdefault(interpolate_frame, {}).update(previous_bbox_dict)
                if bboxdict:
                    x1 = bboxdict['boundingBox']['x']
                    y1 = bboxdict['boundingBox']['y']
                    x2 = bboxdict['boundingBox']['x'] + bboxdict['boundingBox']['w']
                    y2 = bboxdict['boundingBox']['y'] + bboxdict['boundingBox']['h']
                    boxes_in_frame.setdefault(frame_int, []).append([x1, y1, x2, y2])
                    new_bbox_dict = {obj['objectType'] : [x1, y1, x2, y2]}
                    boxes_in_frame_with_name.setdefault(frame_int, {}).update(new_bbox_dict)
                    previous_frame = frame_int
                    previous_bbox = [x1, y1, x2, y2]
                    previous_bbox_dict = new_bbox_dict
        # Frame -> Object
        for frame in boxes_in_frame:
            if sequence[8:12] == '0002':
                image_height = 720
                image_width = 1280
            else:
                image_height = 1080
                image_width = 1920
            x1list = [x[0] for x in boxes_in_frame[frame]]
            y1list = [x[1] for x in boxes_in_frame[frame]]
            x2list = [x[2] for x in boxes_in_frame[frame]]
            y2list = [x[3] for x in boxes_in_frame[frame]]
            x1min, y1min, x2max, y2max = min(x1list), min(y1list), max(x2list), max(y2list)

            x1, y1, x2, y2 = x1min, y1min, x2max, y2max
            x_delta = int(0.15 * abs(x2 - x1))
            y_delta = int(0.15 * abs(y2 - y1))
            x1_expanded = max(0, x1 - x_delta)
            y1_expanded = max(0, y1 - y_delta)
            x2_expanded = min(image_width, x2 + x_delta)
            y2_expanded = min(image_height, y2 + y_delta)

            expanded_union_box = [x1_expanded, y1_expanded, x2_expanded, y2_expanded]
            time_bbox_list.append([frame, expanded_union_box])
            obj_bbox_in_frame = {}
            for obj_type, bbox in boxes_in_frame_with_name[frame].items():
                bbox_W_np = roiImg2roiW(bbox, expanded_union_box[:2]).astype(int)
                obj_bbox_in_frame.update({obj_type:bbox_W_np.tolist()})
            time_bbox_list_with_name.append([frame, obj_bbox_in_frame])

        sorted_time_bbox_list = sorted(time_bbox_list, key=lambda x: int(x[0]))
        sorted_time_bbox_list_with_name = sorted(time_bbox_list_with_name, key=lambda x: int(x[0]))
        gt_activities[ID].append(sorted_time_bbox_list)
    output_directory = '{}/{}'.format(CGT_DIR, split)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    save_json(gt_activities, '{}/{}/{}.json'.format(CGT_DIR, split, sequence))

