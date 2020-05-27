from glob import glob
import json
import os.path as osp
import argparse

mapping = {
            'Opening' : 'person_opens_vehicle_door',
            'Closing_Trunk' : 'Closing_Trunk',
            'Exiting' : 'person_exits_vehicle',
            'Open_Trunk' : 'Open_Trunk',
            'Closing' : 'person_closes_vehicle_door',
            'Entering' : 'person_enters_vehicle',
            'person_carries_heavy_object' : "Transport_HeavyCarry",
            'person_puts_down_object' : 'person_sets_down_object',
            'person_closes_facility_door' : 'person_closes_facility_door',
            'person_sits_down' : 'person_sitting_down',
            'person_embraces_person' : 'person_person_embrace',
            'person_stands_up' : 'person_standing_up',
            'person_opens_facility_door' : 'person_opens_facility_door',
            'person_talks_on_phone' : 'specialized_talking_phone',
            'person_picks_up_object' : 'person_picks_up_object',
            'person_transfers_object' : 'object_transfer',
}

json_sim = {
  "labels": [
    "Closing_Trunk",
    "Open_Trunk",
    "Riding",
    "Talking",
    "Transport_HeavyCarry",
    "abandon_package",
    "hand_interaction",
    "object_transfer",
    "person_closes_facility_door",
    "person_closes_vehicle_door",
    "person_enters_through_structure",
    "person_enters_vehicle",
    "person_exits_through_structure",
    "person_exits_vehicle",
    "person_laptop_interaction",
    "person_loads_vehicle",
    "person_opens_facility_door",
    "person_opens_vehicle_door",
    "person_person_embrace",
    "person_picks_up_object",
    "person_purchasing",
    "person_reading_document",
    "person_sets_down_object",
    "person_sitting_down",
    "person_standing_up",
    "person_unloads_vehicle",
    "specialized_talking_phone",
    "specialized_texting_phone",
    "vehicle_drops_off_person",
    "vehicle_picks_up_person",
    "vehicle_reversing",
    "vehicle_starting",
    "vehicle_stopping",
    "vehicle_turning_left",
    "vehicle_turning_right",
    "vehicle_u_turn",
    "heu_negative"
  ],
  "database": {

}}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate json file for sim data")
    parser.add_argument("--video_folder", type=str, help="Path to the folder containing videos")

    return parser.parse_args()

#videos = glob('/data/yzhang/20191026_diva_sim_data_processed/*')
#videos = glob('/data/yzhang/meva_sim_data_20200313_processed/*')
#prefix = '20191026_diva_sim_data'
#prefix = 'meva_sim_data_20200313'
#prefix = 'meva_sim_data_20200315'
#prefix = 'meva_sim_data_20200320'
#prefix = 'meva_sim_data_20200413'
#prefix = 'meva_sim_data_20200427'
#prefix = 'meva_sim_data_20200505'
#prefix = 'meva_sim_data_20200510'
#videos = glob('/data/yzhang/%s_processed/*'%prefix)
#videos = glob('/data/yzhang/%s_flow/*'%prefix)
#videos = glob('/data/yzhang/%s/*.mp4'%prefix)
#videos = [v for v in videos if '_seg.mp4' not in v]


args = parse_args()
video_folder = osp.dirname(args.video_folder) if args.video_folder.endswith('/') else args.video_folder
prefix = osp.basename(video_folder)
videos = glob(osp.join(video_folder, '*.mp4'))
videos = [v for v in videos if '_seg.mp4' not in v]

for v in videos:
    name = osp.basename(osp.splitext(v)[0])
    #if name in exclude_list:
    #    continue
    #class_name = '_'.join(name.split('_')[:-4])
    class_name = '_'.join(name.split('_')[:-2])
    content =    {
      "subset": "training",
      "annotations": {
        "label": mapping[class_name]
      }
    }
    json_sim["database"][prefix+'_'+name] = content

with open(video_folder+'.json', 'w') as f:
    json.dump(json_sim, f, indent=4)
