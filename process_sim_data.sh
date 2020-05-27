Folder=/data/yzhang/meva_sim_data_20200521

python extract_frames.py --video_folder $Folder
python generate_sim_json.py --video_folder $Folder
