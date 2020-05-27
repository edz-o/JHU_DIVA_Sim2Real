import argparse
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import cv2
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames")
    parser.add_argument("--video_folder", type=str, help="Path to the folder containing videos")

    return parser.parse_args()

#cmd = 'ffmpeg -i {}/image_%05d.jpg -vcodec mpeg4 {}.mp4'
#videos = glob('/data/yzhang/20191026_diva_sim_data_processed/*')
#out_path = '/data/yzhang/20191026_diva_sim_data_video'
#videos = glob('/data/yzhang/meva_sim_data_20200427/*')
#out_path = '/data/yzhang/meva_sim_data_20200427_image'
#videos = glob('/data/yzhang/meva_sim_data_20200510/*.mp4')
#videos = [v for v in videos if '_seg.mp4' not in v]
#out_path = '/data/yzhang/meva_sim_data_20200510_image'

args = parse_args()
video_folder = osp.dirname(args.video_folder) if args.video_folder.endswith('/') else args.video_folder
videos = glob(osp.join(video_folder, '*.mp4'))
videos = [v for v in videos if '_seg.mp4' not in v]
out_path = video_folder + '_image'
os.makedirs(out_path, exist_ok=True)

def extract_frame(video_path):
    idx = 0
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):
        video_name = osp.splitext(osp.basename(video_path))[0]
        os.makedirs(osp.join(out_path, video_name), exist_ok=True)
        ret, frame = cap.read()
        if ret==True:
            cv2.imwrite(osp.join(out_path, video_name, '%05d.jpg'%idx), frame)
            idx += 1
        else:
            break
    cap.release()

def compose_video(img_dir):
    os.system(cmd.format(img_dir, osp.join(out_path, osp.basename(img_dir))))

def compose_video_resize(img_dir):
    fps = 24
    size = (360,360)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    imgs = glob(osp.join(img_dir, 'image_*.jpg'))
    imgs = sorted(imgs)
    #pdb.set_trace()
    out = cv2.VideoWriter(osp.join(out_path, osp.basename(img_dir)+'.mp4'), fourcc, fps, size)
    for img in imgs:
        img = cv2.imread(img)
        resized = cv2.resize(img, size)
        out.write(resized)
    out.release()

# Sequential program for debugging
#for vid in tqdm(videos):
#    extract_frame(vid)

with Pool(10) as p:
    r = list(tqdm(p.imap(extract_frame, videos), total=len(videos)))

