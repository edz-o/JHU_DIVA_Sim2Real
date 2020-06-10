import glob
import cv2
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
classes= {
  'Closing':6,
  'Closing_Trunk':10,
  'Entering':7,
  'Exiting':8,
  'Open_Trunk':9,
  'Opening':5
}

## RGB STUFF
root = '/data/ue_data/fix_size_depth_npy_texture_0604/imgs'
out_root = '/data/ue_data/sim_meva_20200604_6classes_rgb'
prepend = 'sim_meva_20200604_6classes_rgb'
name_list_out = 'name_lists/sim_meva_train_6classes_depth_0604.txt'
name_out = open(name_list_out,'w')
for class_name,label in classes.items():
  for cam_name in os.listdir(os.path.join(root,class_name)):
    for seq_name in os.listdir(os.path.join(root, class_name, cam_name)):
      print(class_name,cam_name,seq_name)
      rgb_frames = sorted(os.listdir(os.path.join(root, class_name, cam_name,seq_name ,'rgb')),
                          key=lambda x:int(x[:-4]))
      out_dir  = os.path.join(out_root,'{:s}_{:s}_{:s}'.format(class_name,cam_name,seq_name))
      name_out.write('{:s}/{:s}_{:s}_{:s} {:d} {:d} {:d}\n'.format(prepend,class_name,cam_name,seq_name,0,len(rgb_frames)-1,label))
      if not os.path.exists(out_dir):
        os.makedirs(out_dir)
      for i,frame in enumerate(rgb_frames):
        rgb = cv2.imread(os.path.join(root, class_name, cam_name, seq_name, 'rgb',frame))
        cv2.imwrite(os.path.join(out_dir,'image_{:05d}.jpg'.format(i)),rgb)
name_out.close()

## DEPTH CONVERSION
max_depth = 3000
root = '/data/ue_data/fix_size_depth_npy_texture_0604/imgs'
out_root = '/data/ue_data/sim_meva_20200604_6classes_depth'
dataset_len = len(glob.glob(os.path.join(root,'*','*','*','depth')))
for class_name,label in classes.items():
  for cam_name in os.listdir(os.path.join(root,class_name)):
    for seq_name in os.listdir(os.path.join(root, class_name, cam_name)):
      depth_frames = sorted(os.listdir(os.path.join(root, class_name, cam_name,seq_name ,'depth')),
                          key=lambda x:int(x[:-4]))
      out_dir  = os.path.join(out_root,'{:s}_{:s}_{:s}'.format(class_name,cam_name,seq_name))
      if not os.path.exists(out_dir):
        os.makedirs(out_dir)
      for i,frame in enumerate(depth_frames):
        depth = np.load(os.path.join(root, class_name, cam_name, seq_name, 'depth',frame))
        depth[np.where(depth>60000)] = 0
        depth  = ((depth / max_depth) * 256).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir,'depth_{:05d}.jpg'.format(i)),depth)

      print('{:d}/{:d}'.format(i,dataset_len))
