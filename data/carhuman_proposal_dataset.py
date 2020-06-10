import os
import json
import lmdb
import pdb
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
#from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from scipy.stats import mode
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sys
import csv
import random
from PIL import Image
from all_train_seq import train_seqs
from sklearn.preprocessing import Normalizer

random.seed(1234)
classes= [
  'Closing',
  'Closing_Trunk',
  'Entering',
  'Exiting',
  'Open_Trunk',
  'Opening',
]
six_class_label_map = {
  6:0,
  10:1,
  7:2,
  8:3,
  9:4,
  5:5
}
classes = [
        'null',
        'person_opens_facility_door',
        'person_closes_facility_door',
        'person_enters_through_structure',
        'person_exits_through_structure',
        'person_opens_vehicle_door',
        'person_closes_vehicle_door',
        'person_enters_vehicle',
        'person_exits_vehicle',
        'Open_Trunk',
        'Closing_Trunk',
        'person_loads_vehicle',
        'Unloading',
        'Talking',
        'specialized_talking_phone',
        'specialized_texting_phone',
        'Riding',
        'vehicle_turning_left',
        'vehicle_turning_right',
        'vehicle_u_turn',
        'person_sitting_down',
        'person_standing_up',
        'person_reading_document',
        'object_transfer',
        'person_picks_up_object',
        'person_sets_down_object',
        'Transport_HeavyCarry',
        'hand_interaction',
        'person_person_embrace',
        'person_purchasing',
        'person_laptop_interaction',
        'vehicle_stopping',
        'vehicle_starting',
        'vehicle_reversing',
        'vehicle_picks_up_person',
        'vehicle_drops_off_person',
        'abandon_package',
        'theft'
        ]
def get_prop_seq_name(name):
    return '_'.join(name.split('_')[:-1])


def video_to_tensor(pic):
  """Convert a ``numpy.ndarray`` to tensor.
  Converts a numpy.ndarray (T x H x W x C)
  to a torch.FloatTensor of shape (C x T x H x W)

  Args:
       pic (numpy.ndarray): Video to be converted to tensor.
  Returns:
       Tensor: Converted video.
  """


  return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0].strip()

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def end_frames(self):
        return int(self._data[2])+1

    @property
    def label(self):
        #return int(self._data[3].strip())
        return six_class_label_map[int(self._data[3].strip())]
        #return classes.index(self._data[3].strip())

    @property
    def num_frames(self):
        return int(self._data[2])-int(self._data[1])-1

class DIVA_carhuman_rgb_1005(Dataset):
  def __init__(self, list_file,
               split='train',
               mode='real',
               activities=classes,
               transform=None,
               n_samples=16,
               data_root='',
               depth=0):
    self.activities = activities
    self.split=split
    self.mode=mode
    self.list_file = list_file
    self.indices = []
    self.transform = transform
    self.n_samples = n_samples
    self.stride = int(64 / self.n_samples)
    self.data_root = data_root
    self.depth = depth
    self.depth_transforms = torchvision.transforms.Compose([
      torchvision.transforms.ToPILImage(),
      torchvision.transforms.Resize((56, 56)),
      torchvision.transforms.ToTensor()
    ])

    if 'real' in self.mode:
      self.style = "{:05d}"
    else:
      #self.style = "{:04d}"
      self.style = "image_{:05d}"

    self._init()

    print("{}: {} videos loaded!".format(self.mode, len(self.indices)))


  def __len__(self):
    return len(self.indices)


  def __getitem__(self, idx):
      if self.depth == 1:
        paths,depth_paths, labels = self.indices[idx]
      else:
        paths, labels = self.indices[idx]
      X = []
      D = []

      REAL = 'real' in self.mode

      if 'real' in self.mode and self.split == 'test':
        # Need some sliding window mechanism
        begin_index = 0
        end_index = min(64, len(paths))
      else:
        rand_end = max(0, len(paths) - 64 - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + 64, len(paths))

      out = paths[begin_index:end_index]
      if len(out) < 64:
          out = [ out[int(x)] for x in np.linspace( 0, len(out)-1, 64) ]
      paths = out[::self.stride]

      if self.depth == 1:
        out = depth_paths[begin_index:end_index]
        if len(out) < 64:
          out = [out[int(x)] for x in np.linspace(0, len(out) - 1, 64)]
        depth_paths = out[::self.stride]

      for frame in paths:
        crop = cv2.imread(frame)
        #if crop is None:
        #    print(frame)
        #    pdb.set_trace()
        X.append(crop)

      if self.depth == 1:
        for frame in depth_paths:
#          crop = cv2.imread(frame)[:,:,0]
#          crop = self.depth_transforms(crop)
#          D.append(crop)

          depth = np.load(frame)
          depth = cv2.resize(depth,(56,56))
          D.append(depth.transpose(2,0,1))


      if self.transform:
        X = self.transform(X)

      if self.depth == 1:
        return video_to_tensor(X), labels, torch.from_numpy(np.array(D)).transpose(0,1), REAL, paths
      else:
        return video_to_tensor(X), labels, REAL, paths

  def _parse_list(self):
    self.video_list = [VideoRecord(x.split(' ')) for x in open(self.list_file)]

  def _init(self):
    self._parse_list()


    for i in range(len(self.video_list)):
      record = self.video_list[i]
      root = os.path.join(self.data_root, record.path)
      start = record.start_frames
      end = record.end_frames
      label = record.label
      img_paths = []
      if self.depth == 1:
        sp = root.split('/')
        identifier = sp[-2]
        sp[-2] = identifier[:identifier.rfind('_')]+'_depth_flow_bugfix'
        depth_root = '/'
        for token in sp[1:]:
          depth_root += token
          depth_root += '/'
        depth_paths = []
      for frame_ in range(start, end):
        if os.path.exists(os.path.join(root, self.style.format(frame_)+'.jpg')):
          img_paths.append(os.path.join(root, self.style.format(frame_)+'.jpg'))
          if self.depth == 1:
            depth_paths.append(os.path.join(depth_root, 'depthflow_{:05d}'.format(frame_)+'.npy'))

      if len(img_paths) == 0 or label is None:
        continue

      if self.depth == 1:
        self.indices.append((img_paths, depth_paths,label))
      else:
        self.indices.append((img_paths, label))


if __name__ == '__main__':
  #dump_frames()
  data = DIVA_carhuman_rgb_1005('/data/tk/carhuman_near_proposals_042919/train',
                                include_gt=True,include_sim=True)

  for dat in data:
    print()
