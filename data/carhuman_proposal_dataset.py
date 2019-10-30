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
        return classes.index(self._data[3].strip())

    @property
    def num_frames(self):
        return int(self._data[2])-int(self._data[1])-1

class DIVA_carhuman_rgb_1005(Dataset):
  def __init__(self, list_file,
               split='train',
               mode='real',
               activities=classes,
               transform=None,
               n_samples=16):
    self.activities = activities
    self.split=split
    self.mode=mode
    self.list_file = list_file
    self.indices = []
    self.transform = transform
    self.n_samples = n_samples
    self.stride = int(64 / self.n_samples)

    if 'real' in self.mode:
      self.style = "{:05d}"
    else:
      self.style = "{:04d}"

    self._init()

    print("{}: {} videos loaded!".format(self.mode, len(self.indices)))


  def __len__(self):
    return len(self.indices)


  def __getitem__(self, idx):
      paths, labels = self.indices[idx]
      X = []

      REAL = 'real' in self.mode

      if 'real' in self.mode and self.split == 'test':
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

      for frame in paths:
        crop = cv2.imread(frame)
        if crop is None:
            print(frame)
            pdb.set_trace()      
        X.append(crop)

      if self.transform:
        X = self.transform(X)

      return video_to_tensor(X), labels, REAL, paths

  def _parse_list(self):
    self.video_list = [VideoRecord(x.split(' ')) for x in open(self.list_file)]

  def _init(self):
    self._parse_list()
    for i in range(len(self.video_list)):
      record = self.video_list[i]
      root = record.path
      start = record.start_frames
      end = record.end_frames
      label = record.label
      img_paths = []

      for frame_ in range(start, end):
        if os.path.exists(os.path.join(root, self.style.format(frame_)+'.png')):
          img_paths.append(os.path.join(root, self.style.format(frame_)+'.png'))
          

      
      if len(img_paths) == 0 or label is None:
        continue

      self.indices.append((img_paths, label))
        


if __name__ == '__main__':
  #dump_frames()
  data = DIVA_carhuman_proposals_0429_rgb_v2('/data/tk/carhuman_near_proposals_042919/train',include_gt=True,include_sim=True)
  for dat in data:
    print()
