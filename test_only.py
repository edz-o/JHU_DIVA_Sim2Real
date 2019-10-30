
import os
import argparse
import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
from torch.optim import lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
import pickle
from data import CreateActTrgDataLoader
from scipy.stats import mode
from torchvision import datasets, transforms
import videotransforms

from torch.autograd import Variable
from model.model_inception import I3D

classes= [
  'Closing',
  'Closing_Trunk',
  'Entering',
  'Exiting',
  'Open_Trunk',
  'Opening',
]

def parse_args():
    parser = argparse.ArgumentParser(description="test I3D")
    parser.add_argument("--iter", type=int, help='model iteration')
    parser.add_argument("--batch-size", type=int, help='batch size')
    parser.add_argument("--test-list", type=str, default='/data/wxy/diva_i3d/validate.txt', help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--model-weights", type=str, help='model iteration')
    parser.add_argument("--num-workers", type=int, default=4, help='model number workers')
    parser.add_argument("--out-root", type=str, help='model iteration')
    

    return parser.parse_args()

use_gpu = torch.cuda.is_available()

n_classes = 6

class_weights = None
if use_gpu and class_weights is not None:
  class_weights = class_weights.cuda().float()

torch.cuda.manual_seed_all(1234)
torch.cuda.seed()
np.random.seed(1234)
prop_root = '/data/tk/carhuman_near_proposals_042919/'


def get_i3d(n_class,mode='rgb', multi=False):
    i3d = I3D(n_class, phase='test', multi=multi)
    
    return i3d



def main():
  global model_weights
  global out_root
  args = parse_args()
  model_weights = args.model_weights 
  out_root = args.out_root 
  if not os.path.exists(out_root):
    os.makedirs(out_root)
  model = get_i3d(n_classes,mode='rgb')
  state_dict = torch.load(model_weights)
  model.load_state_dict(state_dict,strict=False)
  model.eval()
  print('LOADED ' , model_weights)


  if use_gpu:
    print("using gpu")

    model = model.cuda()

  sys.stdout.flush()

  test_model(args, model)


def test_model(args, model):
  test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

  val_loaders = CreateActTrgDataLoader(args, 'test')

  loaders = {
    'val':val_loaders
  }
  num_preds = np.zeros(n_classes)
  num_gts = np.zeros(n_classes)
  num_corrects = np.zeros(n_classes)
  for epoch in range(1):
    for phase in ['val']:
      model.train(False)
      running_loss = 0.0
      all_running_corrects = 0
      precisions = []
      recalls = []
      f1s = []
      # Iterate over data.
      count = 0
      for data in loaders[phase]:
        dataset_size = len(loaders[phase])

        inputs, labels, real, paths = data

        if use_gpu:
          inputs = Variable(inputs.cuda()).float()
          labels = Variable(labels.cuda()).long()
        else:
          inputs, labels = Variable(inputs).float(), Variable(labels).float()

        count += 1
        with torch.no_grad():
          vid_len = inputs.size(2)
          outputs, _, = model(inputs)

        preds = torch.max(outputs,dim=1)[1]
        for p,l in zip(preds,labels):
          num_preds[p.cpu().numpy()] += 1
          num_gts[int(l.cpu().numpy())] += 1
          num_corrects[p.item()] += (p.item() == l.item())  # Yi


        precision = np.mean([num_corrects[i]/(num_preds[i]) for i in range(n_classes) if num_preds[i]!=0])
        recall = np.mean([num_corrects[i]/(num_gts[i]) for i in range(n_classes) if num_gts[i]!=0])
        f1 = 2 * (precision * recall) / (precision + recall + 0.0001)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)


        for batch_ind in range(0,outputs.shape[0]):
          prop_id = paths[0][batch_ind].split('/')[-4]+'_' +paths[0][batch_ind].split('/')[-2].split(' ')[0] \
                    +'_'+str(classes.index(paths[0][batch_ind].split('/')[-3]))
          outs = outputs[batch_ind].cpu().numpy()
          if not os.path.exists(os.path.join(out_root,prop_id)):
            os.makedirs(os.path.join(out_root,prop_id))
          np.savetxt(os.path.join(out_root,prop_id,'out.txt'),outs)

        all_running_corrects += torch.sum(preds==labels)

        print('{:d}/{:d}:  {:s}_loss: {:.3f}, acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1_score: {:.3f} \r'.format((count * args.batch_size),
                                                                          dataset_size * args.batch_size,
                                                                          phase,
                                                                          running_loss/count,
                                                                          float(all_running_corrects) / (count * args.batch_size ),
                                                                          precisions[-1],
                                                                          recalls[-1],
                                                                          f1s[-1]),end= '\r')
        sys.stdout.flush()
      epoch_loss = running_loss / (dataset_size*args.batch_size + 0.0001)
      epoch_acc = float(all_running_corrects) / (dataset_size*args.batch_size + 0.0001)
      epoch_prec = precisions[-1]
      epoch_recall = recalls[-1]
      epoch_f1 = 2* (epoch_prec*epoch_recall)/(epoch_prec+epoch_recall+0.0001)


      print('---------  {} Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}  -----------'.format(phase, epoch_loss, epoch_acc, epoch_prec, epoch_recall, epoch_f1))


if __name__ == '__main__':
  main()

