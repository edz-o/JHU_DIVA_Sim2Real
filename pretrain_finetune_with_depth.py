import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_options import TrainOptions
import os
import numpy as np
from data import CreateActSrcDataLoader, CreateActTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
from torch.utils.tensorboard import SummaryWriter

import pdb
import matplotlib.pyplot as plt

torch.cuda.manual_seed_all(1234)
torch.cuda.seed_all()
np.random.seed(1234)
def main():

    opt = TrainOptions()
    args = opt.initialize()

    _t = {'iter time' : Timer()}

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs',args.exp_name))
    opt.print_options(args)

    simloader, realloader = CreateActSrcDataLoader(args), CreateActTrgDataLoader(args, 'train')
    testloader = CreateActTrgDataLoader(args, 'test')

    model, optimizer, scheduler = CreateModel(args)
    #start_iter = 0
    if args.restore_from is not None:   
        restore_point = torch.load(args.restore_from)
        model.load_state_dict(restore_point)
        print('loaded ', args.restore_from)
        


    train_writer = SummaryWriter(os.path.join('/data/tk/diva_snapshots', "logs",args.exp_name))
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()

    loss = ['loss_src', 'loss_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real'] #, 'eval_loss']
    _t['iter time'].tic()

    curr_best_val_acc = 0

    for epoch in range(args.start_epoch,args.num_epochs_stop):
      train_sim_epoch(simloader,model,optimizer,epoch,args,train_writer)
      val_acc,val_loss = val_on_real(realloader,model,epoch,args,train_writer)
      train_writer.add_scalar('pretraining_phase_real_loss', val_loss, epoch)
      train_writer.add_scalar('pretraining_phase_real_acc', val_acc, epoch)
      if val_acc >= curr_best_val_acc:
        curr_best_val_acc = val_acc
        try:
            torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, '{:d}.pth'.format(epoch) ))
        except:
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '{:d}.pth'.format(epoch)))

      scheduler.step()

            

def train_sim_epoch(loader,model,optimizer,current_epoch,args,log):
  for idx, data in enumerate(loader):
    model.train(True)
    optimizer.zero_grad()

    src_img, src_lbl, src_depth, _, _ = data
    src_depth = Variable(src_depth).cuda().float()
    src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()

    src_score, loss_src, depth_loss = model(src_img, lbl=src_lbl, depth_lbl=src_depth)

    if args.depth == 1:
      Loss = loss_src.mean() + depth_loss.mean()
    else:
      Loss = loss_src.mean()
      depth_loss = torch.zeros(2)
    Loss.backward()
    optimizer.step()
      # visualize_and_log_depth(src_img[0],trg_img[0],model,current_epoch,train_writer)
    print('[epoch %d][it %d][src loss %.4f][depth loss %.4f][lr %.4f]' % \
          (current_epoch + 1, idx + 1, loss_src.mean().data, depth_loss.mean().data,
           optimizer.param_groups[0]['lr'] * 10000))

  visualize_and_log_depth('sim_sample',src_img[0], model, current_epoch, log)
def train_sim_epoch(loader,model,optimizer,current_epoch,args,log):
  for idx, data in enumerate(loader):
    model.train(True)
    optimizer.zero_grad()

    src_img, src_lbl, src_depth, _, _ = data
    src_depth = Variable(src_depth).cuda().float()
    src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()

    src_score, loss_src, depth_loss = model(src_img, lbl=src_lbl, depth_lbl=src_depth)

    if args.depth == 1:
      Loss = loss_src.mean() + depth_loss.mean()
    else:
      Loss = loss_src.mean()
      depth_loss = torch.zeros(2)
    Loss.backward()
    optimizer.step()
      # visualize_and_log_depth(src_img[0],trg_img[0],model,current_epoch,train_writer)
    print('[epoch %d][it %d][src loss %.4f][depth loss %.4f][lr %.4f]' % \
          (current_epoch + 1, idx + 1, loss_src.mean().data, depth_loss.mean().data,
           optimizer.param_groups[0]['lr'] * 10000))

  visualize_and_log_depth('sim_sample',src_img[0], model, current_epoch, log)

def val_on_real(loader,model,current_epoch,args,log):
  val_loss = 0
  with torch.no_grad():
    conf = np.zeros((args.num_classes,args.num_classes))
    for idx, data in enumerate(loader):
      model.eval()
      src_img, src_lbl, _, _ = data
      src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()
      src_score, loss_src = model(src_img, lbl=src_lbl,test=True)
      val_loss += loss_src.mean().item()
      preds = torch.argmax(src_score,dim=1)
      for pr,gt in zip(preds,src_lbl):
        conf[pr.item(),gt.item()] += 1
      # visualize_and_log_depth(src_img[0],trg_img[0],model,current_epoch,train_writer)
      acc = (conf.diagonal() / (conf.sum(axis=1) + 0.0000001)).mean()
      print('VAL -- [epoch %d][it %d][src loss %.4f][class mean acc %.4f]' % \
            (current_epoch + 1, idx + 1, loss_src.mean().data, acc))

  epoch_acc = (conf.diagonal() / (conf.sum(axis=1) + 0.0000001)).mean()
  epoch_loss = val_loss/len(loader)
  visualize_and_log_depth('real_sample',src_img[0], model, current_epoch, log)
  return epoch_acc,epoch_loss


def visualize_and_log_depth(name,X,model,current_epoch, log):
  with torch.no_grad():
    model.eval()
    fig = plt.figure()
    _, real_depth = model(x=torch.unsqueeze(X, 0), lbl=None, depth_lbl=None, ssl=False, test=True)
    real = X.transpose(0, 1)
    real_depth = real_depth.transpose(0,1)
    count = 1
    for ind in [0,4,8,12]:
      real_rgb = real[ind]-real[ind].min()
      real_rgb /= real_rgb.max()
      real_rgb = real_rgb.cpu().transpose(0,1).transpose(1,2)

      rdepth = real_depth[ind].cpu()*255

      ax = plt.subplot(2,4,count)
      ax.imshow(real_rgb)
      ax.axis('off')

      ax = plt.subplot(2, 4, 4+count)
      ax.imshow(rdepth[0])
      ax.axis('off')
      count += 1
    log.add_figure(name,fig,current_epoch,close=True)


if __name__ == '__main__':
    main()



