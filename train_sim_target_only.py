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
import tensorboardX
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
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    sourceloader, targetloader = CreateActSrcDataLoader(args), CreateActTrgDataLoader(args, 'train')
    testloader = CreateActTrgDataLoader(args, 'test')
    targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)
    model, optimizer, scheduler = CreateModel(args)
    model_D, optimizer_D, scheduler_D = CreateDiscriminator(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs"))
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    model_D.train()
    model_D.cuda()
    loss = ['loss_src', 'loss_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real'] #, 'eval_loss']
    _t['iter time'].tic()
    best_loss_eval = None
    best_step = 0
    eval_loss = np.array([0])
    current_epoch = start_iter / len(targetloader)
    for i in range(start_iter, args.num_steps):
        model.train()


        optimizer.zero_grad()
        optimizer_D.zero_grad()
        for param in model_D.parameters():
            param.requires_grad = False

        try:
            if args.depth == 1:
                src_img, src_lbl,src_depth, _, _ = next(sourceloader_iter)
                src_depth = Variable(src_depth).cuda().float()

            else:
                src_img, src_lbl,   _, _ = next(sourceloader_iter)
        except StopIteration:
            sourceloader_iter = iter(sourceloader)
            if args.depth == 1:
                src_img, src_lbl, src_depth, _, _ = next(sourceloader_iter)
                src_depth = Variable(src_depth).cuda().float()
            else:
                src_img, src_lbl, _, _ = next(sourceloader_iter)


        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda()

        if args.depth == 1:
            src_score, loss_src,depth_loss = model(src_img, lbl=src_lbl, depth_lbl=src_depth)
        else:
            src_score, loss_src = model(src_img, lbl=src_lbl)


        #coeff = 0.9 * (1000-i)/1000 + 0.1 if i < 1000 else 0.1
        #loss_src *= coeff

        Loss = loss_src.mean() + depth_loss.mean()
        Loss.backward()

        try:
            trg_img, trg_lbl, _, _ = next(targetloader_iter)
        except StopIteration:
            targetloader_iter = iter(targetloader)
            trg_img, trg_lbl, _, _ = next(targetloader_iter)
        trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
        trg_score, loss_trg = model(trg_img, lbl=trg_lbl)

        outD_trg, loss_D_trg_fake = model_D(F.softmax(trg_score, dim=1), 0)  # do not apply softmax

        loss_trg = args.lambda_adv_target * loss_D_trg_fake + loss_trg
        loss_trg.mean().backward()

        for param in model_D.parameters():
            param.requires_grad = True

        src_score, trg_score = src_score.detach(), trg_score.detach()

        outD_src, model_D_loss = model_D(F.softmax(src_score, dim=1), 0) # do not apply softmax

        loss_D_src_real = model_D_loss / 2
        loss_D_src_real.mean().backward()

        outD_trg, model_D_loss = model_D(F.softmax(trg_score, dim=1), 1) # do not apply softmax

        loss_D_trg_real = model_D_loss / 2
        loss_D_trg_real.mean().backward()


        optimizer.step()
        optimizer_D.step()


        for m in loss:
            train_writer.add_scalar(m, eval(m).mean(), i+1)

        epoch = int(i / len(targetloader))
        if epoch > current_epoch:
            current_epoch = epoch
            scheduler.step()
            scheduler_D.step()

        if (current_epoch) % args.save_pred_every == 0 and current_epoch > 10:
            try:
              torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, str(current_epoch)+'.pth' ))
            except:
              torch.save(model.state_dict(), os.path.join(args.snapshot_dir, str(current_epoch)+'.pth' ))
            if current_epoch >= args.num_epochs_stop:
                print('finish training')
                break
            ''' 
            with torch.no_grad():
                model.eval()
                eval_loss = 0
                for test_img, test_lbl, _, _ in testloader:
                    test_score, loss_test = model(test_img, lbl=test_lbl)
                    eval_loss += loss_test.mean().item() * test_img.size(0)
                eval_loss /= len(testloader.dataset)
                if best_loss_eval == None or eval_loss < best_loss_eval:
                    best_loss_eval = eval_loss
                    best_step = i + 1
                print('taking snapshot ... eval_loss: {}'.format(eval_loss))
                try:
                  torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, str(i+1)+'_.pth' ))
                except:
                  torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, str(i+1)+'_.pth' ))
                eval_loss = np.array([eval_loss])
            '''

        if (i+1) % args.print_freq == 0:
            #visualize_and_log_depth(src_img[0],trg_img[0],model,current_epoch,train_writer)
            _t['iter time'].toc(average=False)
            print('[epoch %d][it %d][src loss %.4f][depth loss %.4f][lr %.4f][%.2fs]' % \
                    (current_epoch+1, i + 1, loss_src.mean().data,depth_loss.mean().data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            _t['iter time'].tic()

def visualize_and_log_depth(sim,real,model,current_epoch, log):
  with torch.no_grad():
    model.eval()
    fig = plt.figure()

    _, sim_depth = model(x=torch.unsqueeze(sim,0),lbl=None, depth_lbl=None,ssl=False,test=True)
    _, real_depth = model(x=torch.unsqueeze(real, 0), lbl=None, depth_lbl=None, ssl=False, test=True)

    real = real.transpose(0, 1)
    sim = sim.transpose(0,1)

    count = 1
    for ind in [0,4,8,12]:
      real_rgb = real[ind]-real[ind].min()
      real_rgb /= real_rgb.max()
      real_rgb = real_rgb.cpu().transpose(0,1).transpose(1,2)
      sim_rgb = sim[ind] - sim[ind].min()
      sim_rgb /= sim_rgb.max()
      sim_rgb = sim_rgb.cpu().transpose(0, 1).transpose(1, 2)
      rdepth = real_depth[ind].cpu()*255
      sdepth = sim_depth[ind].cpu()*255

      ax = plt.subplot(4,4,count)
      ax.imshow(real_rgb)
      ax.axis('off')

      ax = plt.subplot(4, 4, 4+count)
      ax.imshow(rdepth)
      ax.axis('off')

      ax = plt.subplot(4, 4, 8 + count)
      ax.imshow(sim_rgb)
      ax.axis('off')
      ax = plt.subplot(4, 4, 12 + count)
      ax.imshow(sdepth)
      ax.axis('off')
      count += 1
    log.add_figure('depth prediction vis',fig,current_epoch,close=True)


if __name__ == '__main__':
    main()



