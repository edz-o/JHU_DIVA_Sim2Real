import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_options import TrainOptions
import os
import numpy as np
from data import CreateActTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
import tensorboardX
import pdb

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

    targetloader = CreateActTrgDataLoader(args, 'train')
    targetloader_iter = iter(targetloader)
    testloader = CreateActTrgDataLoader(args, 'test')

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs"))

    bce_loss = torch.nn.BCEWithLogitsLoss()

    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    loss = ['loss_trg', 'eval_loss']
    _t['iter time'].tic()
    best_loss_eval = None
    best_step = 0
    eval_loss = np.array([0])
    for i in range(start_iter, args.num_steps):
        model.train()
        model.module.adjust_learning_rate(args, optimizer, i)

        optimizer.zero_grad()

        try:
            trg_img, trg_lbl, _, paths = next(targetloader_iter)
        except StopIteration:
            targetloader_iter = iter(targetloader)
            trg_img, trg_lbl, _, paths = next(targetloader_iter)

        trg_score, loss_trg = model(trg_img, lbl=trg_lbl)


        loss_trg = loss_trg
        loss_trg.mean().backward()


        optimizer.step()

        if (i + 1) % args.save_pred_every == 0 and i > 1000:
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
                torch.save(model.module.state_dict(), os.path.join(args.snapshot_dir, str(i+1)+'.pth' ))
                eval_loss = np.array([eval_loss])

        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_trg.mean().data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

        for m in loss:
            train_writer.add_scalar(m, eval(m).mean(), i+1)

if __name__ == '__main__':
    main()



