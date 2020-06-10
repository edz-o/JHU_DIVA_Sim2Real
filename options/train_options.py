import argparse
import os.path as osp
class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="adaptive segmentation netowork")
        parser.add_argument("--depth", type=int, default=0, help="use depth or not")
        parser.add_argument("--model", type=str, default='DeepLab',help="available options : DeepLab and VGG")
        parser.add_argument("--batch-size", type=int, default=1, help="input batch size.")
        parser.add_argument("--num-workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--train-list", type=str, default='/data/wxy/diva_i3d/name_lists/training_loso.txt', help="Path to the directory containing the source dataset.")
        parser.add_argument("--test-list", type=str, default='/data/wxy/diva_i3d/name_lists/validate_loso.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--sim-list", type=str, default='/data/wxy/diva_i3d/name_lists/sim_training.txt', help="Path to the file listing the images in the source dataset.")
        parser.add_argument("--data_root", type=str, default='', help="data root")
        parser.add_argument("--sim_data_root", type=str, default='', help="data root of simulated data")
        # parser.add_argument("--data-label-folder-target", type=str, default=None, help="Path to the soft assignments in the target dataset.")
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--learning-rate-D", type=float, default=1e-4, help="initial learning rate for discriminator.")
        parser.add_argument('--milestones', default='40,80,100', type=str, help='Learning rate scheduling, when to multiply learning rate by gamma')
        parser.add_argument('--milestones_D', default='40,80,100', type=str, help='Learning rate scheduling for discriminator, when to multiply learning rate by gamma')
        parser.add_argument("--lambda-adv-target", type=float, default=0.001, help="lambda_adv for adversarial training.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--num-steps", type=int, default=250000, help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=120000, help="Number of training steps for early stopping.")
        parser.add_argument("--num-epochs-stop", type=int, default=130, help="Number of training steps for early stopping.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")
        parser.add_argument("--save-pred-every", type=int, default=10000, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-freq", type=int, default=10, help="print loss and time fequency.")
        parser.add_argument("--snapshot-dir", type=str, default='/path/to/snapshots/', help="Where to save snapshots of the model.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
        parser.add_argument("--exp-name", type=str, default='', help="initial model.")
        parser.add_argument("--start-epoch", type=int, default=0, help="initial model.")
        

        return parser.parse_args()

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')

