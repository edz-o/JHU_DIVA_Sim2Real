from data.carhuman_proposal_dataset import DIVA_carhuman_rgb_1005
import numpy as np
from torch.utils import data
import videotransforms
from torchvision import datasets, transforms

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}

train_transforms = transforms.Compose([
    videotransforms.ResizeShortSideAndRandomCrop(226, 224),
    videotransforms.RandomHorizontalFlip(),
])

test_transforms = transforms.Compose([
    videotransforms.ResizeShortSideAndCenterCrop(226, 224),
])


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        if item[1]==-1:
            print("111")
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    print(count)
    print(N)
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def CreateActSrcDataLoader(args, mode='train'):
    if mode == 'train':
        source_dataset = DIVA_carhuman_rgb_1005 (
                        args.sim_list,
                        mode='sim',
                        transform=train_transforms,
                        data_root=args.sim_data_root,
                        depth=1
            )
    else:
        source_dataset = DIVA_carhuman_rgb_1005 (
                    args.test_list,
                    mode='sim',
                    transform=test_transforms,
                    data_root=args.sim_data_root
        )


    source_dataloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return source_dataloader


def CreateActTrgDataLoader(args, mode):
    if mode == 'train':
        source_dataset = DIVA_carhuman_rgb_1005 (
                    args.train_list,
                    split=mode,
                    mode='real',
                    transform=train_transforms,
                    data_root=args.data_root
        )
    else:
        source_dataset = DIVA_carhuman_rgb_1005 (
                    args.test_list,
                    split=mode,
                    mode='real',
                    transform=test_transforms,
                    data_root=args.data_root
        )

    source_dataloader = data.DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    return source_dataloader



