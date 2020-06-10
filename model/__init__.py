from model.discriminator import FCDiscriminator, NLayerDiscriminator
from model.model_inception import I3D
import torch
import torch.optim as optim
import pdb
def create_scheduler(args, optimizer):
    milestones = [ int(epoch) for epoch in args.milestones.split(',') ]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return scheduler

def create_scheduler_D(args, optimizer):
    milestones = [ int(epoch) for epoch in args.milestones_D.split(',') ]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return scheduler

def CreateModel(args):
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=args.set)
        if args.set == 'train':
            optimizer = optim.SGD(model.optim_parameters(args),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model

    if args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=19, init_weights=args.init_weights, restore_from=args.restore_from)
        if args.set == 'train':
            optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                 'lr': args.learning_rate * 2}
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.99))
            optimizer.zero_grad()
            return model, optimizer
        else:
            return model

    if args.model == 'I3D-inception':
        model = I3D(num_classes=args.num_classes, phase=args.set, partial_bn=False)
        # print(model)
        pretrained_dict = torch.load(args.init_weights)
        #dict_new = {}
        #for k,v in pretrained_dict.items():
        #    if k != 'features.18.weight' and k != 'features.18.bias':
        #        dict_new[k] = v
        #net_dict = model.state_dict()
        #net_dict.update(dict_new)
        #model.load_state_dict(net_dict, strict=False)
        model.load_state_dict(pretrained_dict, strict=False)
        if args.set == 'train':
            #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
            #                      lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
            scheduler = create_scheduler(args, optimizer)
            model.add_depth_module()

            optimizer.zero_grad()
            #model = torch.nn.DataParallel(model).cuda()
            model = model.cuda()
            return model, optimizer, scheduler
        else:
            model = torch.nn.DataParallel(model).cuda()
            return model


def CreateDiscriminator(args):
    #discriminator = FCDiscriminator(num_classes=args.num_classes)
    discriminator = NLayerDiscriminator(input_nc=1024)
    optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    scheduler = create_scheduler_D(args, optimizer)
    optimizer.zero_grad()
    if args.restore_from is not None:
        discriminator.load_state_dict(torch.load(args.restore_from + '_D.pth'))
    discriminator = torch.nn.DataParallel(discriminator).cuda()
    return discriminator, optimizer, scheduler


def CreateSSLModel(args):
    if args.model == 'DeepLab':
        model = Deeplab(num_classes=args.num_classes, init_weights=args.init_weights, restore_from=args.restore_from, phase=args.set)
    elif args.model == 'VGG':
        model = VGG16_FCN8s(num_classes=19, init_weights=args.init_weights, restore_from=args.restore_from)
    else:
        raise ValueError('The model mush be either deeplab-101 or vgg16-fcn')
    return model
