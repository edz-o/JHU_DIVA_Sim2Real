import pdb
import torch.nn as nn
import torch
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F

class MaskedDepthLoss(nn.Module):
  def __init__(self,mask_val=0):
    super(MaskedDepthLoss, self).__init__()
    self.mask_val = mask_val
  # masked L1 norm
  def forward(self, depth_out, depth_gt):
    loss = torch.abs(depth_gt - depth_out)
    if self.mask_val is not None:
      mask_indices=torch.where(depth_gt == self.mask_val)
      loss[mask_indices] = 0
    return loss.mean()

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            BasicConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            BasicConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            BasicConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class I3D(nn.Module):

    def __init__(self, num_classes=400, dropout_drop_prob=0.3, input_channel=3, multi=False, seg=False, \
                seg_classes=3, m_lam=0.5, s_lam=0.5, spatial_squeeze=True, partial_bn=True, phase='train'):
        super(I3D, self).__init__()
        self.phase = phase
        self.loss = None
        self.multi = multi
        self.seg = seg
        self.m_lam = m_lam
        self.s_lam = s_lam
        self.depth = 0
        # self.features = nn.Sequential(
        #     BasicConv3d(input_channel, 64, kernel_size=7, stride=(1,2,2), padding=3), # (64, 16, 112, 112)
        #     nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (64, 16, 56, 56)
        #     BasicConv3d(64, 64, kernel_size=1, stride=1), # (64, 16, 56, 56)
        #     BasicConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 16, 56, 56)
        #     nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 16, 28, 28)
        #     Mixed_3b(), # (256, 16, 28, 28)
        #     Mixed_3c(), # (480, 16, 28, 28)
        #     nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 8, 14, 14)
        #     Mixed_4b(),# (512, 8, 14, 14)
        #     Mixed_4c(),# (512, 8, 14, 14)
        #     Mixed_4d(),# (512, 8, 14, 14)
        #     Mixed_4e(),# (528, 8, 14, 14)
        #     Mixed_4f(),# (832, 8, 14, 14)
        #     nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)), # (832, 8, 7, 7)
        #     Mixed_5b(), # (832, 8, 7, 7)
        #     Mixed_5c(), # (1024, 8, 7, 7)
        #     #nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),# (1024, 8, 1, 1)
        #     nn.AdaptiveAvgPool3d((16,1,1)) # (1024, 16, 1, 1)
        # )self.depth
        self.features0 = nn.Sequential(
            BasicConv3d(input_channel, 64, kernel_size=7, stride=(1, 2, 2), padding=3),  # (64, 16, 112, 112)
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (64, 16, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1),  # (64, 16, 56, 56)
            BasicConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 16, 56, 56)
        )
        self.features1 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 16, 28, 28)
            Mixed_3b(), # (256, 16, 28, 28)
            Mixed_3c(), # (480, 16, 28, 28)
        )
        self.features2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),  # (480, 8, 14, 14)
            Mixed_4b(),  # (512, 8, 14, 14)
            Mixed_4c(),  # (512, 8, 14, 14)
            Mixed_4d(),  # (512, 8, 14, 14)
            Mixed_4e(),  # (528, 8, 14, 14)
            Mixed_4f(),  # (832, 8, 14, 14)
        )
        self.features3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),  # (832, 8, 7, 7)
            Mixed_5b(),  # (832, 8, 7, 7)
            Mixed_5c(),  # (1024, 8, 7, 7)
        )

        self.features4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),  # (192, 16, 28, 28)
            Mixed_3b(),  # (256, 16, 28, 28)
            Mixed_3c(),  # (480, 16, 28, 28)
        )
        self.features5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),  # (480, 8, 14, 14)
            Mixed_4b(),  # (512, 8, 14, 14)
            Mixed_4c(),  # (512, 8, 14, 14)
            Mixed_4d(),  # (512, 8, 14, 14)
            Mixed_4e(),  # (528, 8, 14, 14)
            Mixed_4f(),  # (832, 8, 14, 14)
        )
        self.features6 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),  # (832, 8, 7, 7)
            Mixed_5b(),  # (832, 8, 7, 7)
            Mixed_5c(),  # (1024, 8, 7, 7)
        )


        self.features_last = nn.Sequential(
            nn.AdaptiveAvgPool3d((16, 1, 1))  # (1024, 16, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout3d(dropout_drop_prob),
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),# (157, 16, 1, 1)
        )
        self.depth_loss = MaskedDepthLoss(mask_val=None).cuda()


        self.spatial_squeeze = spatial_squeeze
        #self.softmax = nn.Softmax()
        self._enable_pbn = partial_bn

        self.loss_func = nn.CrossEntropyLoss()
        if partial_bn:
            self.partialBN(True)

    def add_depth_module(self):
        self.depth = 1
        #self.toplayer = torch.nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        self.toplayer = BasicConv3d(1024, 256, kernel_size=1, stride=1)
        # Lateral layers
        self.latlayer1 = BasicConv3d(1024, 256, kernel_size=1, stride=1)
        self.latlayer2 = BasicConv3d(832, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = BasicConv3d(480, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = BasicConv3d(192, 192, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = BasicConv3d(256, 256, kernel_size=3, stride=1,padding=1)
        self.smooth2 = BasicConv3d(256, 256, kernel_size=3, stride=1,padding=1)
        self.smooth3 = BasicConv3d(256, 192, kernel_size=3, stride=1,padding=1)
        self.smooth4 = BasicConv3d(192, 192, kernel_size=3, stride=1,padding=1)
        # Depth prediction
        self.predict_depth1 = torch.nn.Sequential(
            BasicConv3d(192, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.predict_depth2 = torch.nn.Sequential(
            BasicConv3d(64, 2, kernel_size=3, stride=1, padding=1), #64 x 2 for depth-flow.
            torch.nn.ReLU(),
        )
    def forward(self, x, lbl=None, ssl=False):
        feature = self.features(x)
        logits = self.classifier(feature)

        # classification
        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)

        averaged_logits = torch.mean(logits, 2)
        if self.phase == 'train' and not ssl:
            self.loss = self.loss_func(averaged_logits, lbl)
            return feature, torch.unsqueeze(self.loss, 0)
        else:
            return averaged_logits, feature


    def forward(self, x, lbl=None, depth_lbl=None,ssl=False,test=False):
        x0 = self.features0(x)  #(192, 16, 56, 56)
        x1 = self.features1(x0) #(480, 16, 28, 28)
        x2 = self.features2(x1)  #(832, 8, 14, 14)
        x3 = self.features3(x2) # (1024, 8, 7, 7)

        p4 = self.toplayer(x3)
        p3 = self._upsample_add(p4, self.latlayer1(x3))  # 256, 8,7,7
        p3 = self.smooth1(p3)

        p2 = self._upsample_add(p3, self.latlayer2(x2))  # 256 channels, 1/8 size
        p2 = self.smooth2(p2)
        p1 = self._upsample_add(p2, self.latlayer3(x1))  # 256, 1/4 size
        p1 = self.smooth3(p1)

        p0 = self._upsample_add(p1, self.latlayer4(x0))  # 256, 1/4 size
        p0 = self.smooth4(p0)  # 256 channels, 1/4 size

        depth = self.predict_depth2(self.predict_depth1(p0)).squeeze()

        e1 = self.features4(p0)
        e2 = self.features5(e1)
        e3 = self.features6(e2)
        feature = self.features_last(e3)
        logits = self.classifier(feature)

        # classification
        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)

        averaged_logits = torch.mean(logits, 2)

        if test:
            return averaged_logits,depth

        if self.phase == 'train' and not ssl:
            self.loss = self.loss_func(averaged_logits, lbl)
            if self.depth == 1 and depth_lbl is not None:
                depth_loss = self.depth_loss(depth,depth_lbl)
                return feature,torch.unsqueeze(self.loss, 0),depth_loss
            else:
                return feature, torch.unsqueeze(self.loss, 0)
        else:
            return averaged_logits, feature



    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, T, H, W = y.size()
        return F.upsample(x, size=(T,H, W), mode='trilinear') + y

    def partialBN (self, enable):
        self._enable_pbn = enable

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     :return:
    #     """
    #     super(I3D, self).train(mode)
    #     count = 0
    #     if self._enable_pbn:
    #         # print("Freezing BatchNorm3D except the first one.")
    #         for m in self.modules():
    #             if isinstance(m, nn.BatchNorm3d):
    #                 count += 1
    #                 if count >= 2: # _enable_pbn should always be true (2 if self._enable_pbn else 1):
    #                     m.eval()
    #
    #                     # shutdown update in frozen mode
    #                     m.weight.requires_grad = False
    #                     m.bias.requires_grad = False

    def get_optim_policies(self):
        import torch.nn as nn
        #modules_skipped = (
        #    nn.ReLU,
        #    nn.MaxPool2d,
        #    nn.Dropout2d,
        #    nn.Sequential,
        #)
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]
    '''

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i)*args.batch_size / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10
    '''


    def adjust_learning_rate(self, args, optimizer, i):
        if i < 7000:
            optimizer.param_groups[0]['lr'] = args.learning_rate * (0.1**(int(i/5000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate * (0.1**(int(i/5000))) * 2
        else:
            optimizer.param_groups[0]['lr'] = args.learning_rate * (0.1**(2*int(i/5000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate * (0.1**(2*int(i/5000))) * 2

    # def adjust_learning_rate(self, args, optimizer, i):
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     lr = args.learning_rate * (0.1 ** (i // 10000))
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr


