import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import wraps
from time import time
import math
import torchvision
from projects.ora3d.models.detectors.utils.overlap_depth_loss import DisparityLoss


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GhostModule(nn.Module):
    """
        Ghost Module from https://github.com/iamhankai/ghostnet.pytorch.

    """
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.AvgPool2d(stride) if stride > 1 else nn.Sequential(),
            nn.Conv2d(inp, init_channels, kernel_size, 1, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class ResGhostModule(GhostModule):
    """Some Information about ResGhostModule"""
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, relu=True, stride=1):
        assert(ratio > 2)
        super(ResGhostModule, self).__init__(inp, oup-inp, kernel_size, ratio-1, dw_size, relu=relu, stride=stride)
        self.oup = oup
        if stride > 1:
            self.downsampling = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsampling = None

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)

        if not self.downsampling is None:
            x = self.downsampling(x)
        out = torch.cat([x, x1, x2], dim=1)
        return out[:,:self.oup,:,:]


def profile(name, profile_start=0, profile_end=1):
    def profile_wrapper(func):
        @wraps(func)
        def profiled_func(*args, **kwargs):
            # determine if started
            if profiled_func.call_time < profile_end:
                torch.cuda.synchronize()
                start_time = time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                dt = time() - start_time
                if profiled_func.call_time >= profile_start:
                    print("{} takes {} seconds at call time {}".format(name, dt, profiled_func.call_time))
                profiled_func.call_time += 1
            else:
                result = func(*args, **kwargs)
            return result

        # initilize call time to be zero
        profiled_func.call_time = 0
        return profiled_func
    return profile_wrapper


class PSMCosineModule(nn.Module):
    """Some Information about PSMCosineModule"""
    def __init__(self, max_disp=192, downsample_scale=4, input_features=512):
        super(PSMCosineModule, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)

    def forward(self, left_features, right_features):
        cost = Variable(
            torch.FloatTensor(left_features.size()[0],
                              self.depth_channel,
                              left_features.size()[2],
                              left_features.size()[3]).zero_(),
            volatile=not self.training
        ).cuda()

        for i in range(self.depth_channel):
            if i > 0:
                 cost[:, i, :,i:]  = (left_features[:,:,:,i:] * right_features[:,:,:,:-i]).mean(dim=1)
            else:
                 cost[:, i, :, :]  = (left_features * right_features).mean(dim=1)
        cost = cost.contiguous()
        del left_features, right_features
        torch.cuda.empty_cache()
        return cost


class CostVolume(nn.Module):
    """
        While PSV module define depth dimension similar to the depth in real world

        Cost Volume implementation in PSM network and its prior networks define this directly as disparity
    """
    def __init__(self, max_disp=192, downsample_scale=4, input_features=1024, PSM_features=64):
        super(CostVolume, self).__init__()
        self.max_disp = max_disp
        self.downsample_scale = downsample_scale
        self.depth_channel = int(self.max_disp / self.downsample_scale)
        self.down_sample = nn.Sequential(
            nn.Conv2d(input_features, PSM_features, 1),
            nn.BatchNorm2d(PSM_features),
            nn.ReLU(),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(2 * PSM_features, PSM_features, 3, padding=1),
            nn.BatchNorm3d(PSM_features),
            nn.ReLU(),
            nn.Conv3d(PSM_features, PSM_features, 3, padding=1),
            nn.BatchNorm3d(PSM_features),
            nn.ReLU(),
        )
        self.output_channel = PSM_features * self.depth_channel

    def forward(self, left_features, right_features):
        batch_size, _, w, h = left_features.shape
        left_features = self.down_sample(left_features)
        right_features = self.down_sample(right_features)
        cost = Variable(
            torch.FloatTensor(left_features.size()[0],
                              left_features.size()[1]*2,
                              self.depth_channel,
                              left_features.size()[2],
                              left_features.size()[3]).zero_(),
            volatile= not self.training
        ).cuda()

        for i in range(self.depth_channel):
            if i > 0 :
                 cost[:, :left_features.size()[1], i, :,i:]  = left_features[:,:,:,i:]
                 cost[:, left_features.size()[1]:, i, :,i:]  = right_features[:,:,:,:-i]
            else:
                 cost[:, :left_features.size()[1], i, :,:]   = left_features
                 cost[:, left_features.size()[1]:, i, :,:]   = right_features
        cost = cost.contiguous()
        cost = self.conv3d(cost)
        cost = cost.reshape(batch_size, -1, w, h).contiguous()
        del left_features, right_features
        torch.cuda.empty_cache()
        return cost


class CostVolumePyramid(nn.Module):
    """Some Information about CostVolumePyramid"""
    def __init__(self, depth_channel_4, depth_channel_8, depth_channel_16):
        super(CostVolumePyramid, self).__init__()
        self.depth_channel_4 = depth_channel_4  # 24
        self.depth_channel_8 = depth_channel_8  # 24
        self.depth_channel_16 = depth_channel_16  # 96

        input_features = depth_channel_4  # 24
        self.four_to_eight = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
        ).cuda()
        input_features = 3 * input_features + depth_channel_8 # 3 * 24 + 24 = 96
        self.eight_to_sixteen = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, 3, ratio=3),
            nn.AvgPool2d(2),
            BasicBlock(3 * input_features, 3 * input_features),
        ).cuda()
        input_features = 3 * input_features + depth_channel_16 # 3 * 96 + 96 = 384
        self.depth_reason = nn.Sequential(
            ResGhostModule(input_features, 3 * input_features, kernel_size=3, ratio=3),
            BasicBlock(3 * input_features, 3 * input_features),
        ).cuda()
        self.output_channel_num = 3 * input_features

        self.depth_output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(self.output_channel_num, int(self.output_channel_num/2), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/2)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(int(self.output_channel_num/2), int(self.output_channel_num/4), 3, padding=1),
            nn.BatchNorm2d(int(self.output_channel_num/4)),
            nn.ReLU(),
            nn.Conv2d(int(self.output_channel_num/4), 48, 1),
        ).cuda()

    def forward(self, psv_volume_4, psv_volume_8, psv_volume_16):
        psv_4_8 = self.four_to_eight(psv_volume_4)
        psv_volume_8 = torch.cat([psv_4_8, psv_volume_8], dim=1)
        psv_8_16 = self.eight_to_sixteen(psv_volume_8)
        psv_volume_16 = torch.cat([psv_8_16, psv_volume_16], dim=1)
        psv_16 = self.depth_reason(psv_volume_16)
        self.training = True
        if self.training:
            del psv_4_8, psv_volume_8, psv_8_16, psv_volume_16
            torch.cuda.empty_cache()
            return psv_16, self.depth_output(psv_16)

        return psv_16, torch.zeros([psv_volume_4.shape[0], 1, psv_volume_4.shape[2], psv_volume_4.shape[3]]).cuda()


class StereoMerging(nn.Module):
    def __init__(self, base_features):
        super(StereoMerging, self).__init__()
        self.cost_volume_0 = PSMCosineModule(downsample_scale=4, max_disp=48, input_features=base_features).cuda()
        PSV_depth_0 = self.cost_volume_0.depth_channel

        self.cost_volume_1 = PSMCosineModule(downsample_scale=8, max_disp=96, input_features=base_features * 2).cuda()
        PSV_depth_1 = self.cost_volume_1.depth_channel

        self.cost_volume_2 = CostVolume(downsample_scale=16, max_disp=96, input_features=base_features * 4, PSM_features=8).cuda()
        PSV_depth_2 = self.cost_volume_2.output_channel

        self.depth_reasoning = CostVolumePyramid(PSV_depth_0, PSV_depth_1, PSV_depth_2)
        self.final_channel = self.depth_reasoning.output_channel_num + base_features * 4

    def forward(self, left_x, right_x):
        PSVolume_0 = self.cost_volume_0(left_x[0], right_x[0])
        PSVolume_1 = self.cost_volume_1(left_x[1], right_x[1])
        PSVolume_2 = self.cost_volume_2(left_x[2], right_x[2])
        PSV_features, depth_output = self.depth_reasoning(PSVolume_0, PSVolume_1, PSVolume_2)
        features = torch.cat([left_x[2], PSV_features], dim=1)
        del PSVolume_0, PSVolume_1, PSVolume_2, left_x, right_x
        torch.cuda.empty_cache()
        return features, depth_output


def stereo_matching(left_features, right_features, stereo_gt, stereo_matcher, overlap_regions_pt):
    depth_loss = DisparityLoss(maxdisp=48)

    ov_pt = int(overlap_regions_pt[1]/4) if int(overlap_regions_pt[1]/4) < 80 else 80
    features, depth_output = stereo_matcher(left_features, right_features)

    depth_height, depth_width = depth_output.shape[2], depth_output.shape[3]  # 232, 80
    gt_height, gt_width = stereo_gt.shape[1], stereo_gt.shape[2]  # 928, 320

    ov_depth_output = torchvision.transforms.functional.crop(depth_output, 0, (depth_width-ov_pt), depth_height, ov_pt)
    ov_stereo_gt = torchvision.transforms.functional.crop(stereo_gt, 0, (gt_width-ov_pt*4), gt_height, ov_pt*4)
    dep_loss = depth_loss(ov_depth_output, ov_stereo_gt)

    del left_features, right_features, stereo_gt, features
    torch.cuda.empty_cache()
    return dep_loss, depth_output

