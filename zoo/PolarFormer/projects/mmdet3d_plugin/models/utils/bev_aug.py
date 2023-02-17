import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import random

class BEVRandomRotateScale(nn.Module):
    def __init__(self,
                 prob=0.5,
                 max_rotate_degree=22.5,
                 scaling_ratio_range=(0.95, 1.05),
                 pc_range=[-50., -50., -5., 50., 50., 3.]
                 ):
        super(BEVRandomRotateScale, self).__init__()
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.prob = prob
        self.max_rotate_degree = max_rotate_degree
        self.scaling_ratio_range = scaling_ratio_range
        self.pc_range = np.array(pc_range)

    def forward(self, feats, gt_bboxes_3d, gt_labels_3d):
        # B, C, H, W = feat.shape
        prob = random.uniform(0,1)
        if prob > self.prob or not self.training:
            return feats
        else:
            rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
            rotation_matrix = self._get_rotation_matrix(rotation_degree)
            scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])

            # change gt_bboxes_3d
            for batch in range(len(gt_bboxes_3d)):
                gt_bboxes_3d[batch].scale(scaling_ratio)
                gt_bboxes_3d[batch].rotate(rotation_matrix)
                gt_bboxes_3d[batch], gt_labels_3d[batch] = self.filter_gt_bboxes(gt_bboxes_3d[batch], gt_labels_3d[batch])
            # rotation_matrix = torch.from_numpy(rotation_matrix).to(feat.device)
            # scaling_matrix = torch.from_numpy(scaling_matrix).to(feat.device)
            # warp_matrix = rotation_matrix @ scaling_matrix
            # warp_matrix = warp_matrix[:2, :2]
            
            # adapt to polar feature
            rotated_feats = []
            for feat in feats:
                B, C, H, W = feat.shape
                x = torch.arange(0., W, 1.) + 0.5
                y = torch.arange(0., H, 1.) + 0.5
                x = x / W - rotation_degree/360 # inverse 
                x[x<0] += 1.
                x[x>1] -= 1.
                y = y*scaling_ratio / H

                # to [-1, 1]
                x = 2*x - 1
                y = 2*y - 1
                yy, xx = torch.meshgrid(y, x)
                norm_grids = torch.stack([xx, yy], dim=-1)
                norm_grids = norm_grids.squeeze(-1)
                norm_grids = norm_grids.unsqueeze(0).repeat(B, 1, 1, 1)
                rotated_feat = F.grid_sample(feat, norm_grids.to(feat.device), padding_mode='zeros')
                rotated_feats.append(rotated_feat)
            # # Test
            # import ipdb
            # ipdb.set_trace()
            # import cv2
            # test_img = cv2.imread('/home/yqjiang/projects/MultiCamDet/visible_region_pics_trans/test_projection_trans.png')
            # test_img = torch.from_numpy(test_img).permute(2,0,1).unsqueeze(0).float().cuda()
            # vis_pic = F.grid_sample(test_img, norm_grids.to(feat.device), padding_mode='zeros')
            # vis_pic = vis_pic.permute(0, 2, 3, 1)
            # vis_pic = vis_pic.detach().cpu().numpy()
            # cv2.imwrite('./tmp/trash/bev_rot_-10_sacle_1.00002_test.png',vis_pic[0])
            return rotated_feats

    def filter_gt_bboxes(self, gt_bboxes_3d, gt_labels_3d):
        bev_range = self.pc_range[[0, 1, 3, 4]]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask]
        return gt_bboxes_3d, gt_labels_3d
        
    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix
    
    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.],
            [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix


class BEVRandomFlip(nn.Module):
    def __init__(self, prob=0.5,):
        super(BEVRandomFlip, self).__init__()
        self.prob = prob
        
    def forward(self, feats, gt_bboxes_3d):
        prob = random.uniform(0,1)
        if prob > self.prob or not self.training:
            return feats
        else:
            h_or_v = random.uniform(0,1)
            flipped_feats = []
            if h_or_v > 0.5:
                flip_type = 'horizontal'
                for feat in feats:
                    flipped_feat = feat.new(feat.shape)
                    flipped_feat[..., :int(feat.shape[-1]/2)] = feat[..., int(feat.shape[-1]/2):]
                    flipped_feat[..., int(feat.shape[-1]/2):] = feat[..., :int(feat.shape[-1]/2)]
                    flipped_feat = flipped_feat.flip(-1) # flip horizontal = rotate 180 + flip vertical
                    flipped_feats.append(flipped_feat)
            else:
                flip_type = 'vertical'
                for feat in feats:
                    flipped_feat = feat.flip(-1) # the same
                    flipped_feats.append(flipped_feat)
            for batch in range(len(gt_bboxes_3d)):
                gt_bboxes_3d[batch].flip(flip_type)

        return flipped_feats