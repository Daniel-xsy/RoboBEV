from mmdet.models.necks import FPN
from ..utils.positional_encoding import TransSinePositionalEncoding
import torch
import torch.nn as nn
from mmdet.models import NECKS
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F
import math
from ..utils.bev_aug import BEVRandomRotateScale, BEVRandomFlip
from torch.nn.modules.normalization import LayerNorm
import numpy as np

EPSILON = 1e-6

def perspective(matrix, vector):
    """Projection function for projecting 3D points to image plane. Same as OFT codes.

    Args:
        matrix (torch.Tensor): lidar2imag matrix with the shape of \
                                [B, 1, 1, 1, 3, 4].
        vector (torch.Tensor): corners 3d coords with the shape of \
                                [B, X, Y, Z, 3]. X, Y, Z represent grid_num \
                                    along x, y, z axis seperately.
    Returns:
        img_corners (torch.Tensor): corresponding 2d image corners with the \
                                        shape of [B, X, Y, Z, 2]
    """
    vector = vector.unsqueeze(-1)
    homogenous = torch.matmul(matrix[..., :-1], vector) + matrix[..., [-1]]
    homogenous = homogenous.squeeze(-1)
    return homogenous[..., :-1] / homogenous[..., [-1]]

@NECKS.register_module()
class FPN_TRANS(nn.Module):
    """FPN_TRANS as image neck for MultiCamDet.

    Args:
        topdown_layers (int): num of BasicBlock in topdown module. 
            Topdown module is used to transform 3D voxels to BEV map.
        grid_res (float): voxel size. Default: 0.5.
        pc_range (list): Point Cloud Range [x1, y1, z1, x2, y2, z2]. 
            Default: [-50, -50, -5, 50, 50, 3]. Value used in 
            nuscenes dataset in mmdet3d.
        output_size (list): output voxel map size, [Y, Z, X].
            Deafult: [128, 128, 10].
        cam_types: (list): A list of str. Represent camera order.
            Default: ['FRONT','FRONT_RIGHT','FRONT_LEFT', \
                      'BACK','BACK_LEFT','BACK_RIGHT']
            Camera order used in detr3d.
        fpn_cfg: (dict): FPN config. Please refer to mmdet.
            Default: None
    """
    def __init__(self,
                 grid_res=0.5,
                 pc_range=[-50, -50, -5, 50, 50, 3],
                 output_size=[128, 128, 10],
                 nhead=8,
                 num_encoder=4,
                 num_decoder=4,
                 num_levels=3,
                 radius_range=[1., 65., 1.],
                 scales=[1/16., 1/32., 1/64.],
                 use_different_res=False,
                 use_bev_aug=False,
                 output_multi_scale=False,
                 bev_aug_cfg=dict(rot_scale=dict(prob=0.5),
                        flip=dict(prob=0.5)),
                 cam_types=['FRONT','FRONT_RIGHT','FRONT_LEFT',
                                'BACK','BACK_LEFT','BACK_RIGHT'],
                 fpn_cfg=None):
        super(FPN_TRANS, self).__init__()
        self.fpn = FPN(**fpn_cfg)
        self.grid_res = grid_res
        self.pc_range = pc_range
        self.output_size = output_size
        self.cam_types=cam_types
        self.radius = int((radius_range[1] - radius_range[0])/radius_range[-1])
        self.radius_range=radius_range
        self.scales = scales
        self.use_different_res = use_different_res
        self.use_bev_aug = use_bev_aug
        self.output_multi_scale = output_multi_scale
        self.bev_aug = nn.ModuleList([BEVRandomRotateScale(pc_range=pc_range, **bev_aug_cfg['rot_scale']), 
                BEVRandomFlip(**bev_aug_cfg['flip'])])

        # Transformer cfg
        self.num_levels = num_levels
        self.out_channels = fpn_cfg['out_channels']
        self.pos_encoding = TransSinePositionalEncoding(int(self.out_channels/2))
        self.transformer_layers = nn.ModuleList([nn.Transformer(d_model=self.out_channels, nhead=nhead, 
                                        num_encoder_layers=num_encoder, num_decoder_layers=num_decoder) for i in range(num_levels)])


    def _forward_single_camera(self, feature_list, cam2lidar, cam_intrinsic):
        """Forward function for single_camera.

        Args:
            feature_list (list): List of torch.Tensor. 
                Contains feature maps of multiple levels.
                Each element with the shape of [B, C, H, W].
            calib (torch.Tensor): lidar2img matrix with the shape of \
                                    [B, 3, 4]
            cam_type (str): In ['FRAONT', 'FRONT_LEFT', 'FRONT_RIGHT', \
                                'BACK', 'BACK_RIGHT', 'BACK_LEFT']

        Returns:
            ortho (torch.Tensor): single camera BEV feature map with  \
                                    the shape of [B, C, H, W].
            mask (torch.Tensor): Mask indicating which region of BEV map \
                                    could be seen in the image, with the \
                                        shape of [B, 1, H, W]. Possitive value \
                                            indicates visibility.
        """

        ret_list = []
        for i in range(self.num_levels):
            scale = self.scales[i]
            feat = feature_list[i]
            B, C, H, W = feat.shape
            if self.use_different_res:
                R = int(self.radius / 2**i)
            else:
                R = self.radius
            fx, u0, v0 = cam_intrinsic[:, 0, 0], cam_intrinsic[:, 0, -1], cam_intrinsic[:, 1, -1]


            img_x_range = torch.arange(0., float(W), 1., device=feat.device)
            img_x_range = img_x_range.unsqueeze(0).repeat(B, 1)
            img_y_range = torch.arange(0., float(H), 1., device=feat.device)
            img_y_range = img_y_range.unsqueeze(0).repeat(B, 1)
            img_pos = self.pos_encoding(img_x_range, img_y_range)
            
           
            polar_x_range = torch.arange(0., float(W), 1., device=feat.device)
            polar_x_range = polar_x_range.unsqueeze(0).repeat(B, 1)
            polar_y_range = torch.arange(0., float(R), 1., device=feat.device)
            polar_y_range = polar_y_range.unsqueeze(0).repeat(B, 1)
            polar_rays_pos = self.pos_encoding(polar_x_range, polar_y_range) # [B, C*2, R, W]

            img_columns = feat + img_pos
            img_columns = img_columns.permute(2, 0, 3, 1).flatten(1, 2) # [H, B*W, C]
            polar_rays = polar_rays_pos.permute(2, 0, 3, 1).flatten(1, 2) # [R, B*W, C*2]

            bev_out = self.transformer_layers[i](img_columns, polar_rays)
            bev_out = bev_out.view(R, B, W, C).permute(1, 3, 0, 2) # [B, C, R, W]
            ret_list.append(bev_out)

        return ret_list
    

    def forward(self, feature_list, img_metas, gt_bboxes_3d, gt_labels_3d):
        """Forward function for FPN.

        Args:
            feature_list (list): List of torch.Tensor. 
                Contains feature maps of multiple levels.
                Each element with the shape of [N, C, H, W].
                N=B*num_cam.
            img_metas (List): List of dict. Each dict contains  \
                information such as gt, filename and camera matrix. 
                The lenth of equals to batch size.

        Returns:
            topdown (torch.Tensor): multi-camera BEV feature map with \
                                    the shape of [B, C, H, W].
        """
        feature_list = self.fpn(feature_list) # 

        B = len(img_metas)
        N, C, _, _ = feature_list[0].shape # N = B*num_cam
        num_cam = int(N/B)
        calib_infos = [img_meta['lidar2img'] for img_meta in img_metas]
        cam_intrinsic_infos = [img_meta['cam_intrinsic'] for img_meta in img_metas]
        cam2lidar_infos = [img_meta['cam2lidar'] for img_meta in img_metas]

        feature_list = [feat.view(B, num_cam, feat.shape[-3], feat.shape[-2], feat.shape[-1]) 
                                    for feat in feature_list]
        if not self.output_multi_scale:
            res_batch = torch.zeros(B, C, self.output_size[1], self.output_size[0]).cuda()
            mask_batch = torch.zeros(B, 1, self.output_size[1], self.output_size[0]).cuda()
        else:
            if self.use_different_res:
                res_batch = [torch.zeros(B, C, int(self.output_size[1]/2**i), int(self.output_size[0]/2**i)).cuda()
                                for i in range(self.num_levels)]
                mask_batch = [torch.zeros(B, 1, int(self.output_size[1]/2**i), int(self.output_size[0]/2**i)).cuda()
                                for i in range(self.num_levels)]
            else:
                res_batch = [torch.zeros(B, C, self.output_size[1], self.output_size[0]).cuda()
                                for i in range(self.num_levels)]
                mask_batch = [torch.zeros(B, 1, self.output_size[1], self.output_size[0]).cuda()
                                for i in range(self.num_levels)]
        
        # TODO: reduce redundant computation.
        # Process each camera seperately
        for cam_id in range(num_cam):
            cam_type = self.cam_types[cam_id]
            feature_single_cam = [feat[:,cam_id] for feat in feature_list]
            calib_info_single_cam = [torch.from_numpy(batch_cam[cam_id][:3,:]).float() 
                                            for batch_cam in calib_infos]
            cam_intrinsic_single_cam = [torch.from_numpy(batch_cam[cam_id]).float() 
                                            for batch_cam in cam_intrinsic_infos]
            cam2lidar_info_single_cam = [torch.from_numpy(batch_cam[cam_id]).float() 
                                            for batch_cam in cam2lidar_infos]

            calib_info_single_cam = torch.stack(calib_info_single_cam, dim=0).cuda()
            cam_intrinsic_single_cam = torch.stack(cam_intrinsic_single_cam, dim=0).cuda()
            cam2lidar_info_single_cam = torch.stack(cam2lidar_info_single_cam, dim=0).cuda()

            bev_feats = []
            masks = []
            ret_polar_ray_list = self._forward_single_camera(feature_single_cam, cam2lidar_info_single_cam, cam_intrinsic_single_cam)
            
            for lvl in range(self.num_levels):
                feature_size = feature_list[lvl].shape[3:]
                scale = self.scales[lvl]
                bev_feat, mask = self._interpolate_by_image_column(feature_size, scale, cam_type, ret_polar_ray_list[lvl], calib_info_single_cam, cam2lidar_info_single_cam, lvl)
                bev_feats.append(bev_feat)
                masks.append(mask.unsqueeze(1))
                # bev_feat, mask = self._interpolate_by_angle(ret_polar_ray_list[0], cam_intrinsic_single_cam, cam2lidar_info_single_cam)
                if self.output_multi_scale:
                    res_batch[lvl] += bev_feats[lvl]
                    mask_batch[lvl] += masks[lvl]
            if not self.output_multi_scale:
                res_batch += torch.sum(torch.stack(bev_feats,dim=0), dim=0)
                mask_batch[torch.sum(torch.stack(masks,dim=0), dim=0)!=0] += 1
        if not self.output_multi_scale:
            mask_batch[mask_batch==0] = 1 # avoid zero divisor
            res_batch /= mask_batch
            res_batch = [res_batch]
        else:
            for lvl in range(self.num_levels):
                mask_batch[lvl][mask_batch[lvl]==0] = 1
                res_batch[lvl] /= mask_batch[lvl]
        
        # res_batch is list
        if self.use_bev_aug:
            res_batch = self.bev_aug[0](res_batch, gt_bboxes_3d, gt_labels_3d)
            res_batch = self.bev_aug[1](res_batch, gt_bboxes_3d)
        return res_batch
    
    def _interpolate_by_image_column(self, feature_size, scale, cam_type, polar_ray, calib, cam2lidar, lvl):
        
        B = polar_ray.shape[0]
        # Generate voxel corners
        if self.output_multi_scale and self.use_different_res:
            x = torch.arange(0., 1, 1/self.output_size[0]*2**lvl) * 2 * math.pi + 1/self.output_size[0]/2
            y = torch.arange(self.radius_range[0], self.radius_range[1], self.radius_range[2]*2**lvl) + self.radius_range[2]/2 
            z = torch.arange(self.pc_range[2], self.pc_range[5], self.grid_res*2**lvl) + self.grid_res/2
        else:
            x = torch.arange(0., 1, 1/self.output_size[0]) * 2 * math.pi + 1/self.output_size[0]/2
            y = torch.arange(self.radius_range[0], self.radius_range[1], self.radius_range[2]) + self.radius_range[2]/2 
            z = torch.arange(self.pc_range[2], self.pc_range[5], self.grid_res) + self.grid_res/2
        xx, yy, zz = torch.meshgrid(x, y, z)
        corners = torch.cat([xx.unsqueeze(-1).sin()*yy.unsqueeze(-1), xx.unsqueeze(-1).cos()*yy.unsqueeze(-1), zz.unsqueeze(-1)], dim=-1)
        corners = corners.unsqueeze(0).repeat(B,1,1,1,1).cuda() # [B, X, Y, Z, 3]

        # Project to image plane
        img_corners = perspective(calib.view(-1, 1, 1, 1, 3, 4), corners)

        # normalize to [-1, 1]
        img_height, img_width = feature_size
        img_size = corners.new([img_width, img_height]) / scale
        norm_corners = (2 * img_corners / img_size - 1).clamp(-1, 1)

        sampling_pixel = norm_corners[..., 0]

        batch, width, depth, height, _ = norm_corners.size()

        cam_coords = cam2lidar[:, :2, -1]
        xx, yy = torch.meshgrid(x, y)
        grid_map = torch.stack([xx.sin()*yy, xx.cos()*yy], dim=-1).cuda()
        grid_map = grid_map.unsqueeze(0).repeat(B,1,1,1)
        grid_map -= cam_coords
        radius_map = torch.norm(grid_map, dim=-1)
        norm_radius_map = (2*(radius_map-self.radius_range[0])/self.radius -1).clamp(-1, 1)

        sample_loc = torch.stack([sampling_pixel, 
                norm_radius_map.unsqueeze(-1).repeat(1, 1, 1, sampling_pixel.shape[-1])], dim=-1)
        sample_loc = sample_loc.reshape(batch, width, depth*height, 2)


        sampling = F.grid_sample(polar_ray, sample_loc, padding_mode="border")
        sampling = sampling.reshape(batch, sampling.shape[1], width, depth, height)
        sampling = sampling.mean(-1)

        visible = ((abs(sampling_pixel) != 1).all(-1)) & (abs(norm_radius_map) != 1)

        if cam_type == 'FRONT':
            visible[:,int(width/4):int(width*3/4),:] = False
        elif cam_type == 'FRONT_RIGHT' or cam_type == 'BACK_RIGHT':
            visible[:,int(width/2):,:] = False
        elif cam_type == 'BACK_LEFT' or cam_type == 'FRONT_LEFT':
            visible[:,:int(width/2):,:] = False
        elif cam_type == 'BACK':
            visible[:,:int(width/4),:] = False
            visible[:,int(width*3/4):,:] = False

        interpolated_feat = sampling * visible
        interpolated_feat = interpolated_feat.transpose(2,3)
        visible = visible.transpose(1,2)

        return interpolated_feat, visible