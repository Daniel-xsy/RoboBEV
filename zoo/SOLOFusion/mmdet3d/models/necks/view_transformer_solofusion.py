
import os
import copy
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from torch.nn.utils.rnn import pad_sequence
from scipy.ndimage import gaussian_filter1d

from mmcv.runner import BaseModule, force_fp32, auto_fp16
from mmdet3d.ops import bev_pool
from mmcv.cnn import build_conv_layer
from mmcv.cnn import ConvModule

from ..builder import NECKS
from .. import builder
from .view_transformer import ViewTransformerLiftSplatShoot, SELikeModule
from ..detectors.solofusion import generate_forward_transformation_matrix


def finite_check(x, s="", pdb_save=True):
    # return 
    if int(os.environ.get("DEBUG", "0")) == "1":
        if pdb_save:
            try:
                assert (~torch.isfinite(x)).sum() == 0, "{}: {}, {}".format(s, x.min(), x.max())
            except: breakpoint()
        else:
            assert (~torch.isfinite(x)).sum() == 0, "{}: {}, {}".format(s, x.min(), x.max())

def interp_zeroends(x, xp, fp):
    """
    For convenience, assumes the sampling dimension is 0.
    This also fills in the ends with 0

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    assert len(x.shape) == len(xp.shape)
    assert xp.shape == fp.shape

    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    m = torch.cat([m.new_zeros((1, *m.shape[1:])), m, m.new_zeros((1, *m.shape[1:]))], dim=0)
    b = torch.cat([b.new_zeros((1, *b.shape[1:])), b, b.new_zeros((1, *b.shape[1:]))], dim=0)

    indicies = torch.sum(torch.ge(x.unsqueeze(1), xp.unsqueeze(0)), dim=1).long()

    res = torch.gather(m, dim=0, index=indicies) * x + torch.gather(b, dim=0, index=indicies)
    res.scatter_(dim=0, index=xp[[-1]].long(), src=fp[[-1]]) 
    
    return res

@NECKS.register_module()
class ViewTransformerSOLOFusion(ViewTransformerLiftSplatShoot):
    def __init__(self, 
                 extra_depth_net, 
                 loss_depth_weight, 
                 se_config=dict(), 

                 do_history_stereo_fusion=False,
                 stereo_downsample=4, 
                 stereo_group_num=8,
                 stereo_sampling_num=7,

                 stereo_gauss_bin_stdev=2,
                 stereo_spread_before_add_type=None,
                 
                 **kwargs):
        super(ViewTransformerSOLOFusion, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.extra_depthnet = builder.build_backbone(extra_depth_net)
        self.featnet = nn.Conv2d(self.numC_input,
                                 self.numC_Trans,
                                 kernel_size=1,
                                 padding=0)
        self.depthnet = nn.Conv2d(extra_depth_net['num_channels'][0],
                                  self.D,
                                  kernel_size=1,
                                  padding=0)
        self.dcn = nn.Sequential(*[build_conv_layer(dict(type='DCNv2',
                                                        deform_groups=1),
                                                   extra_depth_net['num_channels'][0],
                                                   extra_depth_net['num_channels'][0],
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   bias=False),
                                   nn.BatchNorm2d(extra_depth_net['num_channels'][0])
                                  ])
        self.se = SELikeModule(self.numC_input,
                               feat_channel=extra_depth_net['num_channels'][0],
                               **se_config)

        self.do_history_stereo_fusion = do_history_stereo_fusion
        if self.do_history_stereo_fusion:
            self.stereo_group_num = stereo_group_num
            self.similarity_net = nn.Sequential(
                ConvModule(in_channels=self.stereo_group_num,
                            out_channels=16,
                            kernel_size=1,
                            stride=(1, 1, 1),
                            padding=0,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=dict(type='BN3d'),
                            act_cfg=dict(type='ReLU', inplace=True)),
                ConvModule(in_channels=16,
                            out_channels=8,
                            kernel_size=1,
                            stride=(1, 1, 1),
                            padding=0,
                            conv_cfg=dict(type='Conv3d'),
                            norm_cfg=dict(type='BN3d'),
                            act_cfg=dict(type='ReLU', inplace=True)),
                nn.Conv3d(in_channels=8,
                        out_channels=1,
                        kernel_size=1,
                        stride=1,
                        padding=0))

            self.stereo_eps = 1e-5
            self.stereo_downsample = stereo_downsample
            self.stereo_sampling_num = stereo_sampling_num

            # Setup gaussian sampling
            gaussians = torch.from_numpy(gaussian_filter1d(F.one_hot(torch.arange(self.D)).float().numpy(), stereo_gauss_bin_stdev, mode='constant', cval=0))
            gaussians = gaussians / gaussians.max()
            inv_gaussians = 1 - gaussians
            log_inv_gaussians = torch.log(inv_gaussians + self.stereo_eps)
            log_inv_gaussians[torch.arange(len(log_inv_gaussians)), torch.arange(len(log_inv_gaussians))] = -1000
            self.log_inv_gaussians = nn.Parameter(log_inv_gaussians, requires_grad=False)
            self.bin_centers = nn.Parameter(self.get_bin_centers(), requires_grad=False)

        self.fp16_enabled = False


    def get_bin_centers(self):
        depth_bins = torch.arange(self.grid_config['dbound'][0], 
                                    self.grid_config['dbound'][1], 
                                    self.grid_config['dbound'][2]) # (118, )
        depth_bins = depth_bins + self.grid_config['dbound'][2] / 2 # center them
        assert len(depth_bins) == self.D
        return depth_bins
    
    @force_fp32(apply_to=('curr_global2img', 'curr_img_forward_aug', 'prev_global2img', 'prev_img_forward_aug', 'curr_unaug_cam_to_prev_unaug_cam'))
    def get_prev_meshgrid_sampling_points(self, 
                                          depth_bins_to_sample,
                                          curr_global2img, 
                                          curr_img_forward_aug, 
                                          prev_global2img, 
                                          prev_img_forward_aug,
                                          curr_unaug_cam_to_prev_unaug_cam):
        B, N, _, stereo_H, stereo_W = depth_bins_to_sample.shape
        eps = self.stereo_eps

        ### Sample Stereo feats from prev
        ## First, transform curr stereo meshgrid to global        
        meshgrid = torch.stack(torch.meshgrid(torch.arange(stereo_W), torch.arange(stereo_H), indexing="xy"), dim=2) # Need to be along W for first element in each "2"; fH x fW x 2
        meshgrid = (meshgrid * self.stereo_downsample + self.stereo_downsample / 2).to(curr_global2img) # each pixel exists at its center
        meshgrid_xyd1 = torch.cat([
            meshgrid[None, None, None, :, :, :].repeat(B, N, self.stereo_sampling_num, 1, 1, 1),
            depth_bins_to_sample[:, :, :, :, :, None],
            depth_bins_to_sample.new_ones((B, N, self.stereo_sampling_num, stereo_H, stereo_W, 1))
        ], dim=5) # B x N x 118 x stereo_H x stereo_W x 4
        curr_unaug_cam_meshgrid_xyd1 = torch.inverse(curr_img_forward_aug)[:, :, None, None, None, :, :] @ meshgrid_xyd1.unsqueeze(-1) # B x N x 118 x stereo_H x stereo_W x 4 x 1
        curr_unaug_cam_meshgrid_xyd1[..., :2, 0] *= curr_unaug_cam_meshgrid_xyd1[..., [2], 0]

        global_meshgrid_xyd1 = torch.inverse(curr_global2img)[:, :, None, None, None, :, :] @ curr_unaug_cam_meshgrid_xyd1  # B x N x 118 x stereo_H x stereo_W x 4 x 1
        finite_check(global_meshgrid_xyd1)

        ## Then, transform it to prev cameras
        global_meshgrid_xyd1 = global_meshgrid_xyd1[:, None, :, :, :, :, :, :].repeat(1, N, 1, 1, 1, 1, 1, 1) # B x prev_N x curr_N x 118 x stereo_H x stereo_W x 4 x 1. First N is prev cameras
        prev_unaug_cam_meshgrid_xyd1 = prev_global2img[:, :, None, None, None, None, :, :] @ global_meshgrid_xyd1


        prev_unaug_cam_meshgrid_xyd1[..., :2, 0] /= torch.maximum(prev_unaug_cam_meshgrid_xyd1[..., [2], 0], 
                                                                torch.ones_like(prev_unaug_cam_meshgrid_xyd1[..., [2], 0]) * eps)
        prev_meshgrid_xyd1 = prev_img_forward_aug[:, :, None, None, None, None, :, :] @ prev_unaug_cam_meshgrid_xyd1 # B x prev_N x curr_N x 118 x stereo_H x stereo_W x 4 x 1
        prev_meshgrid_xyd1 = prev_meshgrid_xyd1.squeeze(-1) # B x prev_N x curr_N x 118 x stereo_H x stereo_W x 4
        finite_check(prev_meshgrid_xyd1) 

        return prev_meshgrid_xyd1

    @auto_fp16(apply_to=('curr_sem_feats', ))
    def get_mono_depth(self, curr_sem_feats, rots, trans, intrins, post_rots, post_trans):
        B, N, sem_C, sem_H, sem_W = curr_sem_feats.shape
        curr_sem_feats = curr_sem_feats.view(B * N, sem_C, sem_H, sem_W)
        mono_depth_feat = curr_sem_feats
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1)],dim=1)
        mono_depth_feat = self.se(mono_depth_feat, cam_params)
        mono_depth_feat = self.extra_depthnet(mono_depth_feat)[0]
        mono_depth_feat = self.dcn(mono_depth_feat)
        mono_depth_digit = self.depthnet(mono_depth_feat)

        return mono_depth_digit


    @auto_fp16(apply_to=('curr_sem_feats', 'curr_stereo_feats', 'prev_stereo_feats'))
    def forward(self, 
                curr_sem_feats, 
                rots, trans, intrins, post_rots, post_trans,
                curr_stereo_feats=None, prev_stereo_feats=None, 
                prev_global2img=None, prev_img_forward_aug=None, curr_global2img=None, curr_img_forward_aug=None, curr_unaug_cam_to_prev_unaug_cam=None):

        B, N, sem_C, sem_H, sem_W = curr_sem_feats.shape

        curr_sem_feats = curr_sem_feats.view(B * N, sem_C, sem_H, sem_W)
        curr_img_feat = self.featnet(curr_sem_feats)
        mono_depth_digit = self.get_mono_depth(curr_sem_feats.view(B, N, sem_C, sem_H, sem_W), rots, trans, intrins, post_rots, post_trans)

        if not self.do_history_stereo_fusion:
            assert curr_stereo_feats is None
            depth_digit = mono_depth_digit
        else:
            B, N, stereo_C, stereo_H, stereo_W = curr_stereo_feats.shape
            eps = self.stereo_eps
            
            assert self.data_config['input_size'][0] // self.stereo_downsample == stereo_H
            assert curr_stereo_feats is not None

            # Do stereo
            with torch.no_grad():
                ## Stereo Sampling
                # First figure out what depths to sample
                # Do the gaussian sampling
                gauss_sample_distr_log = mono_depth_digit.log_softmax(dim=1)
                gauss_sample_depth_idxs = []
                for _ in range(self.stereo_sampling_num):
                    curr_gauss_sample_depth_idxs = gauss_sample_distr_log.argmax(dim=1)
                    uncertainty_reduction = self.log_inv_gaussians[curr_gauss_sample_depth_idxs].permute(0, 3, 1, 2)
                    gauss_sample_distr_log = gauss_sample_distr_log + uncertainty_reduction 
                    gauss_sample_depth_idxs.append(curr_gauss_sample_depth_idxs)
                gauss_sample_depth_idxs = torch.stack(gauss_sample_depth_idxs, dim=1) # B*N x k x sem_H x sem_W
                gauss_sample_depth_idxs = gauss_sample_depth_idxs.sort(dim=1).values
                gauss_sample_depths = self.bin_centers[gauss_sample_depth_idxs] # B*N x k x sem_H x sem_W

                # Now we have depth idxs and their depths. upsample it (via repeat) up to stereo_H & stereo_W. 
                sample_depth_idxs = gauss_sample_depth_idxs.view(B * N, self.stereo_sampling_num, sem_H, sem_W)
                sample_depths = F.interpolate(gauss_sample_depths, 
                                            scale_factor=(self.downsample // self.stereo_downsample), 
                                            mode='nearest').view(B, N, self.stereo_sampling_num, stereo_H, stereo_W) # B x N x k x stereo_H x stereo_W

                # Now get the sampling xyd1
                prev_meshgrid_xyd1 = self.get_prev_meshgrid_sampling_points(
                        sample_depths, curr_global2img, curr_img_forward_aug, prev_global2img, prev_img_forward_aug, 
                        curr_unaug_cam_to_prev_unaug_cam)
                
                prev_meshgrid_xyd1 = prev_meshgrid_xyd1.to(curr_sem_feats) # cast back to fp16
                prev_meshgrid_xy = prev_meshgrid_xyd1[..., :2] # B x prev_N x curr_N x k x stereo_H x stereo_W x 2
                prev_meshgrid_d = prev_meshgrid_xyd1[..., 2]
                valid_mask = prev_meshgrid_d > eps
                del prev_meshgrid_xyd1

                # At this point, we have sample_depth_idxs, prev_meshgrid_xy, and valid_mask
                # Normalize xy
                prev_meshgrid_xy_norm = prev_meshgrid_xy
                prev_meshgrid_xy_norm[..., 0] /= self.data_config['input_size'][1]
                prev_meshgrid_xy_norm[..., 1] /= self.data_config['input_size'][0]
                prev_meshgrid_xy_norm = prev_meshgrid_xy_norm * 2 - 1 # B x prev_N x curr_N x k x stereo_H x stereo_W x 2
                
                # Update valid_mask
                valid_mask = (valid_mask & (prev_meshgrid_xy_norm[..., 0] > -1.0)
                                        & (prev_meshgrid_xy_norm[..., 0] < 1.0)
                                        & (prev_meshgrid_xy_norm[..., 1] > -1.0)
                                        & (prev_meshgrid_xy_norm[..., 1] < 1.0)) # B x prev_N x curr_N x 118 x stereo_H x stereo_W

                ## Now do the sampling            
                group_size = (stereo_C // self.stereo_group_num)
                cost_volume = curr_stereo_feats.new_zeros(B, N, self.stereo_group_num, self.stereo_sampling_num, stereo_H, stereo_W) # N here is curr_N
                for prev_cam_idx in range(N):
                    ## Setup some stuff
                    # Get prev cam stuff
                    curr_prev_stereo_feats = prev_stereo_feats[:, prev_cam_idx, :, :, :] # B x C x stereo_H x stereo_W
                    curr_prev_meshgrid_xy_norm = prev_meshgrid_xy_norm[:, prev_cam_idx, :, :, :, :, :] # B x curr_N x 118 x stereo_H x stereo_W x 2
                    curr_valid_mask = valid_mask[:, prev_cam_idx, :, :, :, :] # B x curr_N x 118 x stereo_H x stereo_W

                    # Then, want to only get features from curr stereo for valid locations, so need to prepare for padding and unpadding that.
                    # Need to feed the cost volume afterwards too
                    curr_valid_mask_where = torch.where(curr_valid_mask)
                    curr_valid_mask_where_list = [ # get wheres for every sample in the batch.
                        [dim_where[curr_valid_mask_where[0] == batch_idx] for dim_where in curr_valid_mask_where[1:]] for batch_idx in range(B)]
                    curr_valid_mask_num_list = [ # num valid per sample in batch
                        len(tmp[0]) for tmp in curr_valid_mask_where_list]
                    curr_valid_mask_padded_valid_mask = torch.stack([ # mask on padded version later, True when not a padding value
                        torch.arange(max(curr_valid_mask_num_list), device=curr_prev_stereo_feats.device) < tmp_len for tmp_len in curr_valid_mask_num_list], dim=0) # B x max_length

                    ## Now get the sampled features in padded form
                    curr_prev_meshgrid_xy_norm_valid_list = [ # List of size B, inner is _ x 2
                        tmp[tmp_mask, :] for tmp, tmp_mask in zip(curr_prev_meshgrid_xy_norm, curr_valid_mask)]
                    curr_prev_meshgrid_xy_norm_valid_padded = pad_sequence(curr_prev_meshgrid_xy_norm_valid_list, batch_first=True) # B x max_length x 2
                    curr_prev_sampled_feats_padded = F.grid_sample(curr_prev_stereo_feats,
                                                                curr_prev_meshgrid_xy_norm_valid_padded.unsqueeze(2)) # B x C x max_length x 1
                    curr_prev_sampled_feats_padded = curr_prev_sampled_feats_padded.squeeze(3).permute(0, 2, 1) # B x C x max_length -> B x max_length x C

                    ## Get the corresponding curr features. Doing this to avoid the max-size tensor B x N x C x 118 x stereo_H x stereo_W. 
                    # Biggest tensor we have is B x max_length x C, which should be around B x C x 118 x stereo_H x stereo_W, so without the N factor.
                    with torch.set_grad_enabled(curr_stereo_feats.requires_grad):
                        curr_curr_stereo_feats_valid_list = [ # List of size B, inner is _ x C. ignore 118 dimension for now; it's only needed when indexing into cost volume. 
                            tmp[tmp_where[0], :, tmp_where[2], tmp_where[3]] for tmp, tmp_where in zip(curr_stereo_feats, curr_valid_mask_where_list)]
                        curr_curr_stereo_feats_valid_padded = pad_sequence(curr_curr_stereo_feats_valid_list, batch_first=True) # B x max_length x C

                        assert curr_curr_stereo_feats_valid_padded.shape[1] == curr_prev_sampled_feats_padded.shape[1] == curr_valid_mask_padded_valid_mask.shape[1], \
                            f"{curr_curr_stereo_feats_valid_padded.shape[1]} vs {curr_prev_sampled_feats_padded.shape[1]} vs {curr_valid_mask_padded_valid_mask.shape[1]}"
                    
                        ## Compute the group correlation 
                        curr_cost_volume = curr_prev_sampled_feats_padded * curr_curr_stereo_feats_valid_padded
                        curr_cost_volume = curr_cost_volume.view(B, curr_cost_volume.shape[1], self.stereo_group_num, group_size) # B x max_Length x group_num x group_size
                        curr_cost_volume = curr_cost_volume.sum(dim=3) # B x max_length x group_num

                        ## Now fill in cost_volume. Add it incrementally for now, will average later. Dot product is commutative with average
                        cost_volume[curr_valid_mask_where[0], curr_valid_mask_where[1], :, curr_valid_mask_where[2], curr_valid_mask_where[3], curr_valid_mask_where[4]] += \
                            curr_cost_volume[curr_valid_mask_padded_valid_mask]

                        del curr_cost_volume, curr_prev_sampled_feats_padded, curr_curr_stereo_feats_valid_padded

                with torch.set_grad_enabled(curr_stereo_feats.requires_grad):
                    ## Some points are projected to multiple prev cameras; average over those.
                    num_valid_per_point = valid_mask.float().sum(dim=1) # B x curr_N x k x stereo_H x stereo_W
                    num_valid_per_point = num_valid_per_point.unsqueeze(2) # B x curr_N x 1 x k x stereo_H x stereo_W
                    cost_volume = cost_volume / torch.maximum(num_valid_per_point, torch.ones_like(num_valid_per_point))
            
            assert curr_stereo_feats.requires_grad == cost_volume.requires_grad
            
            ## Get the cost volume logits
            cost_volume = self.similarity_net(cost_volume.view(B * N, self.stereo_group_num, self.stereo_sampling_num, stereo_H, stereo_W))
            stereo_depth_digit = cost_volume.squeeze(1) # B*N x k x stereo_H x stereo_W
            stereo_depth_digit = F.avg_pool2d(stereo_depth_digit.view(B * N, self.stereo_sampling_num, stereo_H, stereo_W), 
                                            self.downsample // self.stereo_downsample, 
                                            self.downsample // self.stereo_downsample).view(B, N, self.stereo_sampling_num, sem_H, sem_W) # B x N x k x sem_H x sem_W
            stereo_depth_digit = stereo_depth_digit.view(B * N, self.stereo_sampling_num, sem_H, sem_W)
            
            stereo_depth_digit_interp = interp_zeroends(
                torch.arange(self.D).to(sample_depth_idxs.device)[:, None, None, None], 
                sample_depth_idxs.permute(1, 0, 2, 3), 
                stereo_depth_digit.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)

            depth_digit = mono_depth_digit + stereo_depth_digit_interp

        depth_prob = self.get_depth_dist(depth_digit)

        ### Lift
        volume = depth_prob.unsqueeze(1) * curr_img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, sem_H, sem_W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        ### Splat
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        bev_feat = self.voxel_pooling(geom, volume)

        return bev_feat, depth_digit