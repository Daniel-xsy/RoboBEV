# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import math
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from projects.mmdet3d_plugin.models.utils import center_to_corner_box3d

# from mmdet3d.core.bbox.box_np_ops import points_cam2img,center_to_corner_box3d
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
@HEADS.register_module()
class SparseRoIHead3D(CascadeRoIHead):
    r"""The RoIHead for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_
    and `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        stage_loss_weights (Tuple[float]): The loss
            weight of each stage. By default all stages have
            the same weight 1.
        bbox_roi_extractor (dict): Config of box roi extractor.
        mask_roi_extractor (dict): Config of mask roi extractor.
        bbox_head (dict): Config of box head.
        mask_head (dict): Config of mask head.
        train_cfg (dict, optional): Configuration information in train stage.
            Defaults to None.
        test_cfg (dict, optional): Configuration information in test stage.
            Defaults to None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    """

    def __init__(self,
                 num_stages=6,
                 num_level=4,
                 num_cam=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 bbox_roi_extractor=dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=7, sampling_ratio=2),
                     out_channels=256,
                     featmap_strides=[4, 8, 16, 32]),
                 mask_roi_extractor=None,
                 bbox_head=dict(
                     type='DIIHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 pc_range=None,
                 code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                 train_cfg=None,
                 mask_head=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        #self.stage_loss_weights = stage_loss_weights
        #self.proposal_feature_channel = proposal_feature_channel
        self.pc_range = pc_range
        self.scale_clamp = math.log(100000.0 / 16)
        self.code_weights = code_weights
        super(SparseRoIHead3D, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            mask_roi_extractor=mask_roi_extractor,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
            **kwargs)
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler), \
                    'Sparse R-CNN and QueryInst only support `PseudoSampler`'
        self.time_count = 0
        self.time = [0, 0, 0, 0]
        self.time_count2 = 0
        self.time2 = 0

    def _bbox_forward(self, stage, mlvl_feats, proposal_list, object_feats, img_metas, **kwargs):
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The index of current stage in
                iterative process.
            mlvl_feats (List[Tensor]): List of FPN features num_level = 4
            B, N, C, H, W = mlvl_feats[0].size()   N is num_cam
            1, 6, 256, H, W
            rois (Tensor): Rois in total batch. With shape (num_proposal, 5).
                the last dimension 5 represents (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            img_metas (dict): meta information of images.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
                Containing the following results:

                    - cls_score (Tensor): The score of each class, has
                      shape (batch_size, num_proposals, num_classes)
                      when use focal loss or
                      (batch_size, num_proposals, num_classes+1)
                      otherwise.
                    - decode_bbox_pred (Tensor): The regression results
                      with shape (batch_size, num_proposal, 4).
                      The last dimension 4 represents
                      [tl_x, tl_y, br_x, br_y].
                    - object_feats (Tensor): The object feature extracted
                      from current stage
                    - detach_cls_score_list (list[Tensor]): The detached
                      classification results, length is batch_size, and
                      each tensor has shape (num_proposal, num_classes).
                    - detach_proposal_list (list[tensor]): The detached
                      regression results, length is batch_size, and each
                      tensor has shape (num_proposal, 4). The last
                      dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        # num_imgs = len(img_metas)
        import time
        time_start = time.time()
        time1 = time.time()

        bbox_roi_extractor = self.bbox_roi_extractor[stage]

        _, bbox_feats = feature_sampling_reference_box3d(mlvl_feats, proposal_list, self.pc_range, bbox_roi_extractor, img_metas)
        # reference_box_3d, bbox_feats = feature_sampling_reference_point3d(mlvl_feats, proposal_list, self.pc_range,bbox_roi_extractor, img_metas)
        # no mask
        # # only for test sake
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # bbox_feats = torch.rand(1,900,256,7,7).to(device)

        time2 = time.time()
        bbox_head = self.bbox_head[stage]

        # proposal_list = reference_box_3d
        # cls_score: [bs,num_proposal, num_class]
        # bbox_pred: [bs,num_proposal, code_size]
        # object_feats& object_feats  [bs,num_proposal, embed_dims]
        cls_score, bbox_delta, object_feats, attn_feats = bbox_head(bbox_feats, object_feats)

        time3 = time.time()
        # bbox_pred[..., 0:2] = bbox_pred[..., 0:2] + inverse_sigmoid(proposal_list[..., 0:2])
        # bbox_pred[..., 4:5] = bbox_pred[..., 4:5] + inverse_sigmoid(proposal_list[..., 4:5])


        # bbox_delta[..., 0:1] = bbox_delta[..., 0:1].sigmoid()
        # bbox_delta[..., 1:2] = bbox_delta[..., 1:2].sigmoid()
        # bbox_delta[..., 4:5] = bbox_delta[..., 4:5].sigmoid()

        # bbox_delta[..., 2:3] = torch.clamp(bbox_delta[..., 2:3], max=self.scale_clamp)
        # bbox_delta[..., 3:4] = torch.clamp(bbox_delta[..., 3:4], max=self.scale_clamp)
        # bbox_delta[..., 5:6] = torch.clamp(bbox_delta[..., 5:6], max=self.scale_clamp)

        final_bbox_pred = self.apply_deltas(bbox_delta, proposal_list,self.pc_range)
        time4 = time.time()


        # final_bbox_pred[..., 0:1] = final_bbox_pred[..., 0:1].sigmoid()
        # final_bbox_pred[..., 1:2] = final_bbox_pred[..., 1:2].sigmoid()
        # final_bbox_pred[..., 4:5] = final_bbox_pred[..., 4:5].sigmoid()

        # proposal_list = [bbox_pred[:, i, :] for i in range(bbox_pred.size(1))]

        # Remark refine bboxes procedure is in srcn3d_head.py
        # proposal_list = self.bbox_head[stage].refine_bboxes(
        #     rois,
        #     rois.new_zeros(len(rois)),  # dummy arg
        #     bbox_pred.view(-1, bbox_pred.size(-1)),
        #     [rois.new_zeros(object_feats.size(1)) for _ in range(num_imgs)],
        #     img_metas)
        bbox_results = dict(
            cls_score=cls_score,
            # proposal_list=proposal_list,
            decode_bbox_pred = final_bbox_pred.clone(), #torch.cat(proposal_list),
            object_feats=object_feats,
            # attn_feats=attn_feats,
            # detach then use it in label assign
            # detach_cls_score_list=[
            #     cls_score[i].detach() for i in range(num_imgs)
            # ],
            detach_proposal_list = final_bbox_pred.clone().detach()
        )
        time_end = time.time()
        self.time_count += 1
        self.time[0] += time_end - time_start
        self.time[1] += time2 - time1
        self.time[2] += time3 - time2
        self.time[3] += time4 - time3
        # if self.time_count % 100 == 0:
        #     print("time:", self.time[0]/self.time_count)
        #     print("time 1:", self.time[1]/self.time_count)
        #     print("time 2:", self.time[2]/self.time_count)
        #     print("time 3:", self.time[3]/self.time_count)

        return bbox_results

    def apply_deltas(self, deltas, boxes, pc_range):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        cx, cy, w, l, cz, h, sin(rot), cos(rot), vx, vy
        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """

        boxes = boxes.to(deltas.dtype)
        deltas_split = torch.split(deltas, 1, dim=-1)
        boxes_split = torch.split(boxes, 1, dim=-1)
        wx, wy, ww, wl, wz, wh, _, _, _, _ = self.code_weights
        # dx = deltas[:,:, 0::10] / wx
        # dy = deltas[:,:, 1::10] / wy
        # dw = deltas[:,:, 2::10] / ww
        # dl = deltas[:,:, 3::10] / wl
        # dz = deltas[:,:, 4::10] / wz
        # dh = deltas[:,:, 5::10] / wh
        dx = deltas_split[0] / wx
        dy = deltas_split[1] / wy
        dw = deltas_split[2] / ww
        dl = deltas_split[3] / wl
        dz = deltas_split[4] / wz
        dh = deltas_split[5] / wh


        # ctr_x = inverse_sigmoid(boxes[:,:, 0:1])
        # ctr_y = inverse_sigmoid(boxes[:,:, 1:2])
        # ctr_z = inverse_sigmoid(boxes[:,:, 4:5])
        # ctr_x = boxes[:,:, 0:1]
        # ctr_y = boxes[:,:, 1:2]
        # ctr_z = boxes[:,:, 4:5]
        ctr_x = boxes_split[0]
        ctr_y = boxes_split[1]
        ctr_z = boxes_split[4]


        # widths = torch.exp(boxes[:,:, 2:3])
        # lengths = torch.exp(boxes[:,:, 3:4])
        # heights = torch.exp(boxes[:,:, 5:6])
        widths = torch.exp(boxes_split[2])
        lengths = torch.exp(boxes_split[3])
        heights = torch.exp(boxes_split[5])

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)
        dl = torch.clamp(dl, max=self.scale_clamp)

        # sin_rot = deltas[:, :, 6::10] #bs,num_proposals,1
        # cos_rot = deltas[:, :, 7::10]
        # sin_rot = deltas_split[6]
        # cos_rot = deltas_split[7]
        # ry = torch.atan2(sin_rot, cos_rot)

        # zeros, ones = torch.zeros(
        #     ry.size(), dtype=torch.float32).cuda(), torch.ones(
        #     ry.size(), dtype=torch.float32).cuda()
        # rot_list1 = torch.stack([torch.cos(ry), -torch.sin(ry), zeros], dim=0)
        # rot_list2 = torch.stack([torch.sin(ry), torch.cos(ry), zeros], dim=0)
        # rot_list3 = torch.stack([zeros, zeros, ones], dim=0)
        # rot_list = torch.stack([rot_list1, rot_list2, rot_list3], dim=0).squeeze(-1) #3,3,bs,num_proposals_

        # R_list = rot_list.permute(2, 3, 0, 1)  # (bs, N, 3, 3)

        # temp_delta_points = torch.stack([dx * widths,
        #                                  dy * lengths,
        #                                  dz * heights],
        #                                 dim=3)
        # rotated_delta_points = torch.matmul(temp_delta_points, R_list)
        # pred_ctr_x = rotated_delta_points[..., 0] + ctr_x
        # pred_ctr_y = rotated_delta_points[..., 1] + ctr_y
        # pred_ctr_z = rotated_delta_points[..., 2] + ctr_z
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * widths + ctr_y
        pred_ctr_z = dz * widths + ctr_z


        pred_w = torch.exp(dw) * widths
        pred_l = torch.exp(dl) * lengths
        pred_h = torch.exp(dh) * heights

        # reference_box_corners[..., 0:1] = reference_box_corners[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
        # reference_box_corners[..., 1:2] = reference_box_corners[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
        # reference_box_corners[..., 2:3] = reference_box_corners[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]


        pred_ctr_x = (pred_ctr_x-pc_range[0]) / (pc_range[3] - pc_range[0])
        pred_ctr_y = (pred_ctr_y-pc_range[1]) / (pc_range[4] - pc_range[1])
        pred_ctr_z = (pred_ctr_z-pc_range[2]) / (pc_range[5] - pc_range[2])
        pred_ctr_x = torch.clamp(pred_ctr_x, max=1.0, min=0.0)
        pred_ctr_y = torch.clamp(pred_ctr_y, max=1.0, min=0.0)
        pred_ctr_z = torch.clamp(pred_ctr_z, max=1.0, min=0.0)

        # pred_boxes = torch.zeros_like(deltas)
        # pred_boxes[:,:, 0::10] = pred_ctr_x   # x
        # pred_boxes[:,:, 1::10] = pred_ctr_y   # y
        # pred_boxes[:,:, 2::10] = pred_w.log()  # w
        # pred_boxes[:,:, 3::10] = pred_l.log()  # l
        # pred_boxes[:,:, 4::10] = pred_ctr_z   # z
        # pred_boxes[:,:, 5::10] = pred_h.log()   # h
        # pred_boxes[:,:, 6::10] = deltas[:,:, 6::10]  # sin(rot)
        # pred_boxes[:,:, 7::10] = deltas[:,:, 7::10] # cos(rot)
        # pred_boxes[:,:, 8::10] = deltas[:,:, 8::10]  # vx
        # pred_boxes[:,:, 9::10] = deltas[:,:, 9::10]  # vy
        pred_boxes = torch.cat([pred_ctr_x, pred_ctr_y, pred_w.log(), pred_l.log(), pred_ctr_z,
                                pred_h.log(), deltas_split[6], deltas_split[7], deltas_split[8],
                                deltas_split[9]], dim=-1)

        return pred_boxes

    def forward(self,
                mlvl_feats=None,
                mlvl_masks=None,
                mlvl_positional_encodings=None,
                proposal_boxes=None,
                proposal_features=None,
                img_metas=None,
             ):
        """Forward function in training stage.

        Args:
            mlvl_feats (list[Tensor]): list of multi-level img features.
            proposals (Tensor): Decoded proposal bboxes, has shape
                (batch_size, num_proposals, 10)
            proposal_features (Tensor): Expanded proposal
                features, has shape
                (batch_size, num_proposals, proposal_feature_channel)
            img_metas (list[dict]): list of image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape',
                'pad_shape', and 'img_norm_cfg'. For details on the
                values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            imgs_whwh (Tensor): Tensor with shape (batch_size, 4),
                    the dimension means
                    [img_width,img_height, img_width, img_height].
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components of all stage.
        """

        # num_proposals = proposal_boxes.size(1)

        all_stage_bbox_results = []
        proposal_list = proposal_boxes
        proposal_list[..., 0:1] = proposal_list[..., 0:1].sigmoid()
        proposal_list[..., 1:2] = proposal_list[..., 1:2].sigmoid()
        proposal_list[..., 4:5] = proposal_list[..., 4:5].sigmoid()
        # proposal_list = [proposal_boxes[i] for i in range(len(proposal_boxes))]
        object_feats = proposal_features
        # all_stage_loss = {}
        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, mlvl_feats, proposal_list, object_feats, img_metas)
            all_stage_bbox_results.append(bbox_results)
            proposal_list = bbox_results['detach_proposal_list']
            object_feats = bbox_results['object_feats']

        return all_stage_bbox_results



# def feature_sampling_reference_point3d(mlvl_feats, references, pc_range, bbox_roi_extractor, img_metas):
#     w_0 = 200
#     h_0 = 200
#     reference_points = torch.cat((references[..., 0:2], references[..., 4:5]), dim=2)
#     reference_points_3d = references.clone()
#     lidar2img = []
#     for img_meta in img_metas:
#         lidar2img.append(img_meta['lidar2img'])
#     lidar2img = np.asarray(lidar2img)
#     lidar2img = references.new_tensor(lidar2img) # (B, N, 4, 4)
#     reference_points[..., 0:1] = reference_points[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
#     reference_points[..., 1:2] = reference_points[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
#     reference_points[..., 2:3] = reference_points[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]
#     # reference_points (B, num_queries, 4)
#     reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
#     B, num_proposal = reference_points.size()[:2]
#     num_cam = lidar2img.size(1)
#     reference_points = reference_points.view(B, 1, num_proposal, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1) #bs,num_cam,num_proposal,4,1
#     lidar2img = lidar2img.view(B, num_cam, 1, 4, 4).repeat(1, 1, num_proposal, 1, 1)
#     reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
#     # [bs,num_query,num_level,num_cam,2]  (x,y)
#     eps = 1e-5
#     # mask = (reference_points_cam[..., 2:3] > eps)
#     reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
#         reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
#     reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
#     reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
#     reference_points_cam = (reference_points_cam - 0.5) * 2
#     # mask = (mask & (reference_points_cam[..., 0:1] > -1.0)
#     #              & (reference_points_cam[..., 0:1] < 1.0)
#     #              & (reference_points_cam[..., 1:2] > -1.0)
#     #              & (reference_points_cam[..., 1:2] < 1.0))
#     # mask = mask.view(B, num_cam, 1, num_proposal, 1, 1).permute(0, 2, 3, 1, 4, 5)
#     # mask = torch.nan_to_num(mask)

#     sampled_feats = []

#     C = mlvl_feats[0].shape[2]
#     reference_bbox2d_lvl = torch.stack([reference_points_cam[..., 0] - w_0 / 2,
#                                         reference_points_cam[..., 1] - h_0 / 2,
#                                         reference_points_cam[..., 0] + w_0 / 2,
#                                         reference_points_cam[..., 1] + h_0 / 2], dim=3)
#     for cam_id in range(num_cam):
#         mlvl_feats_cam = [feat[:, cam_id, :, :, :] for feat in mlvl_feats]
#         reference_points_cam_lvl = reference_bbox2d_lvl[:, cam_id, :, :]

#         reference_bbox2d_roi = torch.split(reference_points_cam_lvl, 1)
#         reference_bbox2d_roi = [ lvl[0,:,:] for lvl in reference_bbox2d_roi]
#         # reference_bbox2d_lvl = reference_bbox2d_lvl[0, :, :]
#         sampled_rois = bbox2roi(reference_bbox2d_roi)
#         # sampled_rois = torch.zeros([900, 5]).to(reference_bbox2d_lvl.device)
#         # sampled_rois[:, 1:] = reference_bbox2d_lvl.squeeze(0)
#         # test_roi = torch.tensor([0.0,1600.0,900.0,1700.0,1000.0]).unsqueeze(0).cuda()
#         #
#         sampled_feat = bbox_roi_extractor(tuple(mlvl_feats_cam[:bbox_roi_extractor.num_inputs]),
#                                           sampled_rois)  # [num_proposals, C, 7, 7]


#     # for cam_id in range(num_cam):
#     #     C = mlvl_feats[0].shape[2]
#     #     mlvl_feats_cam = [feat[:, cam_id, :, :, :] for feat in mlvl_feats]
#     #     reference_points_cam_lvl = reference_points_cam[:, cam_id, :, :]
#     #     reference_bbox2d_lvl = torch.stack([torch.clamp(reference_points_cam_lvl[:, :, 0] - w_0/2, min = 0),
#     #                                     torch.clamp(reference_points_cam_lvl[:, :, 1] - h_0 / 2, min=0),
#     #                                     torch.clamp(reference_points_cam_lvl[:, :, 0] + w_0/2, max = 1600),
#     #                                     torch.clamp(reference_points_cam_lvl[:, :, 1] + h_0/2, max = 928)], dim = 2)
#     #     # reference_bbox2d_lvl = [reference_bbox2d_lvl[i, :, :] for i in range(reference_bbox2d_lvl.size(0))]
#     #     #reference_bbox2d_lvl = reference_bbox2d_lvl[0, :, :]
#     #     # sampled_rois = bbox2roi(reference_bbox2d_lvl)
#     #     sampled_rois = torch.zeros([900, 5]).to(reference_bbox2d_lvl.device)
#     #     sampled_rois[:, 1:] = reference_bbox2d_lvl.unsqueeze(0)
#     #     # test_roi = torch.tensor([0.0,1600.0,900.0,1700.0,1000.0]).unsqueeze(0).cuda()
#     #     #
#     #     sampled_feat = bbox_roi_extractor(tuple(mlvl_feats_cam[:bbox_roi_extractor.num_inputs]),
#     #                                       sampled_rois)  # [num_proposals, C, 7, 7]

#         sampled_feat = sampled_feat.unsqueeze(0)
#         sampled_feats.append(sampled_feat)

#     sampled_feats = torch.stack(sampled_feats, dim=1)
#     sampled_feats = sampled_feats.permute(0, 2, 3, 1, 4, 5)
#     # mask = mask.reshape(B, 1, num_proposal, num_cam, 1, 1)
#     sampled_feats = sampled_feats.reshape(B, num_proposal, C,  num_cam, 7, 7)
#     # sampled_feats = sampled_feats * mask
#     sampled_feats = sampled_feats.permute(0, 1, 2, 4, 5, 3)
#     sampled_feats = sampled_feats.sum(-1)
#     sampled_feats = sampled_feats.view(B * num_proposal, C, 7, 7)

#     # [bs, num_proposals,256,7*7]
#     # [bs,num_proposals,256,1]
#     return reference_points_3d, sampled_feats #, mask

    # for lvl, feat in enumerate(mlvl_feats):
    #     B, N, C, H, W = feat.size()   # [1,6,256,1600,900]
    #     feat = feat.view(B*N, C, H, W)
    #     reference_points_cam_lvl = reference_points_cam.view(B*N, num_proposal, 1, 2)
    #     reference_bbox2d_lvl = reference_points_cam.view(B*N, num_proposal, 1, 2)
    #
    #
    #     sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
    #     sampled_feat = sampled_feat.view(B, N, C, num_proposal, 1).permute(0, 2, 3, 1, 4)
    #     sampled_feats.append(sampled_feat)
    # sampled_feats = torch.stack(sampled_feats, -1)
    # sampled_feats = sampled_feats.view(B, C, num_proposal, num_cam,  1, len(mlvl_feats))


    # for lvl, feat in enumerate(mlvl_feats):
    #     B, N, C, H, W = feat.size()   # [1,6,256,1600,900]
    #     feat = feat.view(B*N, C, H, W)
    #     reference_points_cam_lvl = reference_points_cam.view(B*N, num_proposal, 1, 2)
    #     reference_bbox2d_lvl = reference_points_cam.view(B*N, num_proposal, 1, 2)
    #
    #
    #     sampled_feat = F.grid_sample(feat, reference_points_cam_lvl)
    #     sampled_feat = sampled_feat.view(B, N, C, num_proposal, 1).permute(0, 2, 3, 1, 4)
    #     sampled_feats.append(sampled_feat)
    # sampled_feats = torch.stack(sampled_feats, -1)
    # sampled_feats = sampled_feats.view(B, C, num_proposal, num_cam,  1, len(mlvl_feats))




# def feature_sampling_reference_box3d(mlvl_feats, reference_boxes, pc_range, bbox_roi_extractor, img_metas):
#     '''

#     Args:
#         mlvl_feats: list of multi-level img features.
#         reference_boxes:  Boxes with shape of (bs, N, 10)
#             cx, cy, w, l, cz, h, sin(rot), cos(rot), vx, vy in TOP LiDAR coords

#     Returns:
#         reference_box_corners: Box corners with the shape of [bs,N, 8, 3].
#         sampled_feats:
#         mask

#     '''

#     batch_size = reference_boxes.shape[0]
#     num_proposals = reference_boxes.shape[1]
#     code_size = reference_boxes.shape[2]
#     reference_box_3d = reference_boxes.clone()

#     #TODO add offsets on reference_boxes deformable module
#     #TODO  reference_box_corners (B, num_queries, 8, 4)
#     # reference_boxes = torch.concat(reference_boxes,dim=0).view(-1,code_size)

#     reference_boxes[..., 0:1] = reference_boxes[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
#     reference_boxes[..., 1:2] = reference_boxes[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
#     reference_boxes[..., 4:5] = reference_boxes[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2]

#     reference_box_corners = boxes3d_to_corners3d(reference_boxes, bottom_center=False)

#     # reference_box_corners = reference_box_corners.clone()
#     # reference_box_corners_3d = reference_box_corners.clone()

#     #(B, N, 8, 3)
#     # restore to real-world scale
#     # reference_box_corners[..., 0:1] = reference_box_corners[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
#     # reference_box_corners[..., 1:2] = reference_box_corners[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
#     # reference_box_corners[..., 2:3] = reference_box_corners[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

#     # transform from lidar coordinate to six camera coordinates

#     lidar2img = []
#     for img_meta in img_metas:
#         lidar2img.append(img_meta['lidar2img'])
#     lidar2img = np.asarray(lidar2img)
#     lidar2img = reference_boxes.new_tensor(lidar2img) # (B, num_cam, 4, 4)
#     reference_box_corners = torch.cat((reference_box_corners, torch.ones_like(reference_box_corners[..., :1])), -1)
#     B , num_proposal = reference_box_corners.size()[:2]
#     num_cam = lidar2img.size(1)
#     reference_box_corners = reference_box_corners.view(B, 1, num_proposal, 8, 4).repeat(1, num_cam, 1, 1, 1).unsqueeze(-1)
#     lidar2img = lidar2img.view(B, num_cam, 1, 1, 4, 4).repeat(1, 1, num_proposal, 8, 1, 1)


#     reference_points_cam = torch.matmul(lidar2img, reference_box_corners).squeeze(-1) #(BS, num_cam, num_proposals, 8, 4)

#     # normalize real-world points back to normalized [-1,-1,1,1] image coordinate
#     eps = 1e-5
#     mask = (reference_points_cam[..., 2:3] > eps) #负深度mask
#     mask = torch.all(mask, dim=3).permute(0,2,1,3) # (BS,  num_proposals, num_cam, 1)
#     reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
#         reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps) #?
#     # reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
#     # reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
#     #reference_points_cam = (reference_points_cam - 0.5) * 2 # origin
#     box_corners_in_image = reference_points_cam

#     # expect box_corners_in_image: [B,N, 8, 2] -- [B,num_cam,N,8,2]
#     minxy = torch.min(box_corners_in_image, dim=3).values # [B,num_cam,N,2]
#     maxxy = torch.max(box_corners_in_image, dim=3).values

#     #TODO expect bbox (B, num_proposal, num_cam, 4)   (x1,y1,x2,y2)
#     reference_bbox2d = torch.cat([minxy, maxxy], dim=3).permute(0, 2, 1, 3) #  [bs, num_cam, num_proposals, 4] --  [bs,  num_proposals,num_cam, 4]
#     ox = img_metas[0]['img_shape'][0][1]
#     oy = img_metas[0]['img_shape'][0][0]
#     full_image_bbox2d = torch.tensor([[0, 0, ox, oy]], dtype=torch.float32).cuda().repeat(reference_bbox2d.size(2), 1)
#     reference_bbox2d_all = reference_bbox2d.reshape(batch_size * num_proposals, num_cam, 4).detach().cpu().numpy()
#     gious = np.stack([bbox_overlaps(reference_bbox2d_all[:, i, :], full_image_bbox2d.cpu().numpy()[i:(i+1), :]) for i in range(reference_bbox2d_all.shape[1])], axis=1)
#     # expect to remove ROI's whose windows is too small
#     gious = torch.from_numpy(gious).cuda()
#     gious = gious[:, :, 0].view(batch_size, num_proposals, num_cam, 1)

#     mask = (mask & (gious > 1e-5))
#     mask = torch.nan_to_num(mask)
#     #reference_bbox2d = mask * reference_bbox2d

#     sampled_feats = []

#     for cam_id in range(num_cam):
#         C = mlvl_feats[0].shape[2]
#         mlvl_feats_cam = [feat[:, cam_id, :, :, :] for feat in mlvl_feats]
#         reference_bbox2d_lvl = reference_bbox2d[:, :, cam_id, :].reshape(B, num_proposals, 4)
#         #reference_bbox2d_lvl = [reference_bbox2d_lvl[i, :, :] for i in range(reference_bbox2d_lvl.size(0))]
#         reference_bbox2d_lvl = reference_bbox2d_lvl[0, :, :]
#         #sampled_rois = bbox2roi(reference_bbox2d_lvl)
#         sampled_rois = torch.zeros([900, 5]).to(reference_bbox2d_lvl.device)
#         sampled_rois[:, 1:] = reference_bbox2d_lvl.unsqueeze(0)
#         # test_roi = torch.tensor([0.0,1600.0,900.0,1700.0,1000.0]).unsqueeze(0).cuda()
#         #
#         sampled_feat = bbox_roi_extractor(tuple(mlvl_feats_cam[:bbox_roi_extractor.num_inputs]), sampled_rois) #[num_proposals, C, 7, 7]

#         sampled_feat = sampled_feat.unsqueeze(0)
#         sampled_feats.append(sampled_feat)


#     sampled_feats = torch.stack(sampled_feats, dim=1)
#     sampled_feats = sampled_feats.permute(0,3,2,1,4,5)
#     mask = mask.reshape(B, 1, num_proposal, num_cam, 1, 1)
#     #sampled_feats = sampled_feats.view(B, num_proposal, C,  7, 7, num_cam, len(mlvl_feats))
#     sampled_feats = sampled_feats * mask
#     sampled_feats = sampled_feats.permute(0,2,1,4,5,3)
#     sampled_feats = sampled_feats.sum(-1)
#     sampled_feats  = sampled_feats.view(B*num_proposal, C, 7, 7)

#     # [bs, num_proposals,256,7*7]
#     # [bs,num_proposals,256,1]
#     return reference_box_3d, sampled_feats, mask


def feature_sampling_reference_box3d(mlvl_feats, reference_boxes, pc_range, bbox_roi_extractor, img_metas):
    '''

    Args:
        mlvl_feats: list of multi-level img features.
        reference_boxes:  Boxes with shape of (bs, N, 10)
            cx, cy, w, l, cz, h, sin(rot), cos(rot), vx, vy in TOP LiDAR coords

    Returns:
        reference_box_corners: Box corners with the shape of [bs,N, 8, 3].
        sampled_feats:
        mask

    '''
    batch_size = reference_boxes.shape[0]
    num_proposals = reference_boxes.shape[1]
    code_size = reference_boxes.shape[2]
    reference_box_3d = reference_boxes.clone()

    #TODO add offsets on reference_boxes deformable module
    #TODO  reference_box_corners (B, num_queries, 8, 4)
    # reference_boxes = torch.concat(reference_boxes,dim=0).view(-1,code_size)

    reference_boxes[..., 0:1] = reference_boxes[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_boxes[..., 1:2] = reference_boxes[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_boxes[..., 4:5] = reference_boxes[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_box_corners = boxes3d_to_corners3d(reference_boxes, bottom_center=False)

    # reference_box_corners = reference_box_corners.clone()
    # reference_box_corners_3d = reference_box_corners.clone()

    #(B, N, 8, 3)
    # restore to real-world scale
    # reference_box_corners[..., 0:1] = reference_box_corners[..., 0:1]*(pc_range[3] - pc_range[0]) + pc_range[0]
    # reference_box_corners[..., 1:2] = reference_box_corners[..., 1:2]*(pc_range[4] - pc_range[1]) + pc_range[1]
    # reference_box_corners[..., 2:3] = reference_box_corners[..., 2:3]*(pc_range[5] - pc_range[2]) + pc_range[2]

    # transform from lidar coordinate to six camera coordinates

    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_boxes.new_tensor(lidar2img) # (B, num_cam, 4, 4)
    reference_box_corners = torch.cat((reference_box_corners, torch.ones_like(reference_box_corners[..., :1])), -1)
    B , num_proposal = reference_box_corners.size()[:2]
    num_cam = lidar2img.size(1)
    reference_box_corners = reference_box_corners.view(B, 1, num_proposal, 8, 4).repeat(1, num_cam, 1, 1, 1).unsqueeze(-1)
    lidar2img = lidar2img.view(B, num_cam, 1, 1, 4, 4).repeat(1, 1, num_proposal, 8, 1, 1)


    reference_points_cam = torch.matmul(lidar2img, reference_box_corners).squeeze(-1) #(BS, num_cam, num_proposals, 8, 4)

    # normalize real-world points back to normalized [-1,-1,1,1] image coordinate
    eps = 1e-5
    # mask = (reference_points_cam[..., 2:3] > eps) #负深度mask
    # mask = torch.all(mask, dim=3).permute(0,2,1,3) # (BS,  num_proposals, num_cam, 1)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps) #?
    # reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    # reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]
    #reference_points_cam = (reference_points_cam - 0.5) * 2 # origin
    box_corners_in_image = reference_points_cam

    # expect box_corners_in_image: [B,N, 8, 2] -- [B,num_cam,N,8,2]
    minxy = torch.min(box_corners_in_image, dim=3).values # [B,num_cam,N,2]
    maxxy = torch.max(box_corners_in_image, dim=3).values

    #TODO expect bbox (B, num_proposal, num_cam, 4)   (x1,y1,x2,y2)
    reference_bbox2d = torch.cat([minxy, maxxy], dim=3).permute(0, 2, 1, 3) #  [bs, num_cam, num_proposals, 4] --  [bs,  num_proposals,num_cam, 4]
    # ox = img_metas[0]['img_shape'][0][1]
    # oy = img_metas[0]['img_shape'][0][0]
    # full_image_bbox2d = torch.tensor([[0, 0, ox, oy]], dtype=torch.float32).cuda().repeat(reference_bbox2d.size(2), 1)
    # reference_bbox2d_all = reference_bbox2d.reshape(batch_size * num_proposals, num_cam, 4).detach().cpu().numpy()
    # gious = np.stack([bbox_overlaps(reference_bbox2d_all[:, i, :], full_image_bbox2d.cpu().numpy()[i:(i+1), :]) for i in range(reference_bbox2d_all.shape[1])], axis=1)
    # expect to remove ROI's whose windows is too small
    # gious = torch.from_numpy(gious).cuda()
    # gious = gious[:, :, 0].view(batch_size, num_proposals, num_cam, 1)

    # mask = (mask & (gious > 0))
    # mask = torch.nan_to_num(mask)
    #reference_bbox2d = mask * reference_bbox2d

    # sampled_feats = []
    # mlvl_feats_lvl_cams = [torch.split(feat, 1, dim=1) for feat in mlvl_feats]
    sampled_rois = None
    for cam_id in range(num_cam):
        # mask_cam = torch.squeeze(mask, dim=-1)[:, :, cam_id].reshape(-1)
        # mlvl_feats[0]: (b, n=6, c, h, w)
        # mlvl_feats_cam[0]: (b*n, c, h, w)
        B = mlvl_feats[0].shape[0]
        C = mlvl_feats[0].shape[2]
        # mlvl_feats_cam = [torch.squeeze(feats_lvl[cam_id], 1) for feats_lvl in mlvl_feats_lvl_cams]
        # mlvl_feats_cam = [feat[:, cam_id, :, :, :] for feat in mlvl_feats]
        reference_bbox2d_lvl = reference_bbox2d[:, :, cam_id, :].reshape(B, num_proposals, 4)
        # reference_bbox2d_lvl = [reference_bbox2d_lvl[i, :, :] for i in range(reference_bbox2d_lvl.size(0))]
        # reference_bbox2d_lvl = reference_bbox2d_lvl[0, :, :]
        reference_bbox2d_roi = torch.split(reference_bbox2d_lvl, 1)
        reference_bbox2d_roi = [lvl[0, :, :] for lvl in reference_bbox2d_roi]
        # sampled_rois = bbox2roi(reference_bbox2d_roi)
        if sampled_rois is None:
            temp_roi = bbox2roi(reference_bbox2d_roi)
            temp_roi[:, 0] = temp_roi[:, 0] + cam_id * B
            sampled_rois = temp_roi
        else:
            temp_roi = bbox2roi(reference_bbox2d_roi)
            temp_roi[:, 0] = temp_roi[:, 0] + cam_id * B
            sampled_rois = torch.cat([sampled_rois, temp_roi], dim=0)
        # test_roi = torch.tensor([0.0,1600.0,900.0,1700.0,1000.0]).unsqueeze(0).cuda()

        # reference_bbox2d_lvl = torch.split(reference_bbox2d_lvl.squeeze(0),1)
        # sampled_rois = bbox2roi(reference_bbox2d_lvl)
        # sampled_rois: (num_query*b, 5)
        # sampled_feat = bbox_roi_extractor(mlvl_feats_cam[:bbox_roi_extractor.num_inputs],
        #                                   sampled_rois)  # [num_proposals, C, 7, 7]

        # sampled_feat = sampled_feat.unsqueeze(0)
        # sampled_feats.append(sampled_feat)

    # sampled_feats = []
    # mlvl_feats_lvl_cams = [torch.split(feat, 1, dim=1) for feat in mlvl_feats]
    # sampled_rois = None
    # for cam_id in range(num_cam):
    #     # mask_cam = torch.squeeze(mask, dim=-1)[:, :, cam_id].reshape(-1)
    #     # mlvl_feats[0]: (b, n=6, c, h, w)
    #     # mlvl_feats_cam[0]: (b*n, c, h, w)
    #     C = mlvl_feats[0].shape[2]
    #     mlvl_feats_cam = [torch.squeeze(feats_lvl[cam_id], 1) for feats_lvl in mlvl_feats_lvl_cams]
    #     # mlvl_feats_cam = [feat[:, cam_id, :, :, :] for feat in mlvl_feats]
    #     reference_bbox2d_lvl = reference_bbox2d[:, :, cam_id, :].reshape(B, num_proposals, 4)
    #     # reference_bbox2d_lvl = [reference_bbox2d_lvl[i, :, :] for i in range(reference_bbox2d_lvl.size(0))]
    #     # reference_bbox2d_lvl = reference_bbox2d_lvl[0, :, :]
    #     reference_bbox2d_roi = torch.split(reference_bbox2d_lvl, 1)
    #     reference_bbox2d_roi = [lvl[0, :, :] for lvl in reference_bbox2d_roi]
    #     sampled_rois = bbox2roi(reference_bbox2d_roi)
    #     # if sampled_rois is None:
    #     #     sampled_rois = bbox2roi(reference_bbox2d_roi)
    #     # else:
    #     #     sampled_rois = torch.cat([sampled_rois, bbox2roi(reference_bbox2d_roi)], dim=0)
    #     # test_roi = torch.tensor([0.0,1600.0,900.0,1700.0,1000.0]).unsqueeze(0).cuda()
    #
    #     # reference_bbox2d_lvl = torch.split(reference_bbox2d_lvl.squeeze(0),1)
    #     # sampled_rois = bbox2roi(reference_bbox2d_lvl)
    #     # sampled_rois: (num_query*b, 5)
    #     sampled_feat = bbox_roi_extractor(mlvl_feats_cam[:bbox_roi_extractor.num_inputs],
    #                                       sampled_rois)  # [num_proposals, C, 7, 7]
    #
    #     sampled_feat = sampled_feat.unsqueeze(0)
    #     sampled_feats.append(sampled_feat)
    # sampled_feats = torch.stack(sampled_feats, dim=1)

    mlvl_feats_cam = [feat[0, :, :, :, :] for feat in mlvl_feats]
    sampled_feat = bbox_roi_extractor(mlvl_feats_cam[:bbox_roi_extractor.num_inputs],
                                      sampled_rois)  # [num_cam * num_proposals, C, 7, 7]
    sampled_feat = sampled_feat.view(num_cam, B, num_proposal, C, 7, 7)
    sampled_feat = sampled_feat.permute(1, 0, 2, 3, 4, 5)
    sampled_feats = sampled_feat

    sampled_feats = sampled_feats.permute(0, 2, 3, 1, 4, 5)
    sampled_feats = sampled_feats.reshape(B, num_proposal, C, num_cam, 7, 7)
    sampled_feats = sampled_feats.permute(0, 1, 2, 4, 5, 3)

    # mask = mask.reshape(B, 1, num_proposal, num_cam, 1, 1)
    # #sampled_feats = sampled_feats.view(B, num_proposal, C,  7, 7, num_cam, len(mlvl_feats))
    # sampled_feats = sampled_feats * mask

    sampled_feats = sampled_feats.sum(-1)
    sampled_feats = sampled_feats.view(B*num_proposal, C, 7, 7)

    # [bs, num_proposals,256,7*7]
    # [bs,num_proposals,256,1]
    return reference_box_3d, sampled_feats






def boxes3d_to_corners3d(boxes3d, bottom_center=True):
    """Convert kitti center boxes to corners.

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        boxes3d (torch.tensor): Boxes with shape of (bs, N, 10)
            cx, cy, w, l, cz, h, sin(rot), cos(rot), vx, vy in TOP LiDAR coords,
            see the definition of ry in nuScenes dataset.
        bottom_center (bool, optional): Whether z is on the bottom center
            of object. Defaults to True.

    Returns:
        torch.tensor: Box corners with the shape of [bs,N, 8, 3].
    """

    #TODO add batch size dimension in corners
    bs = boxes3d.shape[0]
    boxes_num = boxes3d.shape[1]

    #TODO complete this line
    cx, cy, w, l, cz, h, sin_rot, cos_rot, vx, vy = tuple([boxes3d[:, :, i] for i in range(boxes3d.shape[2])])
    ry = torch.atan2(sin_rot.clone(), cos_rot.clone())
    w = w.exp()
    l = l.exp()
    h=h.exp()

    # w, l, h: (B,N)
    x_corners = torch.stack(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dim=2)   # (B,N,8)
    y_corners = torch.stack(
        [-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dim=2)  #.T
    if bottom_center:
        z_corners = torch.zeros((bs, boxes_num, 8), dtype=torch.float32).cuda()
        z_corners[:, :, 4:8] = torch.unsqueeze(h, 2).expand(bs, boxes_num, 4)  # (bs, N, 8)
    else:
        z_corners = torch.stack([
            -h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.
        ], dim=2)  # .T

    #ry = rot # (bs, N)
    zeros, ones = torch.zeros(
        ry.size(), dtype=torch.float32).cuda(), torch.ones(
            ry.size(), dtype=torch.float32).cuda()
    rot_list1 = torch.stack([torch.cos(ry), -torch.sin(ry), zeros], dim=0)
    rot_list2 = torch.stack([torch.sin(ry), torch.cos(ry), zeros], dim=0)
    rot_list3 = torch.stack([zeros, zeros, ones], dim=0)
    rot_list = torch.stack([rot_list1, rot_list2, rot_list3], dim=0)  # (3, 3, bs, N)

    R_list = rot_list.permute(2, 3, 0, 1)  # (bs, N, 3, 3)

    temp_corners = torch.stack([x_corners, y_corners, z_corners],
                                  dim=3)  # (bs, N, 8, 3)
    rotated_corners = torch.matmul(temp_corners, R_list)  # (bs, N, 8, 3)
    x_corners = rotated_corners[:, :, :, 0] #(bs, N, 8, 1)
    y_corners = rotated_corners[:, :, :, 1]
    z_corners = rotated_corners[:, :, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, :, 0], boxes3d[:, :, 1], boxes3d[:, :, 4] #(bs, N)

    x = torch.unsqueeze(x_loc, 2) + x_corners.reshape(-1, boxes_num, 8) #(bs,N,8)
    y = torch.unsqueeze(y_loc, 2) + y_corners.reshape(-1, boxes_num, 8)
    z = torch.unsqueeze(z_loc, 2) + z_corners.reshape(-1, boxes_num, 8)

    corners = torch.stack(
        [x, y, z],
        dim=3)

    return corners.type(torch.float32)

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)