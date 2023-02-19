import torch
import torchvision
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.ora3d.models.transformer.utils.grid_mask import GridMask
from projects.ora3d.models.detectors.utils.overlap_stereo import stereo_matching, StereoMerging
from projects.ora3d.models.detectors.utils.sgbm import generate_disparity_map
from projects.ora3d.models.detectors.utils.find_overlap_region import find_overlap_region, overlap_region_pts
import numpy as np


@DETECTORS.register_module()
class Ora3D(MVXTwoStageDetector):
    """Ora3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Ora3D,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.stereo_matcher = StereoMerging(256)

    def overlap_stereo_matching(self, left_feats, right_feats, overlap_regions_pts, stereo_gt):
        stereo_matching_losses = list()

        for i in range(6):
            stereo_loss, depth_output = stereo_matching(left_feats[i], right_feats[i], stereo_gt[i], self.stereo_matcher, overlap_regions_pts[i])
            stereo_matching_losses.append(stereo_loss)

        total_stereo_loss = None

        for loss in stereo_matching_losses:
            if total_stereo_loss is None:
                total_stereo_loss = loss
            total_stereo_loss += loss

        del left_feats, right_feats, stereo_gt, stereo_matching_losses
        torch.cuda.empty_cache()

        return total_stereo_loss

    def crop_feature_map(self, img_feats):
        stereo_matching_pair = [(1, 0), (5, 1), (0, 2), (4, 3), (2, 4), (3, 5)]
        mlvl_left_features = list()
        mlvl_right_features = list()

        for pair in stereo_matching_pair:
            left_features, right_features = list(), list()
            right_idx, left_idx = pair[0], pair[1]

            for i in range(3):
                lx_ = int(img_feats[i][left_idx].shape[2] * 0.8)
                x_width = int(img_feats[i][left_idx].shape[2] * 0.2)
                y_height = img_feats[i][left_idx].shape[1]
                left_feature = torchvision.transforms.functional.crop(img_feats[i][left_idx].unsqueeze(0), 0, lx_,
                                                                      y_height, x_width)
                left_features.append(left_feature)
                right_feature = torchvision.transforms.functional.crop(img_feats[i][right_idx].unsqueeze(0), 0, 0,
                                                                       y_height, x_width)
                right_features.append(right_feature)

            mlvl_left_features.append(left_features)
            mlvl_right_features.append(right_features)

        return mlvl_left_features, mlvl_right_features

    def preprocess_stereo_matching_gt(self, img_a, img_b):
        stereo_matching_pair = [(1, 0), (5, 1), (0, 2), (4, 3), (2, 4), (3, 5)]
        stereo_matching_gt = list()

        for t_idx in range(6):
            left_idx = stereo_matching_pair[t_idx][1]
            right_idx = stereo_matching_pair[t_idx][0]

            img_left = img_b[left_idx].permute(1, 2, 0)  # right side of left image
            img_right = img_a[right_idx].permute(1, 2, 0)  # left side of right image

            stereo_matching_disparity = generate_disparity_map(
                img_left.cpu().numpy(),
                img_right.cpu().numpy()
            )
            stereo_matching_gt.append(torch.FloatTensor(stereo_matching_disparity))

        stereo_matching_gt = torch.stack(stereo_matching_gt).unsqueeze(dim=1).cuda().float()
        return stereo_matching_gt

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        lidar2img = list()
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        overlap_regions = find_overlap_region(lidar2img)
        overlap_regions_pts = overlap_region_pts(overlap_regions)

        img_a = torchvision.transforms.functional.crop(img, 0, 0, 928, 320)
        img_b = torchvision.transforms.functional.crop(img, 0, 1280, 928, 320)
        stereo_matching_gt = self.preprocess_stereo_matching_gt(img_a, img_b)
        mlvl_left_features, mlvl_right_features = self.crop_feature_map(img_feats)
        total_stereo_loss = self.overlap_stereo_matching(mlvl_left_features, mlvl_right_features, overlap_regions_pts, stereo_matching_gt)

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        del mlvl_left_features, mlvl_right_features, img_feats, img
        torch.cuda.empty_cache()
        return img_feats_reshaped, total_stereo_loss

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats, total_stereo_loss = self.extract_img_feat(img, img_metas)
        return img_feats, total_stereo_loss

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        del outs, loss_inputs
        torch.cuda.empty_cache()
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        img_feats, total_stereo_loss = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        stereo_loss_weight = 0.5
        total_stereo_loss = total_stereo_loss * stereo_loss_weight
        losses.update({'loss_stereo': total_stereo_loss})

        del img_feats
        torch.cuda.empty_cache()
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
