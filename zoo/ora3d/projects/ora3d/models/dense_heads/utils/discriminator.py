import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=256, out_features=100),
            nn.BatchNorm1d(900),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2),
        )

    def forward(self, input_feature):
        x = self.discriminator(input_feature)
        return F.sigmoid(x)


def discriminator_targets(all_bbox_2d_preds):
    total_overlap_label = list()
    for layer_preds in all_bbox_2d_preds:
        temp_overlap_label = [0] * 900
        for c_idx, bboxes in enumerate(layer_preds):
            bboxes_list = bboxes.tolist()
            for b_idx, bbox in enumerate(bboxes_list):
                if temp_overlap_label[b_idx] == 1:
                    continue
                x_, y_ = bbox[0], bbox[1]
                if 0 <= x_ < 320 and 0 <= y_ < 928:
                    temp_overlap_label[b_idx] = 1
                elif 1280 <= x_ < 1600 and 0 <= y_ < 928:
                    temp_overlap_label[b_idx] = 1
        total_overlap_label.append(temp_overlap_label)
    del all_bbox_2d_preds
    torch.cuda.empty_cache()
    return total_overlap_label
