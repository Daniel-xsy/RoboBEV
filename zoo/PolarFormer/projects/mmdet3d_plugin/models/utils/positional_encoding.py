# Copyright (c) OpenMMLab. All rights reserved.
import math
from turtle import forward

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
from mmcv.runner import BaseModule


@POSITIONAL_ENCODING.register_module()
class TransSinePositionalEncoding(BaseModule):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.,
                 init_cfg=None):
        super(TransSinePositionalEncoding, self).__init__(init_cfg)
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, x_range, y_range, z_pos=None):
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        bs = x_range.shape[0]
        if len(x_range.shape) <=2:
            x_len = x_range.shape[-1]
            y_len = y_range.shape[-1]
            x_embed = x_range.unsqueeze(-2).repeat(1, y_len, 1)
            y_embed = y_range.unsqueeze(-1).repeat(1, 1, x_len)
        else:
            y_len, x_len = x_range.shape[1:]
            x_embed = x_range
            y_embed = y_range

        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale

        if z_pos:
            z_embed = z_pos.view(bs,1,1).repeat(1, y_len, x_len)
            num_feat_xy = self.num_feats-2
            num_feat_z = 4
            
            dim_t_xy = torch.arange(
                num_feat_xy, dtype=torch.float32, device=x_range.device)
            dim_t_z = torch.arange(
                num_feat_z, dtype=torch.float32, device=x_range.device)
            dim_t_xy = self.temperature**(2 * (dim_t_xy // 2) / num_feat_xy)
            dim_t_z = self.temperature**(2 * (dim_t_z // 2) / num_feat_z)

            pos_x = x_embed[:, :, :, None] / dim_t_xy
            pos_y = y_embed[:, :, :, None] / dim_t_xy
            pos_z = z_embed[:, :, :, None] / dim_t_z
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            B, H, W = bs, y_len, x_len
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos_z = torch.stack(
                (pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos = torch.cat((pos_y, pos_x, pos_z), dim=3).permute(0, 3, 1, 2)
        else:
            dim_t = torch.arange(
                self.num_feats, dtype=torch.float32, device=x_range.device)
            dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            # use `view` instead of `flatten` for dynamically exporting to ONNX
            B, H, W = bs, y_len, x_len
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                dim=4).view(B, H, W, -1)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == '__main__':
    # PE = PositionalEncoding(128,max_len=80)
    # input = torch.zeros(80,200,128)
    # output = PE(input)
    # import ipdb
    # ipdb.set_trace()
    # print(output.shape)
    pe = TransSinePositionalEncoding(128)
    x_range = torch.arange(2.5, 4.5, 2/200).unsqueeze(0)
    y_range = torch.arange(0., 80., 1).unsqueeze(0)
    import ipdb
    ipdb.set_trace()
    pos = pe(x_range, y_range)
    print(pos.shape)