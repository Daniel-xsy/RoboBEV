import copy
import logging
from collections import defaultdict
from itertools import chain
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad
from torch.cuda.amp import GradScaler

from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.runner.fp16_utils import LossScaler, wrap_fp16_model
from mmcv.runner.hooks.hook import HOOKS, Hook
from mmcv.runner.hooks import OptimizerHook

@HOOKS.register_module()
class WarmupFp16OptimizerHook(OptimizerHook):
    """FP16 optimizer hook (using PyTorch's implementation).
    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.
    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.
    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """

    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 coalesce: bool = True,
                 bucket_size_mb: int = -1,
                 warmup_loss_scale_value: float = 1.,
                 warmup_loss_scale_iters: int = 100,
                 loss_scale: Union[float, str, dict] = 512.,
                 distributed: bool = True):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.warmup_loss_scale_value = warmup_loss_scale_value
        self.warmup_loss_scale_iters = warmup_loss_scale_iters
        self.distributed = distributed
        self._scale_update_param = None
        if loss_scale == 'dynamic':
            self.loss_scaler = GradScaler()
        elif isinstance(loss_scale, float):
            self._scale_update_param = loss_scale
            self.loss_scaler = GradScaler(init_scale=loss_scale)
        elif isinstance(loss_scale, dict):
            self.loss_scaler = GradScaler(**loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                                f'"dynamic", got {loss_scale}')

        self.post_warmup_scale = self.loss_scaler.get_scale()

    def before_run(self, runner) -> None:
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
            scaler_state_dict = runner.meta['fp16']['loss_scaler']
            self.loss_scaler.load_state_dict(scaler_state_dict)

    def copy_grads_to_fp32(self, fp16_net: nn.Module,
                            fp32_weights: Tensor) -> None:
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights,
                                            fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(
                        fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net: nn.Module,
                            fp32_weights: Tensor) -> None:
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                            fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner) -> None:
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.
        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        # clear grads of last iteration
        runner.model.zero_grad()
        runner.optimizer.zero_grad()

        self.loss_scaler.scale(runner.outputs['loss']).backward()
        self.loss_scaler.unscale_(runner.optimizer)
        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        if runner._iter < self.warmup_loss_scale_iters:
            self.loss_scaler.update(self.warmup_loss_scale_value)
        elif runner._iter == self.warmup_loss_scale_iters:
            runner.logger.info("Ending FP16 Warmup, setting scale to {}".format(self.post_warmup_scale))
            self.loss_scaler.update(self.post_warmup_scale)
        else:
            self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()