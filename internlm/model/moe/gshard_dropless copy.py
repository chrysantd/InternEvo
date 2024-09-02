# Copyright (c) InternLM. All rights reserved.
from typing import Any, Tuple

import torch
from torch import Tensor

from internlm.accelerator import AcceleratorType, get_accelerator
from internlm.core.context import global_context as gpc
from internlm.utils.logger import get_logger

logger = get_logger(__file__)

internlm_accelerator = get_accelerator()


# Based on https://github.com/pytorch/pytorch/pull/40762
class AllToAll(torch.autograd.Function):
    """
    All to all communication
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group: torch.distributed.ProcessGroup = None,
        async_op=False,
    ) -> Tensor:  # type: ignore

        ctx.input_shape = inputs.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return inputs, None

        inputs = inputs.contiguous()
        out = (
            torch.empty_like(inputs)
            if output_split_sizes is None
            else inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]))
        )
        handle = torch.distributed.all_to_all_single(
            out,
            inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

        # if async_op=False, handle will be None
        return out, handle

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, _) -> Tuple[None, Tensor]:
        if ctx.needs_input_grad[0]:
            # Bypass the function if we are using only 1 GPU.
            world_size = torch.distributed.get_world_size(group=ctx.group)
            if world_size == 1:
                return grad_output, None, None, None, None

            grad_output = grad_output.contiguous()
            out = torch.empty(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
            torch.distributed.all_to_all_single(
                out,
                grad_output,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(x, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
    return AllToAll.apply(x, output_split_sizes, input_split_sizes, group, async_op)


gmm_ops = None
gmm_fwd, gmm_bwd, gmm_func = None, None, None


def try_import_gmm_fwd_bwd():
    try:
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU:
            from grouped_gemm.backend import gmm

            return gmm, gmm

        elif internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:

            def gmm_fwd(x, weight, batch_sizes, group_type=0):
                if (x.requires_grad or weight.requires_grad) and group_type != 0:
                    raise ValueError("group_type must be zero to compute gradients of x and weight!")

                group_list = torch.cumsum(batch_sizes, dim=-1)
                group_list = group_list.tolist()
                return gmm_ops.npu_gmm([x], [weight], [], group_list, group_type)[0]

            def gmm_bwd(x, weight, grad_outputs, batch_sizes):
                group_list = torch.cumsum(batch_sizes, dim=-1)
                group_list = group_list.tolist()

                dx, dw, _ = gmm_ops.npu_gmm_backward([grad_outputs], [x], [weight], group_list)
                return dx[0], dw[0], None, None, None

            return gmm_fwd, gmm_bwd
        else:
            if gpc.is_rank_for_log():
                logger.warning("gmm is not support. Please note this!")
            return None, None
    except (ModuleNotFoundError, ImportError):
        if gpc.is_rank_for_log():
            logger.warning("gmm is not support. Please note this!")
        return None, None


def try_import_gmm_func():
    try:
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.GPU:
            # To enable gemm permute optimizations:
            #   python3 -m pip install --verbose git+https://github.com/fanshiqing/grouped_gemm@v1.1.3
            from grouped_gemm.ops import gmm

        elif internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU:

            def gmm(x, weight, batch_sizes):
                from mindspeed.ops.gmm import npu_gmm

                group_list = torch.cumsum(batch_sizes, dim=-1)
                group_list = group_list.tolist()

                return npu_gmm(x, weight, bias=None, group_list=group_list, group_type=0)

        else:
            if gpc.is_rank_for_log():
                logger.warning("gmm is not support. Please note this!")
            return None

        return gmm
    except (ModuleNotFoundError, ImportError):
        if gpc.is_rank_for_log():
            logger.warning("gmm is not support. Please note this!")
        return None


def try_import_gmm():
    """
    get gemm ops

    return:
        gmm_fwd: gmm forward ops
        gmm_bwd: gmm backward ops
        gmm_func: gmm autograd function
    """
    global gmm_fwd, gmm_bwd, gmm_func
    if gpc.config.parallel.tensor.mode == "isp":
        if gmm_fwd and gmm_bwd:
            return gmm_fwd, gmm_bwd, None
        global gmm_ops
        if internlm_accelerator.get_accelerator_backend() == AcceleratorType.NPU and not gmm_ops:
            from mindspeed.op_builder import GMMOpBuilder

            gmm_ops = GMMOpBuilder().load()
        gmm_fwd, gmm_bwd = try_import_gmm_fwd_bwd()
        return gmm_fwd, gmm_bwd, None
    else:
        if gmm_func:
            return None, None, gmm_func
        gmm_func = try_import_gmm_func()
        return None, None, gmm_func
