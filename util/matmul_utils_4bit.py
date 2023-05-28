# From https://github.com/johnsmith0031/alpaca_lora_4bit/blob/main/matmul_utils_4bit.py
# MIT License

# Copyright (c) 2023 John Smith

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np
from gptq_llama import quant_cuda


# Global Buffer
buffer_mat_dic = {}
use_new = True
auto_switch = True
auto_switch_thd = 8
debug = False
faster = True
cache_buffer = True

def get_buffer(shape_of_qweight, dtype=torch.float16, device='cuda'):
    if not cache_buffer:
        return torch.zeros((shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device)
    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros((shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device)
    else:
        if buffer_mat_dic[shape_of_qweight].device != device:
            buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(device)
        if buffer_mat_dic[shape_of_qweight].dtype != dtype:
            buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(dtype=dtype)
    return buffer_mat_dic[shape_of_qweight]


def _matmul4bit_v1(x, qweight, scales, zeros):
    """
    input x: (n, m)
    qweight: (j, k)
    where m == j*8

    perform x @ qweight

    return y:
    """
    if debug:
        print('_matmul4bit_v1')
    assert qweight.shape[0] * 8 == x.shape[-1]
    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype=torch.float16, device=x.device)
    dtype = x.dtype
    x = x.half()
    quant_cuda.vecquant4matmul_v1_faster(x, qweight, y, scales, zeros)
    y = y.to(dtype)
    return y.reshape(outshape)


def _matmul4bit_v2(x, qweight, scales, zeros, g_idx):
    """
    input x: (n, m)
    qweight: (j, k)
    where m == j*8

    perform x @ qweight

    return y:
    """
    if debug:
        print('_matmul4bit_v2')
    assert qweight.shape[0] * 8 == x.shape[-1]
    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype=torch.float16, device=x.device)
    dtype = x.dtype
    if faster:
        x = x.half()
        quant_cuda.vecquant4matmul_faster(x, qweight, y, scales, zeros, g_idx, x.shape[-1] // 2)
    else:
        x = x.float()
        quant_cuda.vecquant4matmul(x, qweight, y, scales, zeros, g_idx)
    y = y.to(dtype)
    return y.reshape(outshape)


def _matmul4bit_v1_recons(x, qweight, scales, zeros, transpose=False):
    if debug:
        print('_matmul4bit_v1_recons')
    if not transpose:
        assert qweight.shape[0] * 8 == x.shape[-1]
    else:
        assert qweight.shape[1] == x.shape[-1]
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
    quant_cuda.vecquant4recons_v1(qweight, buffer, scales, zeros)
    if not transpose:
        output = torch.matmul(x, buffer)
    else:
        output = torch.matmul(x, buffer.T)
    return output


def _matmul4bit_v2_recons(x, qweight, scales, zeros, g_idx, transpose=False):
    if debug:
        print('_matmul4bit_v2_recons')
    if not transpose:
        assert qweight.shape[0] * 8 == x.shape[-1]
    else:
        assert qweight.shape[1] == x.shape[-1]
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
    quant_cuda.vecquant4recons_v2(qweight, buffer, scales, zeros, g_idx)
    x = x.to(scales.dtype)
    if not transpose:
        output = torch.matmul(x, buffer)
    else:
        output = torch.matmul(x, buffer.T)
    return output


def matmul4bit(x, qweight, scales, zeros, g_idx=None):
    # detect if zeros is int32
    if zeros.dtype != torch.int32:
        # use v1
        if use_new:
            if auto_switch:
                if np.prod(x.shape[:-1]) > auto_switch_thd:
                    output = _matmul4bit_v1_recons(x.half(), qweight, scales.half(), zeros.half())
                else:
                    output = _matmul4bit_v1(x, qweight, scales, zeros)
        else:
            output = _matmul4bit_v1(x, qweight, scales, zeros)
    else:
        if g_idx is None:
            g_idx = torch.zeros(qweight.shape[0] * 8, dtype=torch.int32, device=x.device)
        # use v2
        if use_new:
            if auto_switch:
                if np.prod(x.shape[:-1]) > auto_switch_thd:
                    output = _matmul4bit_v2_recons(x.half(), qweight, scales.half(), zeros, g_idx)
                else:
                    output = _matmul4bit_v2(x, qweight, scales, zeros, g_idx)
        else:
            output = _matmul4bit_v2(x, qweight, scales, zeros, g_idx)
    return output


def v2_to_v1(scales, zeros):
    """
    Convert zeros in V2 model to V1 model when group_num = 1, for debugging
    depreciated
    """
    assert zeros.shape[0] == 1
    z_mat = torch.zeros((zeros.shape[1], 256), dtype=torch.int, device=zeros.device) + zeros.reshape((-1,1))
    z_buffer = torch.zeros((z_mat.shape[0] * 8, z_mat.shape[1]), dtype=torch.float16, device=zeros.device)
    z_zeros = torch.zeros(z_mat.shape[1], dtype=torch.float16, device=zeros.device)
    z_scales = torch.ones(z_mat.shape[1], dtype=torch.float16, device=zeros.device)
    quant_cuda.vecquant4recons_v1(z_mat, z_buffer, z_scales, z_zeros)
    z_buffer = z_buffer[:,0]
    zeros_recons = z_buffer * scales + scales
    return zeros_recons