# From https://github.com/johnsmith0031/alpaca_lora_4bit/blob/main/autograd_4bit.py
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
import torch.nn as nn
import time
import math
from torch.cuda.amp import custom_bwd, custom_fwd
from colorama import init, Fore, Back, Style

init(autoreset=True)

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


def get_buffer(shape_of_qweight, dtype=torch.float16, device="cuda"):
    if not cache_buffer:
        return torch.zeros(
            (shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device
        )
    if shape_of_qweight not in buffer_mat_dic.keys():
        buffer_mat_dic[shape_of_qweight] = torch.zeros(
            (shape_of_qweight[0] * 8, shape_of_qweight[1]), dtype=dtype, device=device
        )
    else:
        if buffer_mat_dic[shape_of_qweight].device != device:
            buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(
                device
            )
        if buffer_mat_dic[shape_of_qweight].dtype != dtype:
            buffer_mat_dic[shape_of_qweight] = buffer_mat_dic[shape_of_qweight].to(
                dtype=dtype
            )
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
        print("_matmul4bit_v1")
    assert qweight.shape[0] * 8 == x.shape[-1]
    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros(
        (x.shape[0], qweight.shape[-1]), dtype=torch.float16, device=x.device
    )
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
        print("_matmul4bit_v2")
    assert qweight.shape[0] * 8 == x.shape[-1]
    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros(
        (x.shape[0], qweight.shape[-1]), dtype=torch.float16, device=x.device
    )
    dtype = x.dtype
    if faster:
        x = x.half()
        quant_cuda.vecquant4matmul_faster(
            x, qweight, y, scales, zeros, g_idx, x.shape[-1] // 2
        )
    else:
        x = x.float()
        quant_cuda.vecquant4matmul(x, qweight, y, scales, zeros, g_idx)
    y = y.to(dtype)
    return y.reshape(outshape)


def _matmul4bit_v1_recons(x, qweight, scales, zeros, transpose=False):
    if debug:
        print("_matmul4bit_v1_recons")
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
        print("_matmul4bit_v2_recons")
    if not transpose:
        assert qweight.shape[0] * 8 == x.shape[-1]
    else:
        assert qweight.shape[1] == x.shape[-1]
    buffer = get_buffer(qweight.shape, dtype=scales.dtype, device=qweight.device)
    quant_cuda.vecquant4recons_v2(qweight, buffer, scales, zeros, g_idx)
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
                    output = _matmul4bit_v1_recons(
                        x.half(), qweight, scales.half(), zeros.half()
                    )
                else:
                    output = _matmul4bit_v1(x, qweight, scales, zeros)
        else:
            output = _matmul4bit_v1(x, qweight, scales, zeros)
    else:
        if g_idx is None:
            g_idx = torch.zeros(
                qweight.shape[0] * 8, dtype=torch.int32, device=x.device
            )
        # use v2
        if use_new:
            if auto_switch:
                if np.prod(x.shape[:-1]) > auto_switch_thd:
                    output = _matmul4bit_v2_recons(
                        x.half(), qweight, scales.half(), zeros, g_idx
                    )
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
    z_mat = torch.zeros(
        (zeros.shape[1], 256), dtype=torch.int, device=zeros.device
    ) + zeros.reshape((-1, 1))
    z_buffer = torch.zeros(
        (z_mat.shape[0] * 8, z_mat.shape[1]), dtype=torch.float16, device=zeros.device
    )
    z_zeros = torch.zeros(z_mat.shape[1], dtype=torch.float16, device=zeros.device)
    z_scales = torch.ones(z_mat.shape[1], dtype=torch.float16, device=zeros.device)
    quant_cuda.vecquant4recons_v1(z_mat, z_buffer, z_scales, z_zeros)
    z_buffer = z_buffer[:, 0]
    zeros_recons = z_buffer * scales + scales
    return zeros_recons


class AutogradMatmul4bitCuda(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, x, qweight, scales, zeros, g_idx, bits, maxq):
        ctx.save_for_backward(qweight, scales, zeros, g_idx)
        if g_idx is None:
            output = _matmul4bit_v1_recons(x, qweight, scales, zeros)
        else:
            output = _matmul4bit_v2_recons(x, qweight, scales, zeros, g_idx)
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, zeros, g_idx = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            if g_idx is None:
                grad = _matmul4bit_v1_recons(
                    grad_output, qweight, scales, zeros, transpose=True
                )
            else:
                grad = _matmul4bit_v2_recons(
                    grad_output, qweight, scales, zeros, g_idx, transpose=True
                )
        return grad, None, None, None, None, None, None


try:
    import triton_utils as tu

    class AutogradMatmul4bitTriton(torch.autograd.Function):
        @staticmethod
        @custom_fwd(cast_inputs=torch.float16)
        def forward(ctx, x, qweight, scales, qzeros, g_idx, bits, maxq):
            output = tu.triton_matmul(x, qweight, scales, qzeros, g_idx, bits, maxq)
            ctx.save_for_backward(qweight, scales, qzeros, g_idx)
            ctx.bits, ctx.maxq = bits, maxq
            output = output.clone()
            return output

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            qweight, scales, qzeros, g_idx = ctx.saved_tensors
            bits, maxq = ctx.bits, ctx.maxq
            grad_input = None

            if ctx.needs_input_grad[0]:
                grad_input = tu.triton_matmul_transpose(
                    grad_output, qweight, scales, qzeros, g_idx, bits, maxq
                )
            return grad_input, None, None, None, None, None, None

except ImportError:
    print('Triton not found. Please run "pip install triton".')


AutogradMatmul4bit = AutogradMatmul4bitCuda
backend = "cuda"


def switch_backend_to(to_backend):
    global AutogradMatmul4bit
    global backend
    if to_backend == "cuda":
        AutogradMatmul4bit = AutogradMatmul4bitCuda
        backend = "cuda"
        print(Style.BRIGHT + Fore.GREEN + "Using CUDA implementation.")
    elif to_backend == "triton":
        # detect if AutogradMatmul4bitTriton is defined
        if "AutogradMatmul4bitTriton" not in globals():
            raise ValueError("Triton not found. Please install triton")
        AutogradMatmul4bit = AutogradMatmul4bitTriton
        backend = "triton"
        print(Style.BRIGHT + Fore.GREEN + "Using Triton implementation.")
    else:
        raise ValueError("Backend not supported.")


def matmul4bit_with_backend(x, qweight, scales, qzeros, g_idx, bits, maxq):
    if backend == "cuda":
        return matmul4bit(x, qweight, scales, qzeros, g_idx)
    elif backend == "triton":
        assert qzeros.dtype == torch.int32
        return tu.triton_matmul(x, qweight, scales, qzeros, g_idx, bits, maxq)
    else:
        raise ValueError("Backend not supported.")


# Assumes layer is perfectly divisible into 256 * 256 blocks
class Autograd4bitQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, groupsize=-1, is_v1_model=False):
        super().__init__()
        bits = 4
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.maxq = 2**self.bits - 1
        groupsize = groupsize if groupsize != -1 else in_features
        self.groupsize = groupsize
        self.is_v1_model = is_v1_model
        self.disable_bias = True
        if is_v1_model:
            self.register_buffer("zeros", torch.empty((out_features, 1)))
            self.register_buffer("scales", torch.empty((out_features, 1)))
            self.g_idx = None
        else:
            self.register_buffer(
                "qzeros",
                torch.empty(
                    (
                        math.ceil(in_features / groupsize),
                        out_features // 256 * (bits * 8),
                    ),
                    dtype=torch.int32,
                ),
            )
            self.register_buffer(
                "scales",
                torch.empty((math.ceil(in_features / groupsize), out_features)),
            )
            self.register_buffer(
                "g_idx",
                torch.tensor(
                    [i // self.groupsize for i in range(in_features)], dtype=torch.int32
                ),
            )
        self.register_buffer("bias", torch.empty(out_features))
        self.register_buffer(
            "qweight",
            torch.empty(
                (in_features // 256 * (bits * 8), out_features), dtype=torch.int32
            ),
        )

    def forward(self, x):
        if torch.is_grad_enabled():
            out = AutogradMatmul4bit.apply(
                x,
                self.qweight,
                self.scales,
                self.qzeros if not self.is_v1_model else self.zeros,
                self.g_idx,
                self.bits,
                self.maxq,
            )
        else:
            out = matmul4bit_with_backend(
                x,
                self.qweight,
                self.scales,
                self.qzeros if not self.is_v1_model else self.zeros,
                self.g_idx,
                self.bits,
                self.maxq,
            )
        if not self.disable_bias:
            out += self.bias
        return out


def make_quant_for_4bit_autograd(
    module, names, name="", groupsize=-1, is_v1_model=False
):
    if isinstance(module, Autograd4bitQuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if name1 in names:
            setattr(
                module,
                attr,
                Autograd4bitQuantLinear(
                    tmp.in_features,
                    tmp.out_features,
                    groupsize=groupsize,
                    is_v1_model=is_v1_model,
                ),
            )
    for name1, child in module.named_children():
        make_quant_for_4bit_autograd(
            child,
            names,
            name + "." + name1 if name != "" else name1,
            groupsize=groupsize,
            is_v1_model=is_v1_model,
        )


def model_to_half(model):
    model.half()
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            if m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()
    print(Style.BRIGHT + Fore.YELLOW + "Converted as Half.")


def model_to_float(model):
    model.float()
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear):
            if m.is_v1_model:
                m.zeros = m.zeros.float()
            m.scales = m.scales.float()
            m.bias = m.bias.float()
    print(Style.BRIGHT + Fore.YELLOW + "Converted as Float.")


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def load_llama_model_4bit_low_ram(
    config_path,
    model_path,
    groupsize=-1,
    half=False,
    device_map="auto",
    seqlen=2048,
    is_v1_model=False,
):
    import accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config)
        model = model.eval()
        layers = find_layers(model)
        for name in ["lm_head"]:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(
            model, layers, groupsize=groupsize, is_v1_model=is_v1_model
        )
    model = accelerate.load_checkpoint_and_dispatch(
        model=model,
        checkpoint=model_path,
        device_map=device_map,
        no_split_module_classes=["LlamaDecoderLayer"],
    )

    model.seqlen = seqlen

    if half:
        model_to_half(model)

    tokenizer = LlamaTokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = "left"

    print(
        Style.BRIGHT
        + Fore.GREEN
        + f"Loaded the model in {(time.time()-t0):.2f} seconds."
    )

    return model, tokenizer


def load_llama_model_4bit_low_ram_and_offload(
    config_path,
    model_path,
    lora_path=None,
    groupsize=-1,
    seqlen=2048,
    max_memory=None,
    is_v1_model=False,
):
    import accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

    if max_memory is None:
        max_memory = {0: "24Gib", "cpu": "48Gib"}

    print(Style.BRIGHT + Fore.CYAN + "Loading Model ...")
    t0 = time.time()

    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(config_path)
        model = LlamaForCausalLM(config)
        model = model.eval()
        layers = find_layers(model)
        for name in ["lm_head"]:
            if name in layers:
                del layers[name]
        make_quant_for_4bit_autograd(
            model, layers, groupsize=groupsize, is_v1_model=is_v1_model
        )
    accelerate.load_checkpoint_in_model(
        model, checkpoint=model_path, device_map={"": "cpu"}
    )

    # rotary_emb fix
    for n, m in model.named_modules():
        if "rotary_emb" in n:
            cos_cached = m.cos_cached.clone().cpu()
            sin_cached = m.sin_cached.clone().cpu()
            break

    if lora_path is not None:
        from peft import PeftModel
        from monkeypatch.peft_tuners_lora_monkey_patch import Linear4bitLt

        model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            is_trainable=True,
        )
        print(Style.BRIGHT + Fore.GREEN + "{} Lora Applied.".format(lora_path))

    model.seqlen = seqlen

    print("Apply half ...")
    for n, m in model.named_modules():
        if isinstance(m, Autograd4bitQuantLinear) or (
            (lora_path is not None) and isinstance(m, Linear4bitLt)
        ):
            if m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()
            m.bias = m.bias.half()

    print("Dispatching model ...")
    device_map = accelerate.infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"]
    )
    model = accelerate.dispatch_model(
        model, device_map=device_map, offload_buffers=True, main_device=0
    )
    torch.cuda.empty_cache()
    print(
        Style.BRIGHT
        + Fore.YELLOW
        + "Total {:.2f} Gib VRAM used.".format(
            torch.cuda.memory_allocated() / 1024 / 1024
        )
    )

    # rotary_emb fix
    for n, m in model.named_modules():
        if "rotary_emb" in n:
            if getattr(m, "_hf_hook", None):
                if isinstance(m._hf_hook, accelerate.hooks.SequentialHook):
                    hooks = m._hf_hook.hooks
                else:
                    hooks = [m._hf_hook]
                for hook in hooks:
                    if hook.offload:
                        if (
                            n + ".sin_cached"
                            not in hook.weights_map.dataset.state_dict.keys()
                        ):
                            hook.weights_map.dataset.state_dict[
                                n + ".sin_cached"
                            ] = sin_cached.clone().cpu()
                            hook.weights_map.dataset.state_dict[
                                n + ".cos_cached"
                            ] = cos_cached.clone().cpu()

    tokenizer = LlamaTokenizer.from_pretrained(config_path)
    tokenizer.truncation_side = "left"

    print(
        Style.BRIGHT
        + Fore.GREEN
        + f"Loaded the model in {(time.time()-t0):.2f} seconds."
    )

    return model, tokenizer
