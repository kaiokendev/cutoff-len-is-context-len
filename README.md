# cutoff-len-is-context-len
Demonstration that finetuning RoPE model on larger sequences than the pre-trained model adapts the model context limit

## A Length-Extrapolatable Transformer
https://arxiv.org/abs/2212.10554
> Position modeling plays a critical role in Transformers. In this paper, we focus on length extrapolation, i.e., training on short texts while evaluating longer sequences. We define attention resolution as an indicator of extrapolation. Then we propose two designs to improve the above metric of Transformers. Specifically, we introduce a relative position embedding to explicitly maximize attention resolution. Moreover, we use blockwise causal attention during inference for better resolution. We evaluate different Transformer variants with language modeling. Experimental results show that our model achieves strong performance in both interpolation and extrapolation settings. The code will be available at [this https URL](https://aka.ms/LeX-Transformer).

## RoFormer: Enhanced Transformer with Rotary Position Embedding
https://arxiv.org/abs/2104.09864
> Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: [this https URL](https://huggingface.co/docs/transformers/model_doc/roformer).

## RoPE with xPos implementation
from Lucidrains (Phil Wang)

[x-transformers](https://github.com/lucidrains/x-transformers)

Adapted for HuggingFace Transformers LLaMa

## Dilated RoPE Explanation
https://kaiokendev.github.io/til#extending-context-to-8k
