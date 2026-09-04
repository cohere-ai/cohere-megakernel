"""Prefill phase: PyTorch and FlashAttention kernels.

Prefill runs as ordinary PyTorch, chunked over the prompt, and writes straight
into the paged KV cache that decode then reads. It takes the GPU to itself:
the session pauses active decode requests for the duration and resumes them
once prefill completes.
"""
