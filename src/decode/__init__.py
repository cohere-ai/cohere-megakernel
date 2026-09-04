"""Decode phase: the persistent megakernel, its host runtime, and its driver.

The megakernel covers decode only. Everything in this package is device code
for it (``megakernel.cuh``, ``launch.cuh``, ``gemm-n8-wgmma.cuh``), the host
runtime that launches it (``runtime.cu``, ``abi.h``), or the Python side that
builds its schedules and descriptors (``schedule.py``).
"""
