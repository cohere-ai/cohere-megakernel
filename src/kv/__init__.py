"""Paged KV cache, shared by both phases.

``cache.{h,cpp}`` owns the block pool, sliding-window eviction, and the prefix
cache; ``pool.py`` is the Python-side state and the shared physical K/V arena.
Prefill writes into it and decode reads from it.
"""
