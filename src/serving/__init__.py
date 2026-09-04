"""Serving layer: HTTP endpoint, request admission, and output parsing.

``session.py`` is the piece that joins the two phases, driving prefill and
continuous-batch decode for admitted requests. ``server.py`` and ``parse.py``
sit above it, dealing in HTTP requests and assistant text.
"""
