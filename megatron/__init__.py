# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Top-level Megatron package for the local CLER checkout.

This file intentionally makes the repository's ``megatron`` tree a regular
package instead of a namespace package. On shared environments we also inject
third-party wheels into ``PYTHONPATH`` for server-side dependencies; if one of
those environments exposes a regular ``megatron`` package, Python would prefer
that package over this checkout and partially shadow local modules like
``megatron.core.parallel_state``.
"""

