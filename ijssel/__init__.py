"""Stream-like API for Python."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

from .main import Stream
from .util import identity

__all__ = [
    'identity',
    'Stream',
    ]

# Suppress lint warnings about unused symbols.
_ = identity, Stream
