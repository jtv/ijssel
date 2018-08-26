"""Exception types raised by IJssel."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type
__all__ = [
    'NotIterable',
    ]


class NotIterable(TypeError):
    """A Stream was constructed from a non-iterable object."""
