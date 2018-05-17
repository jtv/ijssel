"""Helpers for the IJssel package."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type
__all__ = [
    'bind_kwargs',
    'head',
    'identity',
    'ifilter',
    'imap',
    'negate',
    'scan_until',
    ]

import itertools
from sys import version_info


if version_info.major >= 3:
    ifilter = filter
else:
    ifilter = itertools.ifilter


if version_info.major >= 3:
    imap = map
else:
    imap = itertools.imap


def bind_kwargs(function, kwargs=None):
    """Bind keyword arguments to function.

    Returns a callable, which when called, in turn calls function.  The call
    adds keyword arguments as specified.

    It passes on any positional arguments to function, as well as the given
    keyword arguments.

    Any keyword arguments passed into the call are currently ignored, and
    `**kwargs` used instead.  This may change.

    :param function: Any callable.
    :param kwargs: Keyword arguments, or None.
    :return: A callable.
    """
    if kwargs is None or kwargs == {}:
        return function
    else:
        return lambda *args: function(*args, **kwargs)


def identity(arg):
    """Return arg."""
    return arg


def negate(arg):
    """Return the Boolean negation of arg."""
    return not arg


def head(iterable, limit):
    """Iterate over at most the first limit items in iterable."""
    sentinel = object()
    iterator = iter(iterable)
    count = 0
    item = None
    while count < limit:
        item = next(iterator, sentinel)
        if item is sentinel:
            count = limit
        else:
            yield item
            count += 1


def scan_until(iterable, criterion):
    """Iterate over iterable until criterion(item) is true."""
    sentinel = object()
    iterator = iter(iterable)
    item = next(iterator, sentinel)
    while item is not sentinel:
        if criterion(item):
            return
        yield item
        item = next(iterator, sentinel)
