"""Helpers for the IJssel package."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type
__all__ = [
    'bind_kwargs',
    'identity',
    'ifilter',
    'ifilterfalse',
    'imap',
    'int_types',
    'negate',
    'scan_until',
    'uniq',
    ]

import itertools
from sys import version_info


# Guaranteed-lazy versions of various iteration helpers.
if version_info.major >= 3:
    ifilter = filter
    ifilterfalse = itertools.filterfalse
    imap = map
else:
    ifilter = itertools.ifilter
    ifilterfalse = itertools.ifilterfalse
    imap = itertools.imap


# The integer type(s).  Python 2 has two, Python 3 just one.
if version_info.major >= 3:
    int_types = (int, )
else:
    int_types = (int, long)


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


def uniq(iterable, key):
    """Iterate over iterable, eliminate values with repeat keys.

    The key is a callable which takes the key as its argument.  If a series of
    consecutive items return the same key, only the first of those comes out.
    """
    # A blank object does not equal anything.
    last_key = object()
    for item in iterable:
        item_key = key(item)
        if item_key != last_key:
            last_key = item_key
            yield item
