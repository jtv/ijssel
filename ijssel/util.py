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
    'int_types',
    'negate',
    'scan_until',
    'uniq',
    ]

import itertools
from sys import version_info


# Guaranteed-lazy version of filter.
if version_info.major >= 3:
    ifilter = filter
else:
    ifilter = itertools.ifilter


# Guaranteed-lazy version of map.
if version_info.major >= 3:
    imap = map
else:
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
