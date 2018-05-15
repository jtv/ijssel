"""Stream-like API for Python."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type
__all__ = [
    'Streamless',
    ]

import functools
from itertools import chain

from .util import (
    bind_kwargs,
    head,
    identity,
    ifilter,
    imap,
    merge_dicts,
    negate,
    scan_until,
    )


class Streamless:
    """Stream class.

    Lets you iterate and aggregate over anything that's iterable, using a
    "fluent" interface.  Any Streamless object wraps a sequence of items.

    Streams are iterated lazily except where specified.  That means that there
    are many things you can safely do on infinite streams, or recover from
    errors halfway through.

    Many methods take both a function and a "kwargs" as parameters.  That's
    shorthand for parameter binding.  To avoid confusion with positional
    arguments, it binds only keyword arguments.

    The kwargs trick can save you some hard-to-read antics.  For instance, if
    you have a sequence of lists and you want to sort them all in reverse
    order, you can pass the "reverse=True" parameter to each call:

        stream.map(sorted, {'reverse': True})

    If an exception occurs
    """
    def __init__(self, iterable=()):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def list(self):
        """Consume all items, return as list."""
        return list(self.iterable)

    def tuple(self):
        """Consume all items, return as tuple."""
        return tuple(self.iterable)

    def all(self):
        """Return bool: Is each item true?

        If there are no items, then the answer is True.

        Consumes all items.
        """
        return all(self.iterable)

    def any(self):
        """Return bool: Is any item true?

        If there are no items, then the answer is False.

        Consumes items up to and including the first which is not true, or
        if all are true, consumes all items.
        """
        return any(self.iterable)

    def negate(self):
        """Logically negate each item."""
        return self.map(negate)

    def count(self):
        """Return number of items.

        Consumes all items.
        """
        total = 0
        for _ in self.iterable:
            total += 1
        return total

    def empty(self):
        """Is the stream empty?

        Consumes one item, if there is one.
        """
        sentinel = object()
        return next(iter(self), sentinel) is sentinel

    def for_each(self, function, kwargs=None):
        """Execute function(item) for each item.

        Consumes all items.
        """
        call = bind_kwargs(function, kwargs)
        for item in self.iterable:
            call(item)

    def drain(self):
        """Consume all items."""
        for _ in self.iterable:
            pass

    def filter(self, criterion=identity, kwargs=None):
        """Drop any items for which criterion(item) is not true."""
        return Streamless(ifilter(bind_kwargs(criterion, kwargs), self.iterable))

    def filter_out(self, criterion=identity, kwargs=None):
        """Drop any items for which criterion(item) is true."""
        call = bind_kwargs(criterion, kwargs)
        return Streamless(item for item in self.iterable if not call(item))

    def map(self, function, kwargs=None):
        """Transform stream: apply function to each item."""
        return Streamless(imap(bind_kwargs(function, kwargs), self.iterable))

    def limit(self, limit):
        """Iterate only the first limit items."""
        return Streamless(head(self.iterable, limit))

    def until_value(self, sentinel):
        """Iterate items until an item equals sentinel."""
        return Streamless(
            scan_until(self.iterable, lambda item: item == sentinel))

    def until_identity(self, sentinel):
        """Iterate items until until an item is the sentinel object."""
        return Streamless(
            scan_until(self.iterable, lambda item: item is sentinel))

    def until_true(self, criterion, kwargs=None):
        """Stop iterating when criterion(item) is true."""
        return Streamless(
            scan_until(self.iterable, bind_kwargs(criterion, kwargs)))

    def while_true(self, criterion, kwargs=None):
        """Stop iterating when criterion(item) is false."""
        call = bind_kwargs(criterion, kwargs)
        return self.until_true(lambda item: not call(item))

    def concat(self):
        """Items are themselves sequences.  Iterate them all combined."""
        return Streamless(chain.from_iterable(self.iterable))

    def partition(self, key=identity, key_kwargs=None, value=identity,
                  val_kwargs=None):
        """Map items into a dict of lists.

        For each item, computes a key as key(item) and a value as value(item).
        Returns a dict mapping each key to the list of values computed from
        the items which had that key.

        Within each key's list, the values stay in the same order in which
        they occurred in the original stream.
        """
        key_call = bind_kwargs(key, key_kwargs)
        value_call = bind_kwargs(value, val_kwargs)
        partitioning = {}
        for item in self.iterable:
            item_key = key_call(item)
            item_value = value_call(item)
            partitioning.setdefault(item_key, [])
            partitioning[item_key].append(item_value)
        return partitioning

    def reduce(self, function, initial=None, kwargs=None):
        """Use function to combine items.

        Accumulates a single result by iterating over all items and, for each,
        updating the resulting value, like:

            value = function(value, item)

        If initial is not None, it is used as the initial value.

        Consumes all items.
        """
        return functools.reduce(
            bind_kwargs(function, kwargs), self.iterable, initial)

    def sum(self, initial=0):
        """Add up all elements, starting with initial value.

        Consumes all items.
        """
        return sum(self.iterable, initial)
