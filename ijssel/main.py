"""Stream-like API for Python."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type
__all__ = [
    'Stream',
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


# TODO: What happens to original when you iterate a modified stream?
class Stream:
    """Stream class.

    Lets you iterate and aggregate over anything that's iterable, using a
    "fluent" interface.  Any Stream object wraps a sequence of items.

    Streams are iterated lazily except where specified.  So nothing happens
    and your callbacks are not called, until you actually do something to
    pull values from your stream.  A stream will not read all of its items
    into memory at once, which can be useful for large data sets, and it
    won't call any of the functions you pass on items until it really has to.
    You can even have infinite streams, such as one that generates digits of Pi
    on demand.

    It also means that you can't just call e.g. `map` or `filter` with a
    function that does something, and expect it to operate on all your items.
    Your function only gets called as an item is actually read from the stream.
    There's a `drain` method for "go through all items now."

    This is an example of a "terminal" operation: it tries to read all items
    right away, instead of iterating them one by one.  Usually the operations
    that return a stream are non-terminal, meaning they continue to iterate
    lazily.  Operations that don't return a stream are usually terminal.  For
    instance, if you ask for the sum of the elements in a stream, the operation
    is going to have to read all of the items right there.

    Sometimes a terminal operation doesn't really consume all items in the
    stream.  It may stop if a function you passed raises an exception.  But in
    other cases, the operation can simply complete early.  For example, the
    "all" method can stop as soon as it hits a False item.

    Many methods take both a function and a "kwargs" as parameters.  That's
    shorthand for parameter binding.  To avoid confusion with positional
    arguments, it binds only keyword arguments.

    The kwargs trick can save you some hard-to-read antics.  For instance, if
    you have a sequence of lists and you want to sort them all in reverse
    order, you can pass the "reverse=True" parameter to each call:

        stream.map(sorted, {'reverse': True})
    """
    def __init__(self, iterable=()):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def list(self):
        """Return all items as a list.

        Terminal.
        """
        return list(self.iterable)

    def tuple(self):
        """Return all items as a tuple.

        Terminal.
        """
        return tuple(self.iterable)

    def all(self):
        """Return bool: Is each item true?

        If there are no items, then the answer is True.

        Terminal.  But, if there is a false item, stops right after consuming
        that item.
        """
        return all(self.iterable)

    def any(self):
        """Return bool: Is any item true?

        If there are no items, then the answer is False.

        Terminal.  But, if there is a true item, stops right after consuming
        that item.
        """
        return any(self.iterable)

    def negate(self):
        """Logically negate each item."""
        return self.map(negate)

    def count(self):
        """Return number of items.

        Consumes all items.

        Terminal.
        """
        total = 0
        for _ in self.iterable:
            total += 1
        return total

    def empty(self):
        """Is the stream empty?

        Terminal.  Consumes one item, if there is one.
        """
        sentinel = object()
        return next(iter(self), sentinel) is sentinel

    def for_each(self, function, kwargs=None):
        """Execute function(item) for each item.

        Terminal.
        """
        call = bind_kwargs(function, kwargs)
        for item in self.iterable:
            call(item)

    def drain(self):
        """Consume all items.

        Terminal.
        """
        for _ in self.iterable:
            pass

    def filter(self, criterion=identity, kwargs=None):
        """Drop any items for which criterion(item) is not true."""
        return Stream(ifilter(bind_kwargs(criterion, kwargs), self.iterable))

    def filter_out(self, criterion=identity, kwargs=None):
        """Drop any items for which criterion(item) is true."""
        call = bind_kwargs(criterion, kwargs)
        return Stream(item for item in self.iterable if not call(item))

    def map(self, function, kwargs=None):
        """Transform stream: apply function to each item."""
        return Stream(imap(bind_kwargs(function, kwargs), self.iterable))

    def limit(self, limit):
        """Iterate only the first limit items."""
        return Stream(head(self.iterable, limit))

    def until_value(self, sentinel):
        """Iterate items until an item equals sentinel.
        """
        return Stream(
            scan_until(self.iterable, lambda item: item == sentinel))

    def until_identity(self, sentinel):
        """Iterate items until until an item is the sentinel object."""
        return Stream(
            scan_until(self.iterable, lambda item: item is sentinel))

    def until_true(self, criterion, kwargs=None):
        """Stop iterating when criterion(item) is true."""
        return Stream(
            scan_until(self.iterable, bind_kwargs(criterion, kwargs)))

    def while_true(self, criterion, kwargs=None):
        """Stop iterating when criterion(item) is false."""
        call = bind_kwargs(criterion, kwargs)
        return self.until_true(lambda item: not call(item))

    def concat(self):
        """Items are themselves sequences.  Iterate them all combined."""
        return Stream(chain.from_iterable(self.iterable))

    def group(self, key=identity, key_kwargs=None, value=identity,
              val_kwargs=None):
        """Map items into a dict of lists.

        For each item, computes a key as key(item) and a value as value(item).
        Returns a dict mapping each key to the list of values computed from
        the items which had that key.

        Within each key's list, the values stay in the same order in which
        they occurred in the original stream.

        Terminal.
        """
        key_call = bind_kwargs(key, key_kwargs)
        value_call = bind_kwargs(value, val_kwargs)
        groups = {}
        for item in self.iterable:
            item_key = key_call(item)
            item_value = value_call(item)
            groups.setdefault(item_key, [])
            groups[item_key].append(item_value)
        return groups

    def sum(self, initial=0):
        """Add up all elements, starting with initial value.

        Terminal.
        """
        add = lambda l, r: l + r
        return self.reduce(add, initial=initial)

    def reduce(self, function, initial=None, kwargs=None):
        """Use function to combine items.

        Accumulates a single result by iterating over all items and, for each,
        updating the resulting value, like:

            value = function(value, item)

        Uses `initial` as the starting value.

        Terminal.
        """
        return functools.reduce(
            bind_kwargs(function, kwargs), self.iterable, initial)

# TODO: peek
# TODO: join
# TODO: path_join
# TODO: average, mean?
# TODO: catch
# TODO: get
