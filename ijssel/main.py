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
from itertools import (
    chain,
    islice,
    starmap,
    takewhile,
    )
import os.path

from .util import (
    bind_kwargs,
    identity,
    ifilter,
    ifilterfalse,
    imap,
    int_types,
    uniq,
    )


# TODO: Document how map/reduce can combine to compute e.g. averages.
# TODO: How to support parallelism?
# TODO: Document: What happens to original when you iterate a modified stream?
# TODO: next?  Repeatable slicing?

class Stream:
    """Stream class.

    Lets you iterate and aggregate over anything that's iterable, using a
    "fluent" interface.  Any Stream object wraps a sequence of items.

    Streams are iterated lazily except where specified.  So nothing happens
    and your callbacks are not called, until you actually do something to
    pull values from your stream.  A stream will generally not read all of its
    items into memory at once, which can be useful for large data sets, and it
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
    stream.  It may stop if a function you passed raises an exception.  Or
    sometimes the operation can simply complete early.  For example, the `all`
    method can stop as soon as it hits a False item.

    Many methods take both a function and a "kwargs" as parameters.  That's
    shorthand for parameter binding.  To avoid confusion when it comes to
    positional arguments, it binds only keyword arguments.

    The kwargs trick can save you some hard-to-read antics.  For instance, if
    you have a sequence of lists and you want to sort them all in reverse
    order, you can pass the "reverse=True" parameter to each call to `sorted`:

        stream.map(sorted, {'reverse': True})
    """
    def __init__(self, iterable=(), based_on=None):
        """Initialise a new stream.

        :param iterable: Anything that can be iterated: a list, a generator,
            a set, a range, a view, a string.
        :param based_on: Optional original stream on which the new one is
            based.  Normally only used from within the `Stream` class, or in
            classes derived from it.
        """
        # We don't actually use based_on yet.  But having it in the base class
        # will make it easier for derived classes to pass additional
        # attributes down chained streams.
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __getitem__(self, index):
        """Index or slice a stream.

        Indexing retrieves a single item.  This is a terminal operation.

        Slicing limits the stream to a subset.  This is nonterminal.

        :param index: Either an integer index, or a slice.
        :return: If index was an integer index, the item at index.  If index
            was a slice, a Stream.
        :raise IndexError: When passing an integer index that's out of range.
            Negative indexes are considered out of range.
        :raise TypeError: If index is neither an integer nor a slice.
        """
        if isinstance(index, slice):
            start, stop, step = index.start, index.stop, index.step
            return self.evolve(islice(self.iterable, start, stop, step))
        elif isinstance(index, int_types):
            if index < 0:
                raise IndexError("Negative index not supported: %d." % index)
            element_slice = islice(self.iterable, index, index + 1)
            return tuple(element_slice)[0]
        else:
            raise TypeError(
                "Invalid type for Stream indexing/slicing: '%s'." % index)

    def evolve(self, iterable):
        """Create a new instance based on the current one.

        You won't need this in normal use.  If you want to derive your own
        stream class from `Stream`, you may need to override it to pass more
        information from the "original" stream to the "evolved" one.
        """
        return type(self)(iterable)

    def into(self, callee, kwargs=None):
        """Invoke `callee` on the stream's iterable as a whole, return result.

        Use this to call a function or instantiate a class, passing the stream
        as an iterable argument.

        Example: `stream.into(list)` returns the stream's contents as a `list`.
        It's a more "fluent" way of saying `list(stream)`.

        Terminal.

        :return: Whatever callee returns.
        """
        call = bind_kwargs(callee, kwargs)
        return call(self.iterable)

    def apply(self, callee, kwargs=None):
        """Apply callee to iterable, wrap result as new stream.

        The `callee` must be something callable (a function, a class, etc.)
        which takes an iterable as its argument and returns another iterable.
        Of course you can also add keyword arguments through `kwargs`.

        Example: `stream.apply(enumerate)` returns a new stream which applies
        `enumerate` to `stream`.  It will yield tuples of `(number, item)`
        where `number` is an increasing number starting at 0, and `item` is the
        corresponding item from the original stream.

        Example: `stream.apply(sorted)` reads the entier stream into memory
        and returns it as a stream.

        (Unfortunately some standard-library functions take another argument
        before the iterable, and don't accept keyword arguments.  In those
        cases, apply() doesn't do the job, and the stream class will need to
        call `self.evolve(callee(...))` itself.)

        Terminal, if `callee` reads the items in the stream.
        """
        return self.evolve(self.into(callee, kwargs))

    def list(self):
        """Return all items as a list.

        Shorthand for `self.into(list)`.

        Terminal.

        :return: list.
        """
        return self.into(list)

    def count(self):
        """Return number of items.

        Consumes all items.

        Terminal.

        :return: int.
        """
        total = 0
        for _ in self.iterable:
            total += 1
        return total

    def empty(self):
        """Is the stream empty?

        Terminal.  Consumes one item, if there is one.

        :return: bool.
        """
        sentinel = object()
        return next(iter(self), sentinel) is sentinel

    def for_each(self, function, kwargs=None):
        """Execute function(item) for each item.

        You could also write this as `self.map(function, kwargs).drain()`.

        Terminal.
        """
        call = bind_kwargs(function, kwargs)
        for item in self.iterable:
            call(item)

    def drain(self):
        """Consume all items.

        You could also write this as `self.for_each(lambda x: None)`.

        Terminal.
        """
        for _ in self.iterable:
            pass

    def filter(self, criterion=identity, kwargs=None):
        """Drop any items for which criterion(item) is not true.

        :return: Stream.
        """
        return self.evolve(
            ifilter(bind_kwargs(criterion, kwargs), self.iterable))

    def filter_out(self, criterion=identity, kwargs=None):
        """Drop any items for which criterion(item) is true.

        :return: Stream.
        """
        return self.evolve(
            ifilterfalse(bind_kwargs(criterion, kwargs), self.iterable))

    def map(self, function, kwargs=None):
        """Transform stream: apply function to each item.

        :return: Stream.
        """
        return self.evolve(imap(bind_kwargs(function, kwargs), self.iterable))

    def starmap(self, function, kwargs=None):
        """Like map, but each item is a series of arguments.

        Each item should be a list, tuple, or other sequence.  Replaces each
        item with the result of `function(*item)`.

        :return: Stream.
        """
        return self.evolve(
            starmap(bind_kwargs(function, kwargs), self.iterable))

    def catch(self, function, kwargs=None):
        """Iterate exceptions raised by `function(item)` for each item.

        Produces None values for items where no exception was raised.

        Only catches Exception-based exceptions.  All other exceptions are
        simply propagated.

        :return: Stream.
        """
        if kwargs is None:
            kwargs = {}

        def handle(item):
            try:
                function(item, **kwargs)
            except Exception as error:
                return error
            else:
                return None

        return self.map(handle)

    def take_while(self, criterion=identity, kwargs=None):
        """Stop iterating when `criterion(item)` is false.

        :return: Stream.
        """
        return self.evolve(
            takewhile(bind_kwargs(criterion, kwargs), self.iterable))

    def concat(self):
        """Items are themselves sequences.  Iterate them all combined.

        :return: Stream.
        """
        return self.apply(chain.from_iterable)

    def group(self, key=identity, key_kwargs=None, value=identity,
              val_kwargs=None):
        """Map items into a dict of lists.

        For each item, computes a key as key(item) and a value as value(item).
        Returns a dict mapping each key to the list of values computed from
        the items which had that key.

        Within each key's list, the values stay in the same order in which
        they occurred in the original stream.  If the same key/value pair
        occurs twice, the value will be included twice in the list for that
        key.

        Terminal.

        :return: dict.
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

        Shorthand for `self.reduce((lambda l, r: l + r), initial)`.

        Try not to use this on strings.  For that special case, `join` will be
        faster.

        If you're adding floating-point numbers and precision is important to
        your application, simply summing numbers in an arbitrary order may not
        produce the most accurate results.  In floating-point arithmetic,
        adding a very small number to a very large number may not have any
        effect.  So if your stream starts with a very large number, followed by
        a long series of numbers very close to zero, then the near-zero numbers
        may not show up in the result.  For more accurate results you may need
        to sort the numbers in ascending order first, so that the smaller
        numbers build up into a larger one before it gets added to the huge
        number.

        Terminal.

        :return: result of summing initial plus all items.
        """
        add = lambda l, r: l + r
        # Built-in sum function will refuse to run on strings.  Let's not go
        # that far.  We want to present an API that's easy to understand.
        return self.reduce(add, initial=initial)

    def reduce(self, function, initial=None, kwargs=None):
        """Use function to combine items.

        Accumulates a single result by iterating over all items and, for each,
        updating the resulting value, like:

            value = function(value, item)

        Uses `initial` as the starting value.

        Terminal.

        :return: result of repeated invocations of function on initial and all
            items.
        """
        return functools.reduce(
            bind_kwargs(function, kwargs), self.iterable, initial)

    def uniq(self, key=identity, kwargs=None):
        """Filter out any consecutive items with equal `key(item)`.

        If `key` returns the same value for two or more consecutive items
        in the stream, the resulting stream will only contain the first of
        those items.  The other items with the same key are filtered out.
        """
        return self.apply(uniq, {'key': bind_kwargs(key, kwargs)})

    def peek(self, function, kwargs=None):
        """Pass on each item unchanged, but also, run function on it.

        :return: Stream.
        """
        if kwargs is None:
            kwargs = {}

        def process(item):
            function(item, **kwargs)
            return item

        return self.map(process)

    def string_join(self, sep=''):
        """Perform a string join on all items, separated by sep.

        Calls `sep.join(...)` on all items.  Other ways to write this would
        be `self.into(sep.join)` or `sep.join(self)`.

        Terminal.
        """
        return sep.join(self)

    def path_join(self):
        """Join native filesystem path components into a single path.

        Elements must be either byte strings or Unicode strings.  Depending on
        your Python implementation, it may also be an error to mix the two.

        Terminal.
        """
        return os.path.join(*self)

    def sort(self, key=identity, kwargs=None):
        """Return a sorted version of this stream.

        Reads all items into memory, sorts them, and returns a new Stream
        which iterates over the sorted items.

        Terminal.

        :param key: Compute key by which elements should be sorted.
        :return: Stream.
        """
        return self.apply(sorted, {'key': bind_kwargs(key, kwargs)})
