IJssel
======

Stream library for Python.

This is meant to be "a bit like Java streams," but simpler.  It lets you do
similar things: map, filter, count, reduce, and so on -- but in a "fluent"
style.

In standard Python you might write:

    greater_than_10 = lambda value: value > 10
    big_counts = list(filter(greater_than_10, map(count, items)))

But some prefer it more like:

    greater_than = lambda value, threshold: value > threshold
    big_counts = (
        Stream(items)
        .map(count)
        .filter(greater_than, {'threshold': 10})
        .list()
        )

An IJssel stream can iterate anything that you can iterate: ranges, lists,
sets, strings...  Methods that return a sequence also return IJssel streams.


Platform
--------

Works in Python 2.7, 3.6, and pypy.  This will evolve.

Should work on any operating system, as long as it's running a supported Python
version.

External dependencies: minimal.  It'll probably require `six` at some point.
