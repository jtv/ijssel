streamless
==========

Stream library for Python.

This is meant to be "a bit like Java streams," but simpler.  It lets you do
similar things: map, filter, count, reduce, and so on -- but in a "fluent"
style.

In standard Python you might write:

    big_counts = filter(greater_than_10, map(count, items))

But some prefer something more like:

    big_counts = Streamless(items).map(count).filter(greater_than_10)


Python versions
---------------

Works in Python 2.7, 3.6, and pypy.  This will evolve.
