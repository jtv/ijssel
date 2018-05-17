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


Name
----

The IJssel is a large stream in The Netherlands.

Well, technically it's a river.  But work with me here.

And you saw that right: _IJssel,_ not _Ijssel._  The "IJ" is basically the
Dutch version of "Y", so think _Yssel._  If you pronounce it Ice-sel you'll be
close enough.

Unicode also has a ligature for that combo: Ĳssel.  That way it's 5
characters, not 6.


Platform
--------

Works in Python 2.7, 3.6, and pypy.  This will evolve.

Should work on any operating system, as long as it's running a supported Python
version.

External dependencies: minimal.  It'll probably require `six` at some point.
