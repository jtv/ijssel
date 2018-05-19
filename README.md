IJssel
======

Stream library for Python.

This is meant to be "a bit like Java streams," but simpler.  It lets you do
similar things: map, filter, count, reduce, and so on, in a "fluent" style.

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


Use
---

The essential API is in the ijssel.Stream class.  You wrap any kind of
sequence in a Stream, call a method on it.  In many cases it will return
another Stream, on which you can invoke the next method and so on.

It's a lot like shell pipes.  For instance, here's how in a Unix shell you
might sort the lines in a text file, eliminate the duplicates, and count how
many you've got left:

    cat file.txt | sort | uniq | wc -l

(Yes, there's better ways of doing that.  Not the point.)

Here's how you might do it in Python with IJssel:

    with open('file.txt') as lines:
        count = Stream(lines).sort().uniq().count()
    print(count)

So the idea is that you build chains, or pipelines, of operations.  Each step
in the chain transforms the stream in some way, or applies some action to each
item in the stream.  Many of the methods on the stream return another stream,
based on the previous one in the chain but transformed in some way.

The chains don't actually do anything though, until you call a _terminal_
method.  These are the methods which actually consume items from the stream.
Iterating a stream is a terminal operation, as is sorting it, counting its
number of elements, and so on.  Those things can't be done without processing
items from the stream.  Until you call a terminal operation, you're just
setting up operations but not triggering them yet.

As a terminal operation at the "back" of your chain pulls an item from the
stream, the item goes through the stages in your chain from left to right.
Then all of the same steps happen for the next item in the stream, and so on.
