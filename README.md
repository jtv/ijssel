IJssel
======

"Fluent" streams library for Python.

IJssel gives you something like Java's streams library, but simpler.  It lets
you do similar things: map, filter, count, reduce, and so on, in a "fluent"
style.

Without IJssel you might write:

    greater_than_3 = lambda value: value > 3
    big_counts = list(filter(greater_than_3, map(count, items)))

But some people prefer it more like:

    greater_than = lambda value, threshold: value > threshold
    big_counts = (
        Stream(items)
        .map(count)
        .keep_if(greater_than, {'threshold': 3})
        .list()
        )

An IJssel stream can iterate anything that you can iterate: ranges, lists,
sets, strings...  Methods that return a sequence also return IJssel streams.

Of course you can also iterate an IJssel stream just like any other sequence:

    # Print the numbers 0, 1, 2.
    for item in Stream(range(3)):
        print(item)

That means that you can normally use an IJssel stream wherever you can use an
iterable.

Streams iterate lazily.  A stream won't process a single item until you ask for
one.  An infinite stream will work just fine, so long as you don't try to read
all of its items.


Name
----

The IJssel is a large stream in The Netherlands.

Well, technically it's a river.  But work with me here.

And you saw that right: _IJssel,_ not _Ijssel._  The "IJ" is basically the
Dutch version of "Y", so think _Yssel._  If you pronounce it Ice-sel you'll be
close enough.

Unicode also has that combo as a single character: Ĳssel.  That's only
five characters, not the six you'd expect.


Platform
--------

Works in Python 2.7, 3.6, and pypy.  This will evolve.

Should work on any operating system, as long as it's on a supported Python
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


### Processing steps

So things start to happen once you invoke a terminal operation at the "back"
of your chain.  How does that go?

First, it pulls an item from your original iterable.  The item travels through
the stages of your chain, from left to right.  Each stage in the chain can
perform some transformation on the item, delivering the resulting item to the
next stage.

Some stage in the chain, such as a `drop_if`, may decide to drop the item
from its output stream.  If that happens, the item is not enough to give the
terminal operation at the back the item that it asked for.  So the chain goes
back to the beginning, pulling another item from your original iterable, and
running it through the stages.  It keeps doing this until it can deliver a
processed item, or until the iterable runs out.

Assuming the iterable didn't run out, a processed item comes out of the chain,
and gets consumed by the terminal operation.

Next, the whole thing is repeated for the next item, and so on until either the
iterable runs out or the operation at the back of the chain stops asking for
more items.


### Transforming each item in a stream

Call a stream's `map` with any callable that takes an item, and you get a new
stream where each of the original stream's items is replaced with the result
of your callable:

    double = lambda x: 2 * x
    assert Stream(range(3)).map(double).list() == [0, 2, 4]

The transformation is still "lazy," so `double` isn't actually executed until
you ask for the `.list()`.

If you use your own Stream subclass, the result will be a new instance of
your subclass.


### Transforming a stream

Call a stream's `apply` to run the entire stream through a function which
returns a new iterable, and then wrap that result in a new stream:

    stream = Stream(range(3))
    sorted_stream = stream.apply(sorted, {'reverse': True})
    assert isinstance(sorted_stream, Stream)
    assert sorted_stream.list() == [2, 1, 0]

The `sorted` built-in reads the whole stream, and returns a list.  The `apply`
call wraps this in a new stream.

If you use your own Stream subclass, the result will be a new instance of
your subclass.


### Feeding your stream into a callable

You can pass your stream's contents into a function, a lambda, constructor, or
any other kind of callable, and get its result, call `into`.  This is just
like calling your callable and passing the stream's iterable as an argument,
but in a more "fluent" notation:

    assert Stream(range(3)).into(list) == list(Stream(range(3)))

In fact, the `list` method is a convenience shorthand for this call:

    assert Stream(range(3)).list() == Stream(range(3)).into(list)


Differences from Java streams
-----------------------------

If you're used to Java streams, here are a few things you may miss in IJssel.
In many cases there's a simple alternative.


### No parallelism

Future versions of IJssel may be able to distribute jobs to a pool of worker
threads, processes, or servers.

At that point, a lot of things will have to become more complicated.  You'll
want a range of different mechanisms for dispatching asynchronous jobs.
There's two dimensions for parallelism: you may want to implement each stage in
a pipeline as a separate thread or process, or you may want an individual stage
in the pipeline to farm out its work to a pool of workers.  Stages may share
pools, or use different mechanisms.

Keeping items in their original order will be harder, so it will become
optional.  And, I'll have to figure out whether all these things should be
settings on the stream (and get passed along from onge stage to the next), or
they should be individually configurable on each stage (for that one expensive
calculation in a pipeline of mostly cheap steps), or both.


### flatMap

Instead, just combine `map` and `concat`:

    lines = [
        "I have seen things you people wouldn't believe.",
        "Attack ships on fire off the shoulder of Orion.",
        "I watched C-beams glitter in the dark off the Tannhauser Gate.",
        "All those moments will be lost...",
        "Like tears in rain.",
        "Time to die.",
        ]

    words = Stream(lines).map(methodcaller('split')).concat().list()

    assert words == ' '.join(lines).split()


### skip and limit

Use standard Python slicing!

Instead of `stream.skip(10)`, say `stream[10:]`.  Instead of `stream.limit(5)`,
say `stream[:5]`.  And instead of `stream.skip(10).limit(5)`, say
`stream[10:15]`.

Of course these return streams, so if you just want a list with those
elements, call `list`:

    stream[10:15].list()


### concat

IJssel does have a `concat`, but it does something different.

To concatenate two or more streams, just add them!

    combined = Stream([0, 1]) + Stream([2, 3]) + Stream([4, 5])
    assert combined.list() == [0, 1, 2, 3, 4, 5]


### empty

IJssel's `empty` tests whether a stream is empty.

To get an empty stream, just create it without an argument: `Stream()`.


### generate

The IJssel equivalent is to use `itertools.count` as your stream's initialiser:

    Stream(count())[0:5]  # Counts from 0: 0, 1, 2, 3, 4.
    Stream(count(1, 2)[0:5]  # Counts odd numbers: 1, 3, 5, 7, 9.


### min and max

To get the minimum or maximum item in a stream, pump its items into the
built-in `min` or `max` functions, respectively: `stream.into(max)`.


### noneMatch

To check whether none of the items meets some criterion, use:

    not stream.any(criterion)
