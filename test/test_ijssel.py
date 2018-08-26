"""Tests for IJssel Stream class."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type

from io import StringIO
from itertools import count
from operator import methodcaller
import os.path
from textwrap import dedent
from unittest import TestCase

from .factory import (
    make_list,
    make_num,
    )
from ..ijssel import Stream
from ..ijssel.exceptions import NotIterable
from ..ijssel.util import identity


# Standard things to test:
#
# Method that returns streams -
#  * Does not consume items without terminal call.
#  * Propagates exception.
#  * Stops at exception.
#  * Does not consume beyond exception.
#
# Method with a "stop" condition -
#  * Stops at condition.
#  * Does not consume items beyond condition.
#
# Method with a callback -
#  * Adds kwargs.


def generate(sequence, iterations):
    """Yield items from sequence, also appending them to iterations.

    Use this to test which items a stream actually consumes.  Only those
    items which are actually consumed are appended to iterations.
    """
    for item in sequence:
        iterations.append(item)
        yield item


def recorder(processed_items, function=identity):
    """Wrap a callable to record processed items.

    Returns a new callable which takes an item as its argument, appends the
    item to `processed_items`, and returns `function(item)`.
    """
    def process(item):
        processed_items.append(item)
        return function(item)

    return process


def kwargs_recorder(kwargs_list, function=identity, **kwargs):
    """Wrap a callable to record kwargs passed to function.

    Returns a new callable which takes an item as its argument, plus any
    keyword arguments.  It appends the dict of keyword arguments to
    kwargs_list, and returns `function(item)`.
    """
    def process(item, **kwargs):
        kwargs_list.append(kwargs)
        return function(item)

    return process


class SimulatedFailure(Exception):
    """Deliberate failure for test purposes."""


def fail_item(item):
    """Simulate failure when processing item."""
    raise SimulatedFailure("Simulated failure for item '%s'." % item)


class TestIteration(TestCase):
    """Tests for basic iteration."""
    def test_iterates_container(self):
        n = make_num(1)
        self.assertEqual(list(Stream(range(n))), list(range(n)))
        self.assertEqual(list(Stream(list(range(n)))), list(range(n)))
        self.assertEqual(list(Stream(tuple(range(n)))), list(range(n)))
        self.assertEqual(list(Stream({n: 'foo'})), [n])
        self.assertEqual(list(Stream('abc')), list('abc'))

    def test_iterates_empty_container(self):
        self.assertEqual(list(Stream(range(0))), [])
        self.assertEqual(list(Stream([])), [])
        self.assertEqual(list(Stream(tuple())), [])
        self.assertEqual(list(Stream({})), [])
        self.assertEqual(list(Stream('')), [])

    def test_defaults_to_empty_stream(self):
        self.assertEqual(list(Stream()), [])

    def test_requires_iterable_or_None(self):
        self.assertRaises(NotIterable, Stream, None)
        self.assertRaises(NotIterable, Stream, 10)
        self.assertRaises(NotIterable, Stream, 0.1)
        self.assertRaises(NotIterable, Stream, int)

    def test_iterates_sequence(self):
        n = make_num(1)

        def generate():
            for item in range(n):
                yield item

        self.assertEqual(list(Stream(generate())), list(range(n)))

    def test_iterates_range(self):
        self.assertEqual(list(Stream(range(3))), [0, 1, 2])

    def test_lazy(self):
        iterations = []
        sequence = iter(Stream(generate(range(5), iterations)))
        self.assertEqual(iterations, [])
        first = next(sequence)
        self.assertEqual(first, 0)
        self.assertEqual(iterations, [0])

        second = next(sequence)
        self.assertEqual(second, 1)
        self.assertEqual(iterations, [0, 1])

        self.assertEqual(list(sequence), [2, 3, 4])
        self.assertEqual(iterations, [0, 1, 2, 3, 4])


class TestNext(TestCase):
    """Tests for `next` iteration."""
    def test_gets_next_item(self):
        original = make_list()
        stream = Stream(original)
        self.assertEqual([next(stream) for _ in original], original)

    def test_ends_with_StopIteration(self):
        items = make_list()
        stream = Stream(items)
        for _ in items:
            next(stream)

        self.assertRaises(StopIteration, next, stream)


class TestGetItemIndex(TestCase):
    """Tests for `__getitem__` (when passing an index)."""
    def test_returns_item_from_stream(self):
        items = make_list()
        self.assertEqual(
            [Stream(items)[index] for index, _ in enumerate(items)],
            items)

    def test_works_on_generator(self):
        def iterate(inputs):
            for item in inputs:
                yield item

        inputs = make_list()
        index = make_num(max=(len(inputs) - 1))

        self.assertEqual(
            Stream(iterate(inputs))[index],
            inputs[index])

    def test_raises_IndexError_if_out_of_range(self):
        access = lambda stream, index: stream[index]
        self.assertRaises(IndexError, access, Stream([0, 1, 2]), 3)

    def test_raises_IndexError_if_negative(self):
        access = lambda stream, index: stream[index]
        self.assertRaises(IndexError, access, Stream([0, 1, 2]), -1)

    def test_raises_TypeError_if_not_integer(self):
        access = lambda stream, index: stream[index]
        self.assertRaises(TypeError, access, Stream([0, 1, 2]), '0')
        self.assertRaises(TypeError, access, Stream([0, 1, 2]), 0.0)
        self.assertRaises(TypeError, access, Stream([0, 1, 2]), None)
        self.assertRaises(TypeError, access, Stream([0, 1, 2]), object())


class TestGetItemSlice(TestCase):
    """Tests for `__getitem__` (when passing a slice)."""
    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations))[1:3]
        self.assertEqual(iterations, [])

    def test_returns_stream(self):
        self.assertTrue(isinstance(Stream(range(5))[1:3], Stream))

    def test_retrieves_from_start_to_stop(self):
        self.assertEqual(Stream(range(10))[4:8].list(), [4, 5, 6, 7])

    def test_retrieves_from_start_to_end(self):
        self.assertEqual(Stream(range(10))[7:].list(), [7, 8, 9])

    def test_retrieves_from_beginning_to_stop(self):
        self.assertEqual(Stream(range(10))[:4].list(), [0, 1, 2, 3])

    def test_retrieves_from_beginning_to_end(self):
        self.assertEqual(Stream(range(5))[:].list(), [0, 1, 2, 3, 4])

    def test_honours_step(self):
        self.assertEqual(Stream(range(10))[::3].list(), [0, 3, 6, 9])
        self.assertEqual(Stream(range(10))[1:6:2].list(), [1, 3, 5])
        self.assertEqual(Stream(range(10))[:4:2].list(), [0, 2])
        self.assertEqual(Stream(range(10))[5::2].list(), [5, 7, 9])

    def test_does_not_iterate_beyond_stop(self):
        iterations = []
        Stream(generate(range(5), iterations))[:3].drain()
        self.assertEqual(iterations, [0, 1, 2])


class TestAdd(TestCase):
    """Tests for stream addition."""
    def test_returns_stream(self):
        self.assertEqual(type(Stream() + Stream()), Stream)

    def test_empty_stream_adds_nothing(self):
        self.assertEqual((Stream([1, 2]) + Stream()).list(), [1, 2])
        self.assertEqual((Stream() + Stream([1, 2])).list(), [1, 2])

    def test_adds_multiple_streams(self):
        self.assertEqual(
            (Stream([1]) + Stream([2]) + Stream([3])).list(),
            [1, 2, 3])

    def test_adds_container_to_stream(self):
        combined = Stream([1]) + [2]
        self.assertEqual(type(combined), Stream)
        self.assertEqual(combined.list(), [1, 2])

    def test_adds_generator_to_stream(self):
        def generate(start, stop):
            for item in range(start, stop):
                yield item

        combined = Stream([1, 2]) + generate(3, 5)
        self.assertEqual(type(combined), Stream)
        self.assertEqual(combined.list(), [1, 2, 3, 4])

    def test_lazy(self):
        iterations = []
        combined = (
            Stream(generate(range(0, 3), iterations)) +
            Stream(generate(range(3, 6), iterations))
            )
        middle = combined[2:4].list()
        self.assertEqual(middle, [2, 3])
        self.assertEqual(iterations, [0, 1, 2, 3])


class TestApply(TestCase):
    """Tests for `apply`."""
    def test_returns_stream(self):
        inputs = make_list()
        self.assertTrue(isinstance(Stream(inputs).apply(list), Stream))

    def test_applies_function_to_iterable(self):
        self.assertEqual(
            Stream("Hello").apply(enumerate).list(),
            [(0, 'H'), (1, 'e'), (2, 'l'), (3, 'l'), (4, 'o')])


class TestList(TestCase):
    """Tests for `list`."""
    def test_returns_empty_for_empty_stream(self):
        empty = Stream().list()
        self.assertEqual(empty, [])
        self.assertIs(type(empty), list)

    def test_returns_items_as_list(self):
        self.assertEqual(Stream(range(3)).list(), [0, 1, 2])

    def test_turns_generator_into_list(self):
        item = make_num()

        def iterate():
            yield item

        self.assertEqual(Stream(iterate()).list(), [item])


class TestCount(TestCase):
    """Tests for `count`."""
    def test_returns_zero_if_empty(self):
        self.assertEqual(Stream().count(), 0)

    def test_counts_items(self):
        values = make_list()
        self.assertEqual(Stream(values).count(), len(values))

    def test_counts_even_false_items(self):
        self.assertEqual(Stream([None]).count(), 1)


class TestEmpty(TestCase):
    """Tests for `empty`."""
    def test_True_if_empty(self):
        self.assertTrue(Stream().empty())
        self.assertTrue(Stream([]).empty())
        self.assertTrue(Stream(range(0)).empty())

    def test_False_if_nonempty(self):
        self.assertFalse(Stream([False]).empty())
        self.assertFalse(Stream(range(3)).empty())

    def test_consumes_only_one_item(self):
        iterations = []
        empty = Stream(generate(range(3), iterations)).empty()
        self.assertFalse(empty)
        self.assertEqual(iterations, [0])

    def test_works_for_iterator(self):
        self.assertTrue(Stream(iter([])).empty())


class TestForEach(TestCase):
    """Tests for `for_each`."""
    def test_does_nothing_if_empty(self):
        processed = []
        Stream().for_each(recorder(processed))
        Stream([]).for_each(recorder(processed))
        self.assertEqual(processed, [])

    def test_calls_function_on_each_item(self):
        processed = []
        inputs = make_list()
        Stream(inputs).for_each(recorder(processed))
        self.assertEqual(processed, inputs)

    def test_adds_kwargs(self):
        args = []
        n = make_num(1, 3)
        Stream(range(n)).for_each(
            kwargs_recorder(args), kwargs={'kwarg': 'foo'})
        self.assertEqual(args, [{'kwarg': 'foo'}] * n)

    def test_stops_at_exception(self):
        iterations = []
        # This will deliberately fail when item == 3.
        compute = lambda item: 100 / (item - 3)
        stream = Stream(generate([1, 2, 3, 4, 5], iterations))
        self.assertRaises(ZeroDivisionError, stream.for_each, compute)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


class TestDrain(TestCase):
    """Tests for `drain`."""
    def test_consumes_all_items(self):
        iterations = []
        inputs = make_list()
        Stream(generate(inputs, iterations)).drain()
        self.assertEqual(iterations, inputs)


class TestKeepIf(TestCase):
    """Tests for `keep_if`."""
    def test_filters_on_identity_by_default(self):
        # Some values that alternately evaluate as "true" and "false".
        values = [True, False, 0, 1, [], [0], {}, {0: 0}, '', '0']
        # You only get the "true" ones.
        self.assertEqual(
            Stream(values).keep_if().list(),
            [item for item in values if item])

    def test_filters_on_criterion(self):
        even = lambda item: item % 2 == 0
        self.assertEqual(
            Stream([0, 1, 2, 3, 4]).keep_if(even).list(),
            [0, 2, 4])

    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations)).keep_if()
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        arg = make_num()
        criterion = kwargs_recorder(args, lambda item: item > 3)
        length = 5
        stream = Stream(range(length))
        result = stream.keep_if(criterion, kwargs={'kwarg': arg}).list()

        self.assertEqual(result, [4])
        self.assertEqual(args, [{'kwarg': arg}] * length)

    def test_stops_at_exception(self):
        iterations = []
        # This will deliberately fail when item == 3.
        criterion = lambda item: 100 / (item - 3)
        stream = Stream(generate([1, 2, 3, 4, 5], iterations))
        self.assertRaises(
            ZeroDivisionError,
            stream.keep_if(criterion).drain)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


class TestDropIf(TestCase):
    """Tests for `drop_if`."""
    def test_filters_on_identity_by_default(self):
        # Some values that alternately evaluate as "true" and "false".
        values = [True, False, 0, 1, [], [0], {}, {0: 0}, '', '0']
        # You only get the "false" ones.
        self.assertEqual(
            Stream(values).drop_if().list(),
            [item for item in values if not item])

    def test_filters_on_criterion(self):
        even = lambda item: item % 2 == 0
        self.assertEqual(
            Stream([0, 1, 2, 3, 4]).drop_if(even).list(),
            [1, 3])

    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations)).drop_if()
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        arg = make_num()
        criterion = kwargs_recorder(args, lambda item: item > 3)
        length = 5
        stream = Stream(range(length))
        result = stream.drop_if(criterion, kwargs={'kwarg': arg}).list()

        self.assertEqual(result, [0, 1, 2, 3])
        self.assertEqual(args, [{'kwarg': arg}] * length)

    def test_stops_at_exception(self):
        iterations = []
        # This will deliberately fail when item == 3.
        criterion = lambda item: 100 / (item - 3)
        stream = Stream(generate([1, 2, 3, 4, 5], iterations))
        self.assertRaises(
            ZeroDivisionError,
            stream.drop_if(criterion).drain)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


class TestMap(TestCase):
    """Tests for `map`."""
    def test_applies_function(self):
        double = lambda item: 2 * item
        self.assertEqual(
            Stream(range(3)).map(double).list(),
            [0, 2, 4])

    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations)).map(identity)
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        stream = Stream([[1, 2, 3, 4]])
        self.assertEqual(
            stream.map(sorted, kwargs={'reverse': True}).list(),
            [[4, 3, 2, 1]])

    def test_stops_at_exception(self):
        iterations = []
        # This will deliberately fail when item == 3.
        function = lambda item: 100 / (item - 3)
        stream = Stream(generate([1, 2, 3, 4, 5], iterations))
        self.assertRaises(
            ZeroDivisionError,
            stream.map(function).drain)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


class TestStarMap(TestCase):
    """Tests for `starmap`."""
    def test_does_nothing_if_empty(self):
        processed = []
        result = Stream().starmap(recorder(processed)).list()
        self.assertEqual(result, [])
        self.assertEqual(processed, [])

    def test_expands_argument_tuple(self):
        processed = []
        inputs = [(1,), (2,), (3,)]
        result = Stream(inputs).starmap(recorder(processed)).list()
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(processed, [1, 2, 3])

    def test_expands_argument_list(self):
        processed = []
        inputs = [[1], [2], [3]]
        result = Stream(inputs).starmap(recorder(processed)).list()
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(processed, [1, 2, 3])

    def test_expands_any_number_of_arguments(self):
        process = lambda *args: '/'.join('%s' % arg for arg in args)
        inputs = [
            (9, 8, 7),
            (0,),
            [],
            ]
        self.assertEqual(
            Stream(inputs).starmap(process).list(),
            ['9/8/7', '0', ''])

    def test_lazy(self):
        process = lambda *args: 1
        iterations = []
        Stream(generate([[1]], iterations)).starmap(process)
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        arg = make_num()
        inputs = make_list(of=lambda: make_list(min=1, max=1))
        stream = Stream(inputs)
        stream.starmap(kwargs_recorder(args), kwargs={'guh': arg}).drain()
        self.assertEqual(args, [{'guh': arg}] * len(inputs))


class TestCatch(TestCase):
    """Tests for `catch`."""
    def test_yields_None_for_success(self):
        self.assertEqual(
            Stream([1, 2]).catch(identity).list(),
            [None, None])

    def test_yields_exception(self):
        result = Stream([1]).catch(fail_item).list()
        self.assertEqual(len(result), 1)
        [exception] = result
        self.assertEqual(type(exception), SimulatedFailure)

    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations)).catch(identity)
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        arg = make_num()
        inputs = range(make_num())
        Stream(inputs).catch(kwargs_recorder(args), kwargs={'Q': arg}).drain()
        self.assertEqual(args, [{'Q': arg}] * len(inputs))

    def test_propagates_unexpected_exception(self):
        class FatalFail(BaseException):
            """Simulated error, not derived from Exception."""

        def kaboom(item):
            raise FatalFail("Awful simulated error.")

        self.assertRaises(FatalFail, Stream([1]).catch(kaboom).drain)

    def test_continues_after_exception(self):
        result = Stream([1, 2]).catch(fail_item).list()
        self.assertEqual(len(result), 2)
        [fail1, fail2] = result
        self.assertEqual(type(fail1), SimulatedFailure)
        self.assertEqual(type(fail2), SimulatedFailure)

    def test_handles_mixed_exceptions_and_successes(self):
        def iffy(item):
            if item % 2 == 0:
                return "Yay!"
            elif item == 1:
                raise SimulatedFailure("It's a one.")
            else:
                raise ValueError("Some other error.")

        result = Stream([0, 1, 2, 3]).catch(iffy).list()
        self.assertEqual(
            [type(value) for value in result],
            [type(None), SimulatedFailure, type(None), ValueError])


class TestTakeWhile(TestCase):
    """Tests for `while_true`."""
    def test_stops_when_condition_no_longer_met(self):
        few = lambda item: item < 2
        self.assertEqual(
            Stream([0, 1, 2, 3]).take_while(few).list(),
            [0, 1])

    def test_stops_normally_if_condition_never_met(self):
        reasonable = lambda item: item < 10
        self.assertEqual(
            Stream(range(5)).take_while(reasonable).list(),
            [0, 1, 2, 3, 4])

    def test_does_not_consume_beyond_sentinel(self):
        iterations = []
        few = lambda item: item < 2
        stream = Stream(generate(range(5), iterations))
        stream.take_while(few).drain()
        self.assertEqual(iterations, [0, 1, 2])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(5), iterations)).take_while(
            lambda item: not item)
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        check = kwargs_recorder(args, lambda item: True)
        arg = make_num()
        length = make_num()
        Stream(range(length)).take_while(check, kwargs={'kwarg': arg}).drain()
        self.assertEqual(args, [{'kwarg': arg}] * length)


class TestConcat(TestCase):
    """Tests for `concat`."""
    def test_returns_empty_for_empty_stream(self):
        self.assertEqual(Stream().concat().list(), [])

    def test_returns_empty_if_constituent_streams_are_empty(self):
        self.assertEqual(Stream([[], []]).concat().list(), [])

    def test_concatenates_lists(self):
        self.assertEqual(
            Stream([[1, 2], [3, 4]]).concat().list(),
            [1, 2, 3, 4])

    def test_concatenates_iterators(self):
        def iterate(start, end):
            for item in range(start, end):
                yield item

        self.assertEqual(
            Stream([iterate(0, 2), iterate(2, 4)]).concat().list(),
            [0, 1, 2, 3])

    def test_concatenates_strings(self):
        self.assertEqual(Stream(['xy', 'z']).concat().list(), ['x', 'y', 'z'])

    def test_accepts_long_streams(self):
        # Until Python 3.7, function invocations were limited to 255 arguments.
        # Make sure this is not an issue when concatenating large numbers of
        # iterables.
        self.assertEqual(
            Stream([x] for x in range(500)).concat().list(),
            [x for x in range(500)])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate([range(3)], iterations)).concat()
        self.assertEqual(iterations, [])


class TestGroup(TestCase):
    """Tests for `group`."""
    def test_returns_empty_dict_for_empty_stream(self):
        self.assertEqual(Stream().group(), {})

    def test_groups_by_item_by_default(self):
        self.assertEqual(
            Stream(range(3)).group(),
            {0: [0], 1: [1], 2: [2]})

    def test_groups_by_key_result(self):
        half = lambda number: int(number / 2)
        self.assertEqual(
            Stream(range(3)).group(key=half),
            {
                0: [0, 1],
                1: [2],
            })

    def test_preserves_order(self):
        mod_3 = lambda number: number % 3
        inputs = [4, 8, 7, 0, 2, 1, 5, 3, 6, 9]
        self.assertEqual(
            Stream(inputs).group(key=mod_3),
            {
                0: [0, 3, 6, 9],
                1: [4, 7, 1],
                2: [8, 2, 5],
            })

    def test_preserves_duplicates(self):
        even = lambda number: number % 2 == 0
        inputs = [0, 1, 0, 3, 0, 5]
        self.assertEqual(
            Stream(inputs).group(key=even),
            {
                True: [0, 0, 0],
                False: [1, 3, 5],
            })

    def test_adds_key_kwargs(self):
        factor = make_num(1)
        multiply = lambda item, factor: item * factor
        stream = Stream(range(3))
        self.assertEqual(
            stream.group(key=multiply, key_kwargs={'factor': factor}),
            {
                0: [0],
                1 * factor: [1],
                2 * factor: [2],
            })

    def test_computes_value(self):
        factor = make_num(1)
        multiply = lambda item, factor: item * factor
        stream = Stream(range(3))
        self.assertEqual(
            stream.group(value=multiply, val_kwargs={'factor': factor}),
            {
                0: [0],
                1: [factor],
                2: [2 * factor],
            })


class TestSum(TestCase):
    """Tests for `sum`."""
    def test_returns_initial_if_empty(self):
        initial = make_num()
        self.assertEqual(Stream().sum(initial), initial)

    def test_sums_numbers(self):
        self.assertEqual(Stream(range(4)).sum(0), 6)

    def test_sums_strings(self):
        self.assertEqual(Stream(['foo', 'bar']).sum('go'), 'gofoobar')

    def test_initial_defaults_to_zero(self):
        self.assertEqual(Stream(range(4)).sum(), 6)


class TestReduce(TestCase):
    """Tests for `reduce`."""
    def test_defaults_to_initial_value(self):
        initial = make_num()
        multiply = lambda l, r: l * r
        self.assertEqual(
            Stream().reduce(multiply, initial),
            initial)

    def test_combines_initial_value_with_single_item(self):
        initial = make_num()
        value = make_num()
        multiply = lambda l, r: l * r
        self.assertEqual(
            Stream([value]).reduce(multiply, initial),
            initial * value)

    def test_reduces_series(self):
        add = lambda l, r: l + r
        self.assertEqual(
            Stream([1, 2, 3, 4]).reduce(add, 0),
            10)

    def test_combines_initial_value_with_series(self):
        initial = make_num(2)
        multiply = lambda l, r: l * r
        self.assertEqual(
            Stream([1, 2, 3]).reduce(multiply, initial),
            initial * 1 * 2 * 3)

    def test_processes_left_to_right(self):
        concatenate = lambda l, r: '.'.join([l, r])
        self.assertEqual(
            Stream('abc').reduce(concatenate, '0'),
            '0.a.b.c')


class TestUniq(TestCase):
    """Tests for `uniq`."""
    def test_removes_consecutive_identical_values(self):
        self.assertEqual(Stream([1, 1, 1]).uniq().list(), [1])

    def test_keeps_nonidentical_items(self):
        self.assertEqual(Stream([1, 2, 3]).uniq().list(), [1, 2, 3])

    def test_keeps_nonconsecutive_identical_items(self):
        self.assertEqual(Stream([1, 0, 1]).uniq().list(), [1, 0, 1])

    def test_applies_function(self):
        half = lambda item: int(item / 2)
        self.assertEqual(Stream([0, 1, 2, 3, 4]).uniq(half).list(), [0, 2, 4])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(4), iterations)).uniq()
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        inputs = make_list()
        arg = make_num()
        Stream(inputs).uniq(kwargs_recorder(args), kwargs={'n': arg}).drain()
        self.assertEqual(args, [{'n': arg}] * len(inputs))


class TestPeek(TestCase):
    """Tests for `peek`."""
    def test_calls_function_on_items(self):
        processed = []
        Stream(range(3)).peek(recorder(processed)).drain()
        self.assertEqual(processed, [0, 1, 2])

    def test_passes_items_on_unchanged(self):
        inputs = make_list()
        peeked = []
        self.assertEqual(
            Stream(inputs).peek(recorder(peeked)).list(),
            inputs)

    def test_iterates_lazily(self):
        iterations = []
        processed = []
        Stream(generate(range(5), iterations)).peek(recorder(processed))
        self.assertEqual(iterations, [])
        self.assertEqual(processed, [])

    def test_adds_kwargs(self):
        args = []
        length = make_num()
        foo = make_num()
        Stream(range(length)).peek(kwargs_recorder(args), {'foo': foo}).drain()
        self.assertEqual(args, [{'foo': foo}] * length)

    def test_propagates_exception(self):
        self.assertRaises(
            SimulatedFailure,
            Stream(range(1)).peek(fail_item).drain)


class TestStringJoin(TestCase):
    """Tests for `string_join`."""
    def test_joins_unicode_strings(self):
        self.assertEqual(
            Stream(['x', 'y', 'z']).string_join('.'),
            'x.y.z')

    def test_joins_byte_strings(self):
        self.assertEqual(
            Stream([b'x', b'y', b'z']).string_join(b'.'),
            b'x.y.z')

    def test_joins_characters(self):
        self.assertEqual(Stream('foo').string_join('-'), 'f-o-o')


class TestPathJoin(TestCase):
    """Tests for `path_join`."""
    def test_joins_path_elements(self):
        components = ['a', 'b', 'c']
        self.assertEqual(
            Stream(components).path_join(),
            os.path.join(*components))

    def test_fails_if_empty(self):
        self.assertRaises(TypeError, Stream().path_join)

    def test_fails_on_nonstring(self):
        self.assertRaises(
            (AttributeError, TypeError),
            Stream(['x', 1]).path_join)
        self.assertRaises(
            (AttributeError, TypeError),
            Stream(['x', None]).path_join)


class TestSort(TestCase):
    """Tests for `sort`."""
    def test_empty_for_empty_stream(self):
        self.assertEqual(Stream().sort().list(), [])

    def test_sorts(self):
        self.assertEqual(
            Stream([3, 5, 1, 4, 2]).sort().list(),
            [1, 2, 3, 4, 5])

    def test_sorts_by_key(self):
        key = lambda item: item % 5
        self.assertEqual(
            Stream([10, 4, 12, 26, 8]).sort(key).list(),
            [10, 26, 12, 8, 4])

    def test_adds_kwargs(self):
        args = []
        arg = make_num()
        inputs = make_list()
        Stream(inputs).sort(kwargs_recorder(args), kwargs={'x': arg}).drain()
        self.assertEqual(args, [{'x': arg}] * len(inputs))

    def test_sorts_in_reverse_order_if_requested(self):
        self.assertEqual(
            Stream([2, 1, 3]).sort(reverse=True).list(),
            [3, 2, 1])


def count_items(sequence):
    """Return number of items in sequence."""
    total = 0
    for _ in sequence:
        total += 1
    return total


class TestIntegrate(TestCase):
    """Some scenario tests for `Stream`."""
    def test_filter_lengths_greater_than_3(self):
        # Example from README.md.
        greater_than = lambda value, threshold: value > threshold
        long_items = (Stream([
                [0, 1, 2, 3],
                [3, 2, 1],
                [4, 3, 2, 1, 0],
                [0, 1],
                ])
            .map(count_items)
            .keep_if(greater_than, {'threshold': 3})
            .list()
            )
        self.assertEqual(long_items, [4, 5])

    def test_sum_even_numbers(self):
        is_even = lambda number: number % 2 == 0
        self.assertEqual(
            Stream(range(13)).keep_if(is_even).sum(),
            42)

    def test_sort_uniq_count(self):
        # Example from README.md.
        lines = StringIO(dedent("""\
            Gallia est omnis divisa in partes tres..."
            Cui bono?
            Quidquid id est timeo danaos et dona ferentes.
            Cui bono?
            Sic transit gloria mundi.
            Cui bono?
            """))
        lines.seek(0)
        self.assertEqual(Stream(lines).sort().count(), 6)
        lines.seek(0)
        self.assertEqual(Stream(lines).sort().uniq().count(), 4)

    def test_different_ways_of_summing_nested_lists(self):
        inputs = [
            [0, 1, 2, 3],
            [3, 1],
            [2, 0],
            ]
        total = 12

        self.assertEqual(Stream(inputs).map(sum).sum(), total)
        self.assertEqual(Stream(inputs).concat().sum(), total)
        self.assertEqual(Stream(Stream(inputs).sum([])).sum(), total)
        self.assertEqual(
            Stream(inputs).reduce(
                lambda acc, item: acc + Stream(item).sum(), 0),
            total)
        self.assertEqual(
            Stream(inputs).map(Stream).map(methodcaller('sum')).sum(),
            total)
        self.assertEqual(
            Stream(inputs).starmap(lambda *args: sum(args)).sum(),
            total)

        class Accumulator:
            total = 0

            def inc(self, item):
                self.total += item

        accumulator = Accumulator()
        Stream(inputs).concat().for_each(accumulator.inc)
        self.assertEqual(accumulator.total, total)

        accumulator = Accumulator()
        Stream(inputs).map(Stream).for_each(
            methodcaller('for_each', function=accumulator.inc))
        self.assertEqual(accumulator.total, total)

        self.assertEqual(
            Stream(inputs).map(Stream).map(
                methodcaller('map', lambda item: [0] * item)
                ).concat().concat().count(),
            total)

        self.assertEqual(
            Stream(
                Stream(inputs).group(
                    key=lambda item: sum(item)
                    ).values()
                ).concat().concat().sum(),
            total)


class JavaRecipesTest(TestCase):
    """Tests for recipes to reproduce Java stream methods."""
    def test_flatMap_splits_lines_into_words(self):
        # Example from README.md.  Based on Java Stream.flatMap documentation.
        lines = [
            "I have seen things you people wouldn't believe.",
            "Attack ships on fire off the shoulder of Orion.",
            "I watched C-beams glitter in the dark off the Tannhauser Gate.",
            "All those moments will be lost...",
            "Like tears in rain.",
            "Time to die.",
            ]
        words = Stream(lines).map(methodcaller('split')).concat().list()
        self.assertEqual(words, ' '.join(lines).split())

    def test_max_returns_max_number(self):
        self.assertEqual(Stream([3, 2, 1, 99, 0, 5, 4, 3]).into(max), 99)

    def test_generate_is_count(self):
        # Examples from README.md.
        self.assertEqual(Stream(count())[0:5].list(), [0, 1, 2, 3, 4])
        self.assertEqual(Stream(count(1, 2))[0:5].list(), [1, 3, 5, 7, 9])
