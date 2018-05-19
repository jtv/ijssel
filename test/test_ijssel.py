"""Tests for IJssel Stream class."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type

import os.path
from random import randint
from unittest import TestCase

from ..ijssel import Stream
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
        n = randint(1, 10)
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

    def test_iterates_sequence(self):
        n = randint(1, 10)

        def generate():
            for item in range(n):
                yield item

        self.assertEqual(list(Stream(generate())), list(range(n)))

    def test_iterates_range(self):
        self.assertEqual(list(Stream(range(3))), [0, 1, 2])

    def test_does_not_iterate_non_sequence(self):
        self.assertRaises(TypeError, list, Stream(None))
        self.assertRaises(TypeError, list, Stream(10))

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


class TestList(TestCase):
    """Tests for `list`."""
    def test_returns_empty_for_empty_stream(self):
        self.assertEqual(Stream().list(), [])

    def test_returns_items_as_list(self):
        self.assertEqual(Stream(range(3)).list(), [0, 1, 2])

    def test_turns_iterator_into_list(self):
        item = randint(0, 10)

        def iterate():
            yield item

        self.assertEqual(Stream(iterate()).list(), [item])


class TestTuple(TestCase):
    """Tests for `tuple`."""
    def test_returns_empty_for_empty_stream(self):
        self.assertEqual(Stream().tuple(), tuple())

    def test_returns_items_as_tuple(self):
        self.assertEqual(Stream(range(3)).tuple(), (0, 1, 2))

    def test_turns_iterator_into_tuple(self):
        item = randint(0, 10)

        def iterate():
            yield item

        self.assertEqual(Stream(iterate()).tuple(), tuple([item]))


class TestAll(TestCase):
    """Tests for `all`."""
    def test_returns_True_for_empty_stream(self):
        self.assertTrue(Stream().all())

    def test_returns_True_if_all_items_true(self):
        trues = [True, 1, 'y', [0], {0: 0}]
        self.assertTrue(Stream(trues).all())

    def test_returns_False_if_any_item_false(self):
        self.assertFalse(Stream([True, True, False]).all())

    def test_consumes_all_if_True(self):
        trues = [True] * randint(0, 5)
        iterations = []
        Stream(generate(trues, iterations)).all()
        self.assertEqual(iterations, trues)

    def test_stops_early_if_False(self):
        bools = [True] * randint(0, 3) + [False] + [True] * randint(0, 3)
        iterations = []
        Stream(generate(bools, iterations)).all()
        self.assertNotEqual(iterations, [])
        self.assertFalse(iterations[-1])
        self.assertEqual(iterations, bools[:len(iterations)])
        self.assertEqual(len(iterations), iterations.index(False) + 1)


class TestAny(TestCase):
    """Tests for `any`."""
    def test_returns_False_for_empty_stream(self):
        self.assertFalse(Stream().any())

    def test_returns_False_if_all_items_true(self):
        falses = [False, 0, '', [], {}]
        self.assertFalse(Stream(falses).any())

    def test_returns_True_if_any_item_true(self):
        self.assertTrue(Stream([False, False, True]).any())

    def test_consumes_all_if_False(self):
        falses = [False] * randint(0, 5)
        iterations = []
        Stream(generate(falses, iterations)).any()
        self.assertEqual(iterations, falses)

    def test_stops_early_if_True(self):
        bools = [False] * randint(0, 3) + [True] + [False] * randint(0, 3)
        iterations = []
        Stream(generate(bools, iterations)).any()
        self.assertNotEqual(iterations, [])
        self.assertTrue(iterations[-1])
        self.assertEqual(iterations, bools[:len(iterations)])
        self.assertEqual(len(iterations), iterations.index(True) + 1)


class TestNegate(TestCase):
    """Tests for `negate`."""
    def test_returns_Stream(self):
        self.assertIsInstance(Stream([1]).negate(), Stream)

    def test_does_nothing_if_empty(self):
        self.assertEqual(Stream().negate().list(), [])

    def test_negates_true_items_to_False(self):
        trues = [True, 1, 'y', [0], {0: 0}]
        self.assertEqual(Stream(trues).negate().list(), [False] * len(trues))

    def test_negates_false_items_to_True(self):
        falses = [False, 0, '', [], {}]
        self.assertEqual(Stream(falses).negate().list(), [True] * len(falses))

    def test_lazy(self):
        iterations = []
        Stream(generate([1, 2], iterations)).negate()
        self.assertEqual(iterations, [])


class TestCount(TestCase):
    """Tests for `count`."""
    def test_returns_zero_if_empty(self):
        self.assertEqual(Stream().count(), 0)

    def test_counts_items(self):
        values = [randint(0, 10) for _ in range(randint(0, 5))]
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
        inputs = [randint(0, 10) for _ in range(randint(1, 10))]
        Stream(inputs).for_each(recorder(processed))
        self.assertEqual(processed, inputs)

    def test_adds_kwargs(self):
        args = []
        n = randint(1, 3)
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
        inputs = [randint(0, 10) for _ in range(randint(1, 5))]
        Stream(generate(inputs, iterations)).drain()
        self.assertEqual(iterations, inputs)


class TestFilter(TestCase):
    """Tests for `filter`."""
    def test_filters_on_identity_by_default(self):
        # Some values that alternately evaluate as "true" and "false".
        values = [True, False, 0, 1, [], [0], {}, {0: 0}, '', '0']
        # You only get the "true" ones.
        self.assertEqual(
            Stream(values).filter().list(),
            [item for item in values if item])

    def test_filters_on_criterion(self):
        even = lambda item: item % 2 == 0
        self.assertEqual(
            Stream([0, 1, 2, 3, 4]).filter(even).list(),
            [0, 2, 4])

    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations)).filter()
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        arg = randint(0, 10)
        criterion = kwargs_recorder(args, lambda item: item > 3)
        length = 5
        stream = Stream(range(length))
        result = stream.filter(criterion, kwargs={'kwarg': arg}).list()

        self.assertEqual(result, [4])
        self.assertEqual(args, [{'kwarg': arg}] * length)

    def test_stops_at_exception(self):
        iterations = []
        # This will deliberately fail when item == 3.
        criterion = lambda item: 100 / (item - 3)
        stream = Stream(generate([1, 2, 3, 4, 5], iterations))
        self.assertRaises(
            ZeroDivisionError,
            stream.filter(criterion).drain)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


class TestFilterOut(TestCase):
    """Tests for `filter_out`."""
    def test_filters_on_identity_by_default(self):
        # Some values that alternately evaluate as "true" and "false".
        values = [True, False, 0, 1, [], [0], {}, {0: 0}, '', '0']
        # You only get the "false" ones.
        self.assertEqual(
            Stream(values).filter_out().list(),
            [item for item in values if not item])

    def test_filters_on_criterion(self):
        even = lambda item: item % 2 == 0
        self.assertEqual(
            Stream([0, 1, 2, 3, 4]).filter_out(even).list(),
            [1, 3])

    def test_lazy(self):
        iterations = []
        Stream(generate(range(5), iterations)).filter_out()
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        arg = randint(0, 10)
        criterion = kwargs_recorder(args, lambda item: item > 3)
        length = 5
        stream = Stream(range(length))
        result = stream.filter_out(criterion, kwargs={'kwarg': arg}).list()

        self.assertEqual(result, [0, 1, 2, 3])
        self.assertEqual(args, [{'kwarg': arg}] * length)

    def test_stops_at_exception(self):
        iterations = []
        # This will deliberately fail when item == 3.
        criterion = lambda item: 100 / (item - 3)
        stream = Stream(generate([1, 2, 3, 4, 5], iterations))
        self.assertRaises(
            ZeroDivisionError,
            stream.filter_out(criterion).drain)
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
        arg = randint(0, 10)
        inputs = [[randint(0, 10)] for _ in range(randint(1, 10))]
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
        arg = randint(0, 10)
        inputs = range(randint(1, 10))
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


class TestLimit(TestCase):
    """Tests for `limit`."""
    def test_stops_iteration_at_limit(self):
        self.assertEqual(
            Stream(range(5)).limit(2).list(),
            [0, 1])

    def test_does_nothing_if_limit_is_beyond_stream(self):
        self.assertEqual(
            Stream(range(3)).limit(10).list(),
            [0, 1, 2])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(5), iterations)).limit(3)
        self.assertEqual(iterations, [])

    def test_does_not_consume_beyond_limit(self):
        iterations = []
        Stream(generate(range(5), iterations)).limit(2).drain()
        self.assertEqual(iterations, [0, 1])

    def test_limiting_to_zero_produces_empty_stream(self):
        self.assertEqual(Stream(range(5)).limit(0).list(), [])

    def test_raises_TypeError_if_limit_is_not_integer(self):
        self.assertRaises(TypeError, Stream(range(5)).limit, None)
        self.assertRaises(TypeError, Stream(range(5)).limit, '5')
        self.assertRaises(TypeError, Stream(range(5)).limit, [1])

    def test_raises_ValueError_if_limit_is_negative(self):
        self.assertRaises(ValueError, Stream(range(5)).limit, -1)


class TestUntilValue(TestCase):
    """Tests for `until_value`."""
    def test_stops_at_sentinel(self):
        self.assertEqual(Stream(range(5)).until_value(3).list(), [0, 1, 2])

    def test_stops_normally_if_sentinel_not_found(self):
        self.assertEqual(
            Stream(range(5)).until_value(9).list(),
            [0, 1, 2, 3, 4])

    def test_checks_value_not_identity(self):
        self.assertEqual(
            Stream([[0], [1], [2], [3]]).until_value([2]).list(),
            [[0], [1]])

    def test_does_not_consume_beyond_sentinel(self):
        iterations = []
        Stream(generate(range(5), iterations)).until_value(2).drain()
        self.assertEqual(iterations, [0, 1, 2])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(5), iterations)).until_value(2)
        self.assertEqual(iterations, [])


class TestUntilIdentity(TestCase):
    """Tests for `until_identity`."""
    def test_stops_at_sentinel(self):
        stop = object()
        self.assertEqual(
            Stream([0, 1, stop, 3]).until_identity(stop).list(),
            [0, 1])

    def test_stops_normally_if_sentinel_not_found(self):
        self.assertEqual(
            Stream(range(5)).until_identity(9).list(),
            [0, 1, 2, 3, 4])

    def test_checks_identity_not_value(self):
        self.assertEqual(
            Stream([[0], [1], [2], [3]]).until_identity([2]).list(),
            [[0], [1], [2], [3]])

    def test_does_not_consume_beyond_sentinel(self):
        stop = object()
        iterations = []
        stream = Stream(generate([0, 1, stop, 2], iterations))
        stream.until_identity(stop).drain()
        self.assertEqual(iterations, [0, 1, stop])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(5), iterations)).until_identity(2)
        self.assertEqual(iterations, [])


class TestUntilTrue(TestCase):
    """Tests for `until_true`."""
    def test_stops_when_condition_met(self):
        many = lambda item: item > 1
        self.assertEqual(
            Stream([0, 1, 2, 3]).until_true(many).list(),
            [0, 1])

    def test_stops_normally_if_condition_never_met(self):
        huge = lambda item: item > 10
        self.assertEqual(
            Stream(range(5)).until_true(huge).list(),
            [0, 1, 2, 3, 4])

    def test_does_not_consume_beyond_sentinel(self):
        iterations = []
        many = lambda item: item > 1
        stream = Stream(generate(range(5), iterations))
        stream.until_true(many).drain()
        self.assertEqual(iterations, [0, 1, 2])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(5), iterations)).until_true()
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        check = kwargs_recorder(args, lambda item: False)
        arg = randint(0, 10)
        length = randint(1, 5)
        Stream(range(length)).until_true(check, kwargs={'kwarg': arg}).drain()

        self.assertEqual(args, [{'kwarg': arg}] * length)


class TestWhileTrue(TestCase):
    """Tests for `while_true`."""
    def test_stops_when_condition_no_longer_met(self):
        few = lambda item: item < 2
        self.assertEqual(
            Stream([0, 1, 2, 3]).while_true(few).list(),
            [0, 1])

    def test_stops_normally_if_condition_never_met(self):
        reasonable = lambda item: item < 10
        self.assertEqual(
            Stream(range(5)).while_true(reasonable).list(),
            [0, 1, 2, 3, 4])

    def test_does_not_consume_beyond_sentinel(self):
        iterations = []
        few = lambda item: item < 2
        stream = Stream(generate(range(5), iterations))
        stream.while_true(few).drain()
        self.assertEqual(iterations, [0, 1, 2])

    def test_iterates_lazily(self):
        iterations = []
        Stream(generate(range(5), iterations)).while_true(
            lambda item: not item)
        self.assertEqual(iterations, [])

    def test_adds_kwargs(self):
        args = []
        check = kwargs_recorder(args, lambda item: True)
        arg = randint(0, 10)
        length = randint(1, 5)
        Stream(range(length)).while_true(check, kwargs={'kwarg': arg}).drain()
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
        def count(start, end):
            for item in range(start, end):
                yield item

        self.assertEqual(
            Stream([count(0, 2), count(2, 4)]).concat().list(),
            [0, 1, 2, 3])

    def test_concatenates_strings(self):
        self.assertEqual(Stream(['xy', 'z']).concat().list(), ['x', 'y', 'z'])

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
        factor = randint(1, 10)
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
        factor = randint(1, 10)
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
        initial = randint(0, 10)
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
        initial = randint(0, 10)
        multiply = lambda l, r: l * r
        self.assertEqual(
            Stream().reduce(multiply, initial),
            initial)

    def test_combines_initial_value_with_single_item(self):
        initial = randint(1, 10)
        value = randint(1, 10)
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
        initial = randint(2, 10)
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
        inputs = [randint(0, 10) for _ in range(10)]
        arg = randint(0, 10)
        Stream(inputs).uniq(kwargs_recorder(args), kwargs={'n': arg}).drain()
        self.assertEqual(args, [{'n': arg}] * len(inputs))


class TestPeek(TestCase):
    """Tests for `peek`."""
    def test_calls_function_on_items(self):
        processed = []
        Stream(range(3)).peek(recorder(processed)).drain()
        self.assertEqual(processed, [0, 1, 2])

    def test_passes_items_on_unchanged(self):
        inputs = [randint(0, 10) for _ in range(randint(1, 10))]
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
        length = randint(1, 10)
        foo = randint(0, 10)
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
        arg = randint(0, 10)
        inputs = [randint(0, 10) for _ in range(randint(1, 10))]
        Stream(inputs).sort(kwargs_recorder(args), kwargs={'x': arg}).drain()
        self.assertEqual(args, [{'x': arg}] * len(inputs))
