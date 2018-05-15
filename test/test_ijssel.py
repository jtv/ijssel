"""Tests for IJssel Stream class."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type

from random import randint
from unittest import TestCase

from ..ijssel import Stream


# Standard things to test:
#
# Method that returns streams -
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


def record(sequence, iterations):
    """Yield items from sequence, also appending them to iterations.

    Use this to test which items a stream actually consumes.  Only those
    items which are actually consumed are appended to iterations.
    """
    for item in sequence:
        iterations.append(item)
        yield item


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

    def test_evaluates_lazily(self):
        iterations = []
        sequence = iter(Stream(record(range(5), iterations)))
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
        Stream(record(trues, iterations)).all()
        self.assertEqual(iterations, trues)

    def test_stops_early_if_False(self):
        bools = [True] * randint(0, 3) + [False] + [True] * randint(0, 3)
        iterations = []
        Stream(record(bools, iterations)).all()
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
        Stream(record(falses, iterations)).any()
        self.assertEqual(iterations, falses)

    def test_stops_early_if_True(self):
        bools = [False] * randint(0, 3) + [True] + [False] * randint(0, 3)
        iterations = []
        Stream(record(bools, iterations)).any()
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

    def test_consumes_lazily(self):
        iterations = []
        Stream(record([1, 2], iterations)).negate()
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
        empty = Stream(record(range(3), iterations)).empty()
        self.assertFalse(empty)
        self.assertEqual(iterations, [0])

    def test_works_for_iterator(self):
        self.assertTrue(Stream(iter([])).empty())


class TestForEach(TestCase):
    """Tests for `for_each`."""
    def test_does_nothing_if_empty(self):
        def fail(item):
            raise Exception("Deliberately failing at item %s." % item)

        Stream().for_each(fail)
        Stream([]).for_each(fail)
        # The real test is that we get here without an exception.

    def test_calls_function_on_each_item(self):
        processed = []

        def process(item):
            processed.append(item)

        inputs = [randint(0, 10) for _ in range(randint(1, 10))]
        Stream(inputs).for_each(process)

        self.assertEqual(processed, inputs)

    def test_adds_kwargs(self):
        arguments = []

        def process(item, kwarg=None):
            arguments.append(kwarg)

        n = randint(1, 3)
        Stream(range(n)).for_each(process, kwargs={'kwarg': 'foo'})

        self.assertEqual(arguments, ['foo'] * n)

    def test_stops_at_exception(self):
        iterations = []

        def process(item):
            return 100 / (item - 3)

        stream = Stream(record([1, 2, 3, 4, 5], iterations))
        self.assertRaises(ZeroDivisionError, stream.for_each, process)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


class TestDrain(TestCase):
    """Tests for `drain`."""
    def test_consumes_all_items(self):
        iterations = []
        inputs = [randint(0, 10) for _ in range(randint(1, 5))]
        Stream(record(inputs, iterations)).drain()
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

    def test_adds_kwargs(self):
        args = []
        arg = randint(0, 10)

        def criterion(item, kwarg):
            args.append(kwarg)
            return item > 3

        length = 5
        stream = Stream(range(length))
        result = stream.filter(criterion, kwargs={'kwarg': arg}).list()

        self.assertEqual(result, [4])
        self.assertEqual(args, [arg] * length)

    def test_stops_at_exception(self):
        iterations = []

        def criterion(item):
            # This will deliberately fail when item == 3.
            return 100 / (item - 3)

        stream = Stream(record([1, 2, 3, 4, 5], iterations))
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

    def test_adds_kwargs(self):
        args = []
        arg = randint(0, 10)

        def criterion(item, kwarg):
            args.append(kwarg)
            return item > 3

        length = 5
        stream = Stream(range(length))
        result = stream.filter_out(criterion, kwargs={'kwarg': arg}).list()

        self.assertEqual(result, [0, 1, 2, 3])
        self.assertEqual(args, [arg] * length)

    def test_stops_at_exception(self):
        iterations = []

        def criterion(item):
            # This will deliberately fail when item == 3.
            return 100 / (item - 3)

        stream = Stream(record([1, 2, 3, 4, 5], iterations))
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

    def test_adds_kwargs(self):
        stream = Stream([[1, 2, 3, 4]])
        self.assertEqual(
            stream.map(sorted, kwargs={'reverse': True}).list(),
            [[4, 3, 2, 1]])

    def test_stops_at_exception(self):
        iterations = []

        def function(item):
            # This will deliberately fail when item == 3.
            return 100 / (item - 3)

        stream = Stream(record([1, 2, 3, 4, 5], iterations))
        self.assertRaises(
            ZeroDivisionError,
            stream.map(function).drain)
        self.assertEqual(iterations, [1, 2, 3])
        self.assertEqual(stream.list(), [4, 5])


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

    def test_does_not_consume_beyond_limit(self):
        iterations = []
        Stream(record(range(5), iterations)).limit(2).drain()
        self.assertEqual(iterations, [0, 1])


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
        Stream(record(range(5), iterations)).until_value(2).drain()
        self.assertEqual(iterations, [0, 1, 2])


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
        stream = Stream(record([0, 1, stop, 2], iterations))
        stream.until_identity(stop).drain()
        self.assertEqual(iterations, [0, 1, stop])


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
        stream = Stream(record(range(5), iterations))
        stream.until_true(many).drain()
        self.assertEqual(iterations, [0, 1, 2])

    def test_adds_kwargs(self):
        args = []

        def check(item, kwarg):
            args.append(kwarg)
            return False

        arg = randint(0, 10)
        length = randint(1, 5)
        Stream(range(length)).until_true(check, kwargs={'kwarg': arg}).drain()

        self.assertEqual(args, [arg] * length)


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
        stream = Stream(record(range(5), iterations))
        stream.while_true(few).drain()
        self.assertEqual(iterations, [0, 1, 2])

    def test_adds_kwargs(self):
        args = []

        def check(item, kwarg):
            args.append(kwarg)
            return True

        arg = randint(0, 10)
        length = randint(1, 5)
        Stream(range(length)).while_true(check, kwargs={'kwarg': arg}).drain()

        self.assertEqual(args, [arg] * length)


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


class TestPartition(TestCase):
    """Tests for `partition`."""
    def test_returns_empty_dict_for_empty_stream(self):
        self.assertEqual(Stream().partition(), {})

    def test_partitions_by_item_by_default(self):
        self.assertEqual(
            Stream(range(3)).partition(),
            {0: [0], 1: [1], 2: [2]})

    def test_partitions_by_key_result(self):
        half = lambda number: number / 2
        self.assertEqual(
            Stream(range(3)).partition(key=half),
            {
                0: [0, 1],
                1: [2],
            })

    def test_preserves_order(self):
        mod_3 = lambda number: number % 3
        inputs = [4, 8, 7, 0, 2, 1, 5, 3, 6, 9]
        self.assertEqual(
            Stream(inputs).partition(key=mod_3),
            {
                0: [0, 3, 6, 9],
                1: [4, 7, 1],
                2: [8, 2, 5],
            })

    def test_preserves_duplicates(self):
        even = lambda number: number % 2 == 0
        inputs = [0, 1, 0, 3, 0, 5]
        self.assertEqual(
            Stream(inputs).partition(key=even),
            {
                True: [0, 0, 0],
                False: [1, 3, 5],
            })

    def test_adds_key_kwargs(self):
        factor = randint(1, 10)
        multiply = lambda item, factor: item * factor
        stream = Stream(range(3))
        self.assertEqual(
            stream.partition(key=multiply, key_kwargs={'factor': 2}),
            {
                0: [0],
                1 * factor: [1],
                2 * factor: [2],
            })
