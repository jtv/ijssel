"""Tests for util."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type

from random import randint
from unittest import TestCase

from ..ijssel.util import (
    bind_kwargs,
    identity,
#    ifilter,
#    imap,
    int_types,
    )


class TestBindKwargs(TestCase):
    """Tests for bind_kwargs."""
    def test_binds_nothing_by_default(self):
        double = bind_kwargs(lambda value: 2 * value)
        arg = randint(0, 1000)
        self.assertEqual(double(arg), 2 * arg)

    def test_binds_kwargs(self):
        factor = randint(0, 100)
        times = bind_kwargs(
            lambda value, factor: value * factor,
            kwargs={'factor': factor})
        arg = randint(0, 100)
        self.assertEqual(
            times(arg),
            arg * factor)


class TestIdentity(TestCase):
    """Tests for `identity`."""
    def test_returns_argument(self):
        arg = randint(0, 100)
        self.assertEqual(identity(arg), arg)


class TestIntTypes(TestCase):
    """Tests for `int_types`."""
    def test_small_number_fits(self):
        self.assertTrue(isinstance(0, int_types))

    def test_large_number_fits(self):
        # This is an int in Python 3, but a long in Python 2.
        self.assertTrue(isinstance(9999999999999999999999, int_types))

    def test_other_types_do_not_fit(self):
        self.assertFalse(isinstance(None, int_types))
        self.assertFalse(isinstance(0.0, int_types))
        self.assertFalse(isinstance(u'', int_types))
        self.assertFalse(isinstance(b'', int_types))
        self.assertFalse(isinstance([], int_types))
        self.assertFalse(isinstance(object(), int_types))
