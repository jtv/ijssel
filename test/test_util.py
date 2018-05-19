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
#    negate,
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
