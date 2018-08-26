"""Randomised factories for use in tests."""

from __future__ import (
    absolute_import,
    print_function,
    unicode_literals,
    )

__metaclass__ = type

from random import randint


def make_num(min=0, max=9):
    """Return a random number."""
    return randint(min, max)


def make_list(min=1, max=9, of=make_num):
    length = make_num(min=min, max=max)
    return [of() for _ in range(length)]
