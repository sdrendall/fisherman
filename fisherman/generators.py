"""
Some general purpose generator functions used in this project
"""

from collections import Iterable
from itertools import islice

def counter(start=0, increment=1):
    """
    Produces generators that yield an infinite sequence of numbers, beginning with start, and incrementing by increment
    By default, these generators yeilds increasing integers, beginning with zero
    """
    current_number = start
    while True:
        yield current_number
        current_number += increment

def iter_wrap(maybe_iterable):
    """
    Wraps an object so that it is always an iterable.
    """
    if isinstance(maybe_iterable, Iterable):
        for value in maybe_iterable:
            yield value
    else:
        yield maybe_iterable

def seq_chunks(seq, size):
    """
    Yields chunks of size "size" from seq
    """
    start = 0
    for chunk in islice(seq, start, start + size):
        yield chunk
        start += size
