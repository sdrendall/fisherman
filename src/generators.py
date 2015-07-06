"""
Some general purpose generator functions used in this project
"""

def counter(start=0, increment=1):
    """
    Produces generators that yield an infinite sequence of numbers, beginning with start, and incrementing by increment
    By default, these generators yeilds increasing integers, beginning with zero
    """
    current_number = start

    yield current_number
    while True:
        current_number += increment
        yield current_number
