import math
import random

letter_freq = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 7.31,
        'N': 6.95, 'S': 6.28, 'R': 6.02, 'H': 5.92, 'D': 4.32,
        'L': 3.98, 'U': 2.88, 'C': 2.71, 'M': 2.61, 'F': 2.30,
        'Y': 2.11, 'W': 2.09, 'G': 2.03, 'P': 1.82, 'B': 1.49,
        'V': 1.11, 'K': 0.69, 'X': 0.17, 'Q': 0.11, 'J': 0.10, 'Z': 0.07
    }

def exponential_random_choice(max_value, lambd=0.8):
    """
    Returns an integer from 1 to max_value (inclusive),
    with exponentially decreasing probability.

    Args:
        max_value: The maximum value to choose from.
        lambd: The decay rate of the exponential distribution.
    """
    while True:
        x = int(math.exp(random.uniform(0, math.log(max_value))))
        if 1 <= x <= max_value:
            return x
        

def frequency_score(word: str) -> float:
    """
    Calculate a score based on letter frequency in the English language. More frequent letters lead to a higher score.
    """
    return sum(letter_freq.get(letter.upper(), 0) for letter in word)