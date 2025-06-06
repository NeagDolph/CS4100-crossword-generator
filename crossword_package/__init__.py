"""
Crossword Package - Modular Crossword Puzzle System

This package provides a complete crossword puzzle generation and gameplay system
with separate concerns for grid management, word placement, validation, creation, and gameplay.
Word-clue data is loaded from CSV files for realistic crossword creation.
"""

from .word_placement import Direction, WordPlacement
from .crossword_grid import CrosswordGrid
from .crossword_validator import CrosswordValidator
from .crossword_creator import CrosswordCreator
from .crossword_player import CrosswordPlayer
from .crossword_puzzle import CrosswordPuzzle
from .word_data import WordDataManager, WordClue, word_data_manager

# Constants
GRID_SIZE = 15

# Legacy word list
WORDS = ['APPLE', 'BANANA', 'ORANGE', 'GRAPE', 'PEAR', 'PLUM', 'MELON', 'KIWI']

__all__ = [
    'Direction', 'WordPlacement', 'CrosswordGrid', 'CrosswordValidator',
    'CrosswordCreator', 'CrosswordPlayer', 'CrosswordPuzzle',
    'WordDataManager', 'WordClue', 'word_data_manager',
    'GRID_SIZE', 'WORDS'
]

__version__ = '1.0.0'
__author__ = 'CS4100 Final Project' 