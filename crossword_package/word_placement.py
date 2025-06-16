"""
Word Placement Module

Contains the data structures for representing word placements in a crossword puzzle.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

class Direction(Enum):
    """Enumeration for word direction in crossword puzzle"""
    ACROSS = "across"
    DOWN = "down"

@dataclass(frozen=True)
class WordPlacement:
    """
    Represents a word placed in the crossword puzzle.
    
    Attributes:
        word: The word to be placed
        row: Starting row position (0-indexed)
        col: Starting column position (0-indexed)
        direction: Direction of the word (ACROSS or DOWN)
        clue: Associated clue for the word (optional)
        number: Clue number for crossword numbering (optional)
    """
    word: str
    row: int
    col: int
    direction: Direction
    clue: str = ""
    number: int = 0
    length: int = 0

    def get_end_position(self) -> Tuple[int, int]:
        """
        Get the ending position of this word.
        
        Returns:
            Tuple of (end_row, end_col) positions
        """
        if self.direction == Direction.ACROSS:
            return (self.row, self.col + self.length - 1)
        else:
            return (self.row + self.length - 1, self.col)

    def get_positions(self) -> List[Tuple[int, int]]:
        """
        Get all grid positions this word occupies.
        
        Returns:
            List of (row, col) tuples representing all positions
        """
        positions = []
        for i in range(self.length):
            if self.direction == Direction.ACROSS:
                positions.append((self.row, self.col + i))
            else:
                positions.append((self.row + i, self.col))
        return positions

    def get_letter_at_position(self, row: int, col: int) -> str:
        """
        Get the letter at a specific position if this word occupies it.
        
        Args:
            row: Row position to check
            col: Column position to check
            
        Returns:
            The letter at that position, or empty string if position not occupied
        """
        positions = self.get_positions()
        if (row, col) not in positions:
            return ""

        if self.direction == Direction.ACROSS:
            letter_index = col - self.col
        else:
            letter_index = row - self.row

        return self.word[letter_index] if 0 <= letter_index < len(self.word) else ""

    def __str__(self) -> str:
        return f"{self.word} at ({self.row},{self.col}) {self.direction.value}"

    def __repr__(self) -> str:
        return f"WordPlacement('{self.word}', {self.row}, {self.col}, {self.direction})"
