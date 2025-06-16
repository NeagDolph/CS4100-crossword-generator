"""
Word Placement Module

Contains the data structures for representing word placements in a crossword puzzle.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple

class Direction(Enum):
    ACROSS = "across"
    DOWN = "down"

@dataclass
class WordPlacement:
    word: str
    row: int
    col: int
    direction: Direction
    clue: str = ""
    number: int = 0
    
    def get_end_position(self) -> Tuple[int, int]:
        if self.direction == Direction.ACROSS:
            return (self.row, self.col + len(self.word) - 1)
        else:
            return (self.row + len(self.word) - 1, self.col)
    
    def get_positions(self) -> List[Tuple[int, int]]:
        positions = []
        for i in range(len(self.word)):
            if self.direction == Direction.ACROSS:
                positions.append((self.row, self.col + i))
            else:
                positions.append((self.row + i, self.col))
        return positions
    
    def get_letter_at_position(self, row: int, col: int) -> str:
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