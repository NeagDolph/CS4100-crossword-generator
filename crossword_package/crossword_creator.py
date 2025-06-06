"""
Crossword Creator Module

Handles crossword puzzle creation and generation.
Uses the grid for placement operations and validation for checking.
"""

import random
from typing import List, Optional, Tuple
from .crossword_grid import CrosswordGrid
from .crossword_validator import CrosswordValidator
from .word_placement import WordPlacement, Direction

class CrosswordCreator:
    """
    Handles crossword puzzle creation and generation.
    Uses the grid for placement operations and validation for checking.
    """
    
    def __init__(self, grid: CrosswordGrid, available_words: List[str] = None):
        """
        Initialize the crossword creator.
        
        Args:
            grid: The crossword grid to work with
            available_words: List of available words (optional)
        """
        self.grid = grid
        self.word_placements: List[WordPlacement] = []
        self.available_words = available_words or []
    
    def place_word(self, word: str, row: int, col: int, direction: Direction, 
                   clue: str = "") -> bool:
        """
        Attempt to place a word on the grid.
        
        Args:
            word: Word to place
            row: Starting row position
            col: Starting column position
            direction: Direction to place word
            clue: Optional clue for the word
            
        Returns:
            True if word was successfully placed, False otherwise
        """
        if not CrosswordValidator.can_place_word(self.grid, word, row, col, direction):
            return False
        
        # Create placement
        placement = WordPlacement(word.upper(), row, col, direction, clue)
        
        # Check intersections with existing words
        test_placements = self.word_placements + [placement]
        if not CrosswordValidator.validate_intersections(self.grid, test_placements):
            return False
        
        # Place the word
        for i, letter in enumerate(word.upper()):
            if direction == Direction.ACROSS:
                self.grid.set_letter(row, col + i, letter, is_given=True)
            else:
                self.grid.set_letter(row + i, col, letter, is_given=True)
        
        self.word_placements.append(placement)
        return True
    
    def remove_word(self, placement: WordPlacement) -> bool:
        """
        Remove a word from the grid.
        
        Args:
            placement: Word placement to remove
            
        Returns:
            True if word was successfully removed, False otherwise
        """
        if placement not in self.word_placements:
            return False
        
        # Remove letters (only if they're not part of other words)
        positions_to_clear = set(placement.get_positions())
        
        # Check which positions are used by other words
        for other_placement in self.word_placements:
            if other_placement != placement:
                positions_to_clear -= set(other_placement.get_positions())
        
        # Clear positions not used by other words
        for row, col in positions_to_clear:
            self.grid.set_letter(row, col, '')
            self.grid.given_cells.discard((row, col))
        
        self.word_placements.remove(placement)
        return True
    
    def remove_word_by_position(self, row: int, col: int) -> bool:
        """
        Remove a word by specifying a position it occupies.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            True if a word was found and removed, False otherwise
        """
        for placement in self.word_placements:
            if (row, col) in placement.get_positions():
                return self.remove_word(placement)
        return False
    
    def get_word_at_position(self, row: int, col: int) -> Optional[WordPlacement]:
        """
        Get the word placement that occupies a specific position.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            WordPlacement if found, None otherwise
        """
        for placement in self.word_placements:
            if (row, col) in placement.get_positions():
                return placement
        return None
    
    def can_place_word_at_position(self, word: str, row: int, col: int, direction: Direction) -> bool:
        """
        Check if a word can be placed at a specific position.
        
        Args:
            word: Word to check
            row: Starting row position
            col: Starting column position
            direction: Direction to place word
            
        Returns:
            True if word can be placed, False otherwise
        """
        return CrosswordValidator.can_place_word(self.grid, word, row, col, direction)
    
    def clear_puzzle(self):
        """Clear the entire puzzle, resetting to empty state."""
        self.grid = CrosswordGrid(self.grid.size)
        self.word_placements.clear()
    
    def get_puzzle_statistics(self) -> dict:
        """
        Get statistics about the current puzzle.
        
        Returns:
            Dictionary with puzzle statistics
        """
        filled_cells = len(self.grid.get_filled_cells())
        total_cells = self.grid.size * self.grid.size
        blocked_cells = len(self.grid.blocked_cells)
        
        return {
            'word_count': len(self.word_placements),
            'filled_cells': filled_cells,
            'total_cells': total_cells,
            'blocked_cells': blocked_cells,
            'fill_percentage': (filled_cells / (total_cells - blocked_cells) * 100) if total_cells > blocked_cells else 0,
            'intersection_count': CrosswordValidator.count_word_intersections(self.word_placements),
            'is_connected': CrosswordValidator.validate_grid_connectivity(self.word_placements)
        }
    
    def validate_puzzle(self) -> bool:
        """
        Validate the current puzzle for correctness.
        
        Returns:
            True if puzzle is valid, False otherwise
        """
        return (
            CrosswordValidator.validate_intersections(self.grid, self.word_placements) and
            CrosswordValidator.validate_grid_connectivity(self.word_placements)
        ) 