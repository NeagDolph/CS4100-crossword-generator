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
from .word_data import WordDataManager
import numpy as np

class CrosswordCreator:
    """
    Handles crossword puzzle creation and generation.
    Uses the grid for placement operations and validation for checking.
    """
    
    def __init__(self, grid: CrosswordGrid, word_data_manager: Optional[WordDataManager] = None):
        """
        Initialize the crossword creator.
        
        Args:
            grid: The crossword grid to work with
            word_data_manager: Optional WordDataManager for accessing word-clue data
        """
        self.grid = grid
        self.word_placements: List[WordPlacement] = []
        self.word_data_manager = word_data_manager
        
    
    def place_word_placement(self, placement: WordPlacement) -> bool:
        """
        Attempt to place a word placement on the grid.
        
        Args:
            placement: WordPlacement object containing word and position data
            
        Returns:
            True if word was successfully placed, False otherwise
        """
        if not self.can_place_word_placement(placement):
            return False
        
        # Check intersections with existing words
        test_placements = self.word_placements + [placement]
        if not CrosswordValidator.validate_intersections(test_placements):
            return False
        
        # Place the word
        for i, letter in enumerate(placement.word.upper()):
            if placement.direction == Direction.ACROSS:
                self.grid.set_letter(placement.row, placement.col + i, letter, is_given=True)
            else:
                self.grid.set_letter(placement.row + i, placement.col, letter, is_given=True)
        
        self.word_placements.append(placement)
        return True
    
    

    
    def place_word_placement_with_auto_clue(self, placement: WordPlacement) -> bool:
        """
        Place a word placement with automatically retrieved clue from data manager.
        
        Args:
            placement: WordPlacement object (clue will be auto-filled if empty)
            
        Returns:
            True if word was successfully placed, False otherwise
        """
        # Auto-fill clue if not provided
        if not placement.clue and self.word_data_manager:
            clue = self.word_data_manager.get_clue_for_word(placement.word) or ""
            placement = WordPlacement(placement.word, placement.row, placement.col, 
                                    placement.direction, clue, placement.number)
        
        return self.place_word_placement(placement)
    
    def place_word(self, word: str, row: int, col: int, direction: Direction, clue: str = "") -> bool:
        """
        Place a word at the specified position.
        
        Args:
            word: Word to place
            row: Starting row position
            col: Starting column position  
            direction: Direction (ACROSS or DOWN)
            clue: Associated clue (optional)
            
        Returns:
            True if word was successfully placed, False otherwise
        """
        # Auto-fill clue if not provided and word_data_manager is available
        if not clue and self.word_data_manager:
            clue = self.word_data_manager.get_clue_for_word(word) or ""
        
        placement = WordPlacement(word.upper(), row, col, direction, clue)
        return self.place_word_placement(placement)
    
    def get_available_words(self, min_length: int = 3, max_length: int = 15, count: int = 100) -> List[str]:
        """
        Get available words for puzzle creation.
        
        Args:
            min_length: Minimum word length
            max_length: Maximum word length
            count: Number of words to return
            
        Returns:
            List of available words
        """
        if self.word_data_manager:
            return self.word_data_manager.get_random_words(count, min_length, max_length)
        else:
            raise ValueError("WordDataManager not provided")
    
    def find_words_that_intersect(self, existing_word: str, position: int, 
                                  direction: Direction, min_length: int = 3, 
                                  max_length: int = 15) -> List[Tuple[str, str]]:
        """
        Find words that can intersect with an existing word at a specific position.
        
        Args:
            existing_word: The word already placed
            position: Position in the existing word (0-indexed)
            direction: Direction of the intersecting word (opposite of existing)
            min_length: Minimum length of intersecting words
            max_length: Maximum length of intersecting words
            
        Returns:
            List of (word, clue) tuples that can intersect
        """
        if not self.word_data_manager or position >= len(existing_word):
            return []
        
        target_letter = existing_word[position].upper()
        candidates = []
        
        # Get available words in the length range
        available_words = self.word_data_manager.get_words_by_length(min_length, max_length)
        
        # Filter words that can intersect based on direction
        for word in available_words:
            # For ACROSS words, we need to find words that have the target letter
            # at a position that would allow a valid DOWN intersection
            if direction == Direction.ACROSS:
                # Find all positions in the word that match the target letter
                for pos in range(len(word)):
                    if word[pos].upper() == target_letter:
                        # Check if this position would create a valid intersection
                        # by ensuring there's enough space before and after
                        if pos >= min_length - 1 and len(word) - pos >= min_length:
                            clue = self.word_data_manager.get_clue_for_word(word)
                            if clue:
                                candidates.append((word, clue))
                                break  # Only need one valid position per word
            
            # For DOWN words, we need to find words that have the target letter
            # at a position that would allow a valid ACROSS intersection
            else:  # Direction.DOWN
                # Find all positions in the word that match the target letter
                for pos in range(len(word)):
                    if word[pos].upper() == target_letter:
                        # Check if this position would create a valid intersection
                        # by ensuring there's enough space before and after
                        if pos >= min_length - 1 and len(word) - pos >= min_length:
                            clue = self.word_data_manager.get_clue_for_word(word)
                            if clue:
                                candidates.append((word, clue))
                                break  # Only need one valid position per word
        
        return candidates[:50]  # Limit to first 50 matches
    
    def suggest_word_placements(self, word: str, max_suggestions: int = 10) -> List[WordPlacement]:
        """
        Suggest possible placements for a word in the current grid.
        
        Args:
            word: Word to find placements for
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of WordPlacement objects for valid placements
        """
        suggestions = []
        
        # Try all possible positions
        for row in range(self.grid.size):
            for col in range(self.grid.size):
                for direction in [Direction.ACROSS, Direction.DOWN]:
                    # Get clue for the word
                    clue = ""
                    if self.word_data_manager:
                        clue = self.word_data_manager.get_clue_for_word(word) or ""
                    
                    temp_placement = WordPlacement(word.upper(), row, col, direction, clue)
                    
                    if self.can_place_word_placement(temp_placement):
                        # Check if this placement would create valid intersections
                        test_placements = self.word_placements + [temp_placement]
                        
                        if CrosswordValidator.validate_intersections(test_placements):
                            suggestions.append(temp_placement)
                            
                            if len(suggestions) >= max_suggestions:
                                return suggestions
        
        return suggestions
    

    
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
    
    def can_place_word_placement(self, placement: WordPlacement) -> bool:
        """
        Check if a word placement can be placed on the grid.
        
        Args:
            placement: WordPlacement object to check
            
        Returns:
            True if word can be placed, False otherwise
        """
        return CrosswordValidator.can_place_word_placement(self.grid, placement)
    

    
    def clear_puzzle(self):
        """Clear the entire puzzle, resetting to empty state but preserving black squares."""
        # Preserve the black square pattern
        blocked_cells = self.grid.blocked_cells.copy()
        
        # Create new grid and restore black squares
        self.grid = CrosswordGrid(self.grid.size)
        for row, col in blocked_cells:
            self.grid.set_blocked(row, col, True)
        
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
            CrosswordValidator.validate_intersections(self.word_placements) and
            CrosswordValidator.validate_grid_connectivity(self.word_placements)
        )
    
    def fill_empty_cells_with_blocked(self) -> int:
        """
        Fill all empty cells in the grid with blocked cells.
        
        This is used at the end of crossword generation to finalize
        the puzzle by blocking any remaining unfilled cells.
        
        Returns:
            Number of cells that were blocked
        """
        blocked_count = 0
        
        for row in range(self.grid.size):
            for col in range(self.grid.size):
                # Check if cell is empty (not blocked and no letter)
                if not self.grid.is_blocked(row, col) and self.grid.is_empty(row, col):
                    self.grid.set_blocked(row, col, True)
                    blocked_count += 1
        
        return blocked_count
    
    def generate_puzzle_csp(self, max_iterations: int = 25000,
                            max_consecutive_failures: int = 5,
                           place_blocked_squares: bool = True,
                           target_fill: float = 90.0,
                           random_seed: Optional[int] = None,
                           preferred_length: int = 6) -> bool:
        """
        Generate a crossword puzzle using advanced CSP techniques, aiming for high fill rate.
        
        Uses sophisticated backtracking and constraint propagation to achieve target fill percentage.
        
        Args:
            max_iterations: Maximum solver iterations before giving up (default: 25000)
            max_consecutive_failures: Maximum consecutive failures before backtracking (default: 5)
            place_blocked_squares: Whether to place strategic black squares (default: True)
            target_fill: Target puzzle fill percentage (default: 90%)
            random_seed: Random seed for reproducible results (default: None for random)
            preferred_length: Preferred word length for quality and difficulty scoring (default: 6)
            
        Returns:
            True if puzzle generation achieved target fill, False otherwise
        """
        if not self.word_data_manager:
            raise ValueError("WordDataManager is required for CSP puzzle generation")
        
        # Import CSP solver (lazy import to avoid circular dependencies)
        from .csp_solver import CSPSolver
        
        # Clear any existing puzzle
        self.clear_puzzle()
        
        # Initialize CSP solver
        csp_solver = CSPSolver(self.word_data_manager, preferred_length)
        
        # Generate puzzle
        print(f"Starting CSP crossword generation with {max_iterations} iterations and target fill {target_fill}%")
        
        success = csp_solver.solve(
            self, 
            max_iterations=max_iterations,
            max_consecutive_failures=max_consecutive_failures,
            target_fill=target_fill,
            place_blocked_squares=place_blocked_squares,
            random_seed=random_seed
        )
        
        return success
    
    def place_word_in_slot(self, word: str, slot, blocked_number: int = 0, place_blocked_squares: bool = True, clue: str = "", blocked_ratio: float = 0.15) -> bool:
        """
        Place a word in a slot, adding blocked cells if the word is shorter than the slot.
        
        Args:
            word: Word to place
            slot: Slot object representing the available space
            blocked_number: Number of blocked cells in the grid
            place_blocked_squares: Whether to place blocked squares (default: True)
            clue: Associated clue (optional)
            blocked_ratio: Maximum ratio of blocked cells to total cells (default: 15%)
            
        Returns:
            True if word was successfully placed, False otherwise
        """
        if not slot.matches_word(word):
            return False
        
        # Automatically retrieve clue from word manager if not provided
        if not clue and self.word_data_manager:
            clue = self.word_data_manager.get_clue_for_word(word) or ""
        
        placement = WordPlacement(word.upper(), slot.row, slot.col, slot.direction, clue)
        
        # Check if word can be placed
        if not self.can_place_word_placement(placement):
            return False
        
        # Place the word first
        if not self.place_word_placement(placement):
            return False
        

        # Add blocked cells if word is shorter than slot
        if place_blocked_squares and len(word) < slot.length and blocked_number < self.grid.size * self.grid.size * blocked_ratio:
            pos = (slot.row, slot.col + len(word)) if slot.direction == Direction.ACROSS else (slot.row + len(word), slot.col)

            if self.grid.is_empty(pos[0], pos[1]) and not self.grid.is_blocked(pos[0], pos[1]):
                self.grid.set_blocked(pos[0], pos[1], True)
                
        
        return True