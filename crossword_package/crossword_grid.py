"""
Crossword Grid Module

Core grid representation and basic operations for crossword puzzles.
Handles letter placement, removal, and grid state management.
"""

import numpy as np
import copy
from typing import Set, Tuple

class CrosswordGrid:
    """
    Core grid representation and basic operations.
    Handles letter placement, removal, and grid state management.
    """
    
    def __init__(self, size: int = 15):
        """
        Initialize a crossword grid.
        
        Args:
            size: Size of the square grid (default 15x15)
        """
        self.size = size
        # Grid stores letters, empty cells are '', blocked cells are '#'
        self.grid = np.full((size, size), '', dtype='<U1')
        # Track which cells are blocked (black squares)
        self.blocked_cells: Set[Tuple[int, int]] = set()
        # Track which cells are given (pre-filled) vs user input
        self.given_cells: Set[Tuple[int, int]] = set()
        self.user_cells: Set[Tuple[int, int]] = set()
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if position is within grid bounds.
        
        Args:
            row: Row position to check
            col: Column position to check
            
        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= row < self.size and 0 <= col < self.size
    
    def is_blocked(self, row: int, col: int) -> bool:
        """
        Check if a cell is blocked (black square).
        
        Args:
            row: Row position to check
            col: Column position to check
            
        Returns:
            True if cell is blocked, False otherwise
        """
        return (row, col) in self.blocked_cells
    
    def set_blocked(self, row: int, col: int, blocked: bool = True):
        """
        Set a cell as blocked or unblocked.
        
        Args:
            row: Row position
            col: Column position
            blocked: True to block, False to unblock
            
        Raises:
            ValueError: If position is invalid
        """
        if not self.is_valid_position(row, col):
            raise ValueError(f"Invalid position: ({row}, {col})")
        
        if blocked:
            self.blocked_cells.add((row, col))
            self.grid[row, col] = '#'
            # Remove from user/given cells if blocked
            self.given_cells.discard((row, col))
            self.user_cells.discard((row, col))
        else:
            self.blocked_cells.discard((row, col))
            self.grid[row, col] = ''
    
    def get_letter(self, row: int, col: int) -> str:
        """
        Get letter at position.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            Letter at position, or empty string if invalid position
        """
        if not self.is_valid_position(row, col):
            return ''
        return self.grid[row, col]
    
    def set_letter(self, row: int, col: int, letter: str, is_given: bool = False):
        """
        Set letter at position.
        
        Args:
            row: Row position
            col: Column position
            letter: Letter to place (will be uppercased)
            is_given: Whether this is a given letter or user input
            
        Raises:
            ValueError: If position is invalid or blocked
        """
        if not self.is_valid_position(row, col):
            raise ValueError(f"Invalid position: ({row}, {col})")
        if self.is_blocked(row, col):
            raise ValueError(f"Cannot place letter in blocked cell: ({row}, {col})")
        
        self.grid[row, col] = letter.upper() if letter else ''
        
        # Track whether this is given or user input
        pos = (row, col)
        if is_given:
            self.given_cells.add(pos)
            self.user_cells.discard(pos)
        else:
            self.user_cells.add(pos)
            self.given_cells.discard(pos)
    
    def remove_letter(self, row: int, col: int):
        """
        Remove letter at position (only if not given).
        
        Args:
            row: Row position
            col: Column position
            
        Raises:
            ValueError: If trying to remove a given letter
        """
        if (row, col) in self.given_cells:
            raise ValueError(f"Cannot remove given letter at ({row}, {col})")
        self.set_letter(row, col, '')
        self.user_cells.discard((row, col))
    
    def clear_user_input(self):
        """Clear all user input, keeping only given letters."""
        for row, col in list(self.user_cells):
            self.grid[row, col] = ''
        self.user_cells.clear()
    
    def is_empty(self, row: int, col: int) -> bool:
        """
        Check if a cell is empty (not blocked, no letter).
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            True if cell is empty, False otherwise
        """
        if not self.is_valid_position(row, col):
            return False
        return not self.is_blocked(row, col) and self.get_letter(row, col) == ''
    
    def get_empty_cells(self) -> Set[Tuple[int, int]]:
        """
        Get all empty cells in the grid.
        
        Returns:
            Set of (row, col) tuples for empty cells
        """
        empty_cells = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.is_empty(row, col):
                    empty_cells.add((row, col))
        return empty_cells
    
    def get_filled_cells(self) -> Set[Tuple[int, int]]:
        """
        Get all filled cells in the grid.
        
        Returns:
            Set of (row, col) tuples for filled cells
        """
        filled_cells = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.get_letter(row, col) and not self.is_blocked(row, col):
                    filled_cells.add((row, col))
        return filled_cells
    
    def copy(self):
        """
        Create a deep copy of the grid.
        
        Returns:
            New CrosswordGrid instance with identical state
        """
        new_grid = CrosswordGrid(self.size)
        new_grid.grid = self.grid.copy()
        new_grid.blocked_cells = self.blocked_cells.copy()
        new_grid.given_cells = self.given_cells.copy()
        new_grid.user_cells = self.user_cells.copy()
        return new_grid
    
    def __str__(self):
        """
        Pretty print the grid.
        
        Returns:
            String representation of the grid
        """
        lines = []
        for row in self.grid:
            line = ""
            for cell in row:
                if cell == '':
                    line += '.'
                elif cell == '#':
                    line += '█'  # Block character
                else:
                    line += cell
                line += ' '
            lines.append(line)
        return "\n".join(lines)
    
    def __eq__(self, other):
        """Check equality with another grid."""
        if not isinstance(other, CrosswordGrid):
            return False
        return (
            self.size == other.size and
            np.array_equal(self.grid, other.grid) and
            self.blocked_cells == other.blocked_cells and
            self.given_cells == other.given_cells and
            self.user_cells == other.user_cells
        ) 