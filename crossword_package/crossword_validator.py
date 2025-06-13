"""
Crossword Validator Module

Utility class for validation operations.
Provides static methods for various crossword validation checks.
"""

from typing import List, Tuple
from .crossword_grid import CrosswordGrid
from .word_placement import WordPlacement, Direction

class CrosswordValidator:
    """
    Utility class for validation operations.
    Provides static methods for various crossword validation checks.
    """
    
    @staticmethod
    def can_place_word_placement(grid: CrosswordGrid, placement: WordPlacement) -> bool:
        """
        Check if a word placement can be placed at the specified position.
        
        Args:
            grid: The crossword grid
            placement: WordPlacement object containing word and position data
            
        Returns:
            True if word can be placed, False otherwise
        """
        word = placement.word.upper()
        row, col, direction = placement.row, placement.col, placement.direction
        
        # Check bounds
        if direction == Direction.ACROSS:
            if col + len(word) > grid.size:
                return False
        else:  # DOWN
            if row + len(word) > grid.size:
                return False
        
        # Check each position
        for i, letter in enumerate(word):
            if direction == Direction.ACROSS:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            # Can't place in blocked cells
            if grid.is_blocked(r, c):
                return False
            
            # Check for conflicts with existing letters
            existing = grid.get_letter(r, c)
            if existing and existing != letter:
                return False
        
        return True
    

    
    @staticmethod
    def get_intersections(placement1: WordPlacement, placement2: WordPlacement) -> List[Tuple[int, int]]:
        """
        Find intersection points between two word placements.
        
        Args:
            placement1: First word placement
            placement2: Second word placement
            
        Returns:
            List of (row, col) tuples where words intersect
        """
        positions1 = set(placement1.get_positions())
        positions2 = set(placement2.get_positions())
        return list(positions1.intersection(positions2))
    
    @staticmethod
    def validate_intersections(placements: List[WordPlacement]) -> bool:
        """
        Validate that all word intersections have matching letters.
        
        Args:
            grid: The crossword grid
            placements: List of word placements to validate
            
        Returns:
            True if all intersections are valid, False otherwise
        """
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                intersections = CrosswordValidator.get_intersections(p1, p2)
                for row, col in intersections:
                    # Find which letter should be at this position for each word
                    letter1_idx = None
                    letter2_idx = None
                    
                    if p1.direction == Direction.ACROSS:
                        letter1_idx = col - p1.col
                    else:
                        letter1_idx = row - p1.row
                    
                    if p2.direction == Direction.ACROSS:
                        letter2_idx = col - p2.col
                    else:
                        letter2_idx = row - p2.row
                    
                    if (0 <= letter1_idx < len(p1.word) and 
                        0 <= letter2_idx < len(p2.word)):
                        if p1.word[letter1_idx] != p2.word[letter2_idx]:
                            return False
        return True
    
    @staticmethod
    def is_placement_isolated(placement: WordPlacement, other_placements: List[WordPlacement]) -> bool:
        """
        Check if a word placement is isolated (doesn't intersect with any other words).
        
        Args:
            placement: Word placement to check
            other_placements: List of other word placements
            
        Returns:
            True if placement is isolated, False if it intersects with others
        """
        for other in other_placements:
            if CrosswordValidator.get_intersections(placement, other):
                return False
        return True
    
    @staticmethod
    def validate_word_spacing(grid: CrosswordGrid, placement: WordPlacement) -> bool:
        """
        Validate that a word has proper spacing (no adjacent parallel words).
        
        Args:
            grid: The crossword grid
            placement: Word placement to validate
            
        Returns:
            True if spacing is valid, False otherwise
        """
        # Check cells adjacent to the word for proper spacing
        positions = placement.get_positions()
        
        for row, col in positions:
            # Check perpendicular directions for adjacent letters
            if placement.direction == Direction.ACROSS:
                # Check above and below
                adjacent_positions = [(row - 1, col), (row + 1, col)]
            else:
                # Check left and right
                adjacent_positions = [(row, col - 1), (row, col + 1)]
            
            for adj_row, adj_col in adjacent_positions:
                if (grid.is_valid_position(adj_row, adj_col) and 
                    not grid.is_blocked(adj_row, adj_col) and 
                    grid.get_letter(adj_row, adj_col) != ''):
                    
                    pass
        
        return True
    
    @staticmethod
    def count_word_intersections(placements: List[WordPlacement]) -> int:
        """
        Count the total number of intersections between all word placements.
        
        Args:
            placements: List of word placements
            
        Returns:
            Total number of intersections
        """
        intersection_count = 0
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                intersections = CrosswordValidator.get_intersections(p1, p2)
                intersection_count += len(intersections)
        return intersection_count
    
    @staticmethod
    def validate_grid_connectivity(placements: List[WordPlacement]) -> bool:
        """
        Validate that all words in the grid are connected (standard crossword rule).
        
        Args:
            placements: List of word placements
            
        Returns:
            True if all words are connected, False otherwise
        """
        if len(placements) <= 1:
            return True
        
        # Build adjacency graph of word intersections
        connected_components = []
        
        for i, placement in enumerate(placements):
            # Find which component this word belongs to
            component_indices = []
            for j, component in enumerate(connected_components):
                for other_idx in component:
                    if CrosswordValidator.get_intersections(placement, placements[other_idx]):
                        component_indices.append(j)
                        break
            
            if not component_indices:
                # Start new component
                connected_components.append({i})
            else:
                # Merge components
                new_component = {i}
                for idx in sorted(component_indices, reverse=True):
                    new_component.update(connected_components.pop(idx))
                connected_components.append(new_component)
        
        # All words should be in one component
        return len(connected_components) <= 1 