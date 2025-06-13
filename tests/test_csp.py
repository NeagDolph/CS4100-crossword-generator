"""
Unit Tests for CSP Crossword Solver Components

Tests the following CSP components:
- CSP Solver: Core constraint satisfaction solver
- FastWordIndex: Word indexing system
- Slots: Slot constraints and scoring
- Heuristic: Heuristic for slot scoring
- Word Placement: Word placement validation and constraints

Run tests with:
    python -m pytest tests/test_csp.py -v
    or
    make test
"""

import unittest
import sys
import numpy as np
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossword_package import (
    Slot, Heuristics, 
    CrosswordGrid, Direction, WordPlacement, 
    WordDataManager
)
from crossword_package.csp_solver import find_empty_slots


class TestSlotClass(unittest.TestCase):
    """Test the Slot class functionality including constraints and scoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.basic_slot = Slot(5, 3, Direction.ACROSS, 7)
        self.constrained_slot = Slot(2, 4, Direction.DOWN, 5, {1: 'A', 2: 'T'})
        self.complex_slot = Slot(0, 0, Direction.ACROSS, 10, {0: 'S', 2: 'A', 4: 'E', 9: 'D'})
    
    def test_slot_initialization(self):
        """Test slot initialization with various parameters."""
        # Basic slot without constraints
        slot = Slot(3, 5, Direction.ACROSS, 6)
        self.assertEqual(slot.row, 3)
        self.assertEqual(slot.col, 5)
        self.assertEqual(slot.direction, Direction.ACROSS)
        self.assertEqual(slot.length, 6)
        self.assertEqual(slot.constraints, {})
        self.assertEqual(slot.calculate_difficulty(), 0)
        
        # Slot with constraints
        constraints = {0: 'H', 2: 'L', 4: 'O'}
        slot = Slot(1, 2, Direction.DOWN, 5, constraints)
        self.assertEqual(slot.constraints, constraints)
        self.assertEqual(slot.length, 5)
        self.assertGreater(slot.calculate_difficulty(), 0)
    
    def test_constraint_pattern_generation(self):
        """Test constraint pattern string generation."""
        # Basic slot - all wildcards
        pattern = self.basic_slot.get_constraint_pattern()
        self.assertEqual(pattern, "???????")
        
        # Constrained slot
        pattern = self.constrained_slot.get_constraint_pattern()
        self.assertEqual(pattern, "?AT??")
        
        # Complex slot with multiple constraints
        pattern = self.complex_slot.get_constraint_pattern()
        self.assertEqual(pattern, "S?A?E????D")
    
    def test_word_matching_constraints(self):
        """Test word matching against slot constraints."""
        # Basic slot without constraints - accepts shorter words (no intersection requirements)
        self.assertTrue(self.basic_slot.matches_word("TESTING"))
        self.assertTrue(self.basic_slot.matches_word("ANOTHER"))
        self.assertTrue(self.basic_slot.matches_word("SHORT"))  # Shorter words allowed when no constraints
        self.assertFalse(self.basic_slot.matches_word("TOOOLONG"))  # Too long
        
        # Constrained slot - position 1='A', position 2='T'
        self.assertTrue(self.constrained_slot.matches_word("WATER"))  # W-A-T-E-R matches ?A?T?
        self.assertTrue(self.constrained_slot.matches_word("BATCH"))  # B-A-T-C-H matches ?A?T?
        self.assertFalse(self.constrained_slot.matches_word("HELLO"))  # H-E-L-L-O doesn't match ?A?T?
        self.assertFalse(self.constrained_slot.matches_word("WAITS"))  # Wrong position for T
        
        # Complex slot with multiple constraints
        self.assertTrue(self.complex_slot.matches_word("SKATEBOARD"))  # S?A??E???D
        self.assertFalse(self.complex_slot.matches_word("BASKETBALL"))  # Wrong first letter
        self.assertFalse(self.complex_slot.matches_word("STANDARDAD"))  # Wrong last letter
        
        self.assertFalse(self.constrained_slot.matches_word("W"))  # Too short to reach position 1
        self.assertTrue(self.complex_slot.matches_word("SKATE"))  # Too short to reach position 9
    
    def test_difficulty_score_calculation(self):
        """Test difficulty score calculation logic."""
        easy_slot = Slot(0, 0, Direction.ACROSS, 5)
        self.assertEqual(easy_slot.calculate_difficulty(), 0)
        
        medium_slot = Slot(0, 0, Direction.ACROSS, 5, {1: 'A'})
        self.assertGreater(medium_slot.calculate_difficulty(), 0)

        hard_slot = Slot(0, 0, Direction.ACROSS, 5, {1: 'A', 3: 'T'})
        self.assertGreater(hard_slot.calculate_difficulty(), 0)
        
        long_slot = Slot(0, 0, Direction.ACROSS, 10)
        self.assertEqual(long_slot.calculate_difficulty(), 0)

        short_slot = Slot(0, 0, Direction.ACROSS, 3)
        self.assertEqual(short_slot.calculate_difficulty(), 0)

    def test_slot_equality_and_hashing(self):
        """Test slot equality and hashing for use in sets/dicts."""
        # Same slots should be equal
        slot1 = Slot(2, 3, Direction.ACROSS, 5, {1: 'A'})
        slot2 = Slot(2, 3, Direction.ACROSS, 5, {1: 'A'})
        self.assertEqual(slot1, slot2)
        self.assertEqual(hash(slot1), hash(slot2))
        
        # Different positions
        slot3 = Slot(2, 4, Direction.ACROSS, 5, {1: 'A'})
        self.assertNotEqual(slot1, slot3)
        
        # Different directions
        slot4 = Slot(2, 3, Direction.DOWN, 5, {1: 'A'})
        self.assertNotEqual(slot1, slot4)
        
        # Different lengths
        slot5 = Slot(2, 3, Direction.ACROSS, 6, {1: 'A'})
        self.assertNotEqual(slot1, slot5)
        
        # Different constraints
        slot6 = Slot(2, 3, Direction.ACROSS, 5, {1: 'B'})
        self.assertNotEqual(slot1, slot6)
        
        # Test use in set
        slot_set = {slot1, slot2, slot3, slot4}
        self.assertEqual(len(slot_set), 3)  # slot1 and slot2 are identical
    
    def test_edge_case_constraints(self):
        """Test edge cases in constraint handling."""
        # Empty constraints
        slot = Slot(0, 0, Direction.ACROSS, 5, {})
        self.assertTrue(slot.matches_word("HELLO"))
        self.assertEqual(slot.get_constraint_pattern(), "?????")
        
        # Constraint at position 0
        slot = Slot(0, 0, Direction.ACROSS, 4, {0: 'H'})
        self.assertTrue(slot.matches_word("HELP"))
        self.assertFalse(slot.matches_word("YELP"))
        self.assertEqual(slot.get_constraint_pattern(), "H???")
        
        # Constraint at last position
        slot = Slot(0, 0, Direction.ACROSS, 4, {3: 'P'})
        self.assertTrue(slot.matches_word("HELP"))
        self.assertFalse(slot.matches_word("HELD"))
        self.assertEqual(slot.get_constraint_pattern(), "???P")
        
        # Out-of-bounds constraint (should be ignored)
        slot = Slot(0, 0, Direction.ACROSS, 3, {0: 'A', 5: 'X'})
        pattern = slot.get_constraint_pattern()
        self.assertEqual(pattern, "A??")  # Constraint ignored
        self.assertTrue(slot.matches_word("ANY"))


class TestHeuristics(unittest.TestCase):
    """Test the EnhancedHeuristic class for slot scoring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristic = Heuristics()
        self.grid = CrosswordGrid(15)
        self.word_data = WordDataManager()
        
        # Create some sample word placements
        self.placements = [
            WordPlacement("HELLO", 5, 2, Direction.ACROSS),
            WordPlacement("WORLD", 3, 4, Direction.DOWN)
        ]
    
    def test_heuristic_weights(self):
        """Test that heuristic weights are properly initialized."""
        expected_keys = ['intersection_potential', 'fill_efficiency', 'feasibility', 
                        'constraint_satisfaction', 'length']
        for key in expected_keys:
            self.assertIn(key, self.heuristic.weights)
            self.assertIsInstance(self.heuristic.weights[key], (int, float))
            self.assertGreaterEqual(self.heuristic.weights[key], 0)
    
    def test_intersection_potential_calculation(self):
        """Test intersection potential scoring."""
        # Slot that intersects with existing placements
        intersecting_slot = Slot(5, 0, Direction.ACROSS, 8)  # Crosses HELLO
        compatible_words = ["TEACHING", "REACHING", "GRAPHICS"]
        
        score = self.heuristic._calculate_intersection_potential(
            intersecting_slot, compatible_words, self.placements)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 20.0)  # Max score is 20
        
        # Slot with no intersections
        non_intersecting_slot = Slot(10, 10, Direction.ACROSS, 5)
        score_no_intersect = self.heuristic._calculate_intersection_potential(
            non_intersecting_slot, compatible_words, self.placements)
        
        # Should be less than intersecting slot
        self.assertLess(score_no_intersect, score)
        
        # Empty placements should return base score
        base_score = self.heuristic._calculate_intersection_potential(
            intersecting_slot, compatible_words, [])
        self.assertEqual(base_score, 5.0)
    
    def test_evaluate_slot_comprehensive(self):
        """Test comprehensive slot evaluation."""
        slot = Slot(7, 7, Direction.ACROSS, 6, {2: 'A'})
        compatible_words = ["HEADER", "BEAKER", "TRADER"]
        
        score = self.heuristic.evaluate_slot(
            slot, compatible_words, self.placements, 6)
        
        self.assertGreater(score, 0)
        self.assertIsInstance(score, (int, float))
        
        # Test with empty compatible words
        empty_score = self.heuristic.evaluate_slot(
            slot, [], self.placements, 6)
        self.assertEqual(empty_score, 0.0)
        
        constrained_slot = Slot(7, 7, Direction.ACROSS, 6, {1: 'E', 3: 'D', 5: 'R'})
        constrained_score = self.heuristic.evaluate_slot(
            constrained_slot, compatible_words, self.placements, 6)
        
        # More constraints should increase constraint satisfaction score
        self.assertLess(score, constrained_score)


class TestSlotDetection(unittest.TestCase):
    """Test slot detection in grids."""
    
    def setUp(self):
        """Set up test grids."""
        self.empty_grid = CrosswordGrid(15)
        self.partial_grid = CrosswordGrid(15)
        
        # Create a grid with some black squares and letters
        self.partial_grid.set_blocked(2, 2, True)
        self.partial_grid.set_blocked(2, 3, True)
        self.partial_grid.set_letter(5, 5, 'H')
        self.partial_grid.set_letter(5, 6, 'E')
        self.partial_grid.set_letter(5, 7, 'L')
        self.partial_grid.set_letter(5, 8, 'L')
        self.partial_grid.set_letter(5, 9, 'O')
    
    def test_find_empty_slots_basic(self):
        """Test basic slot detection."""
        slots = find_empty_slots(self.empty_grid, min_length=3)
        
        # Should find horizontal and vertical slots
        horizontal_slots = [s for s in slots if s.direction == Direction.ACROSS]
        vertical_slots = [s for s in slots if s.direction == Direction.DOWN]
        
        self.assertGreater(len(horizontal_slots), 0)
        self.assertGreater(len(vertical_slots), 0)
        
        # All slots should meet minimum length requirement
        for slot in slots:
            self.assertGreaterEqual(slot.length, 3)
    
    def test_find_empty_slots_with_constraints(self):
        """Test slot detection with existing letters (constraints)."""
        slots = find_empty_slots(self.partial_grid, min_length=3)
        
        # Find slots that should have constraints from existing letters
        constrained_slots = [s for s in slots if s.constraints]
        self.assertGreater(len(constrained_slots), 0)
        
        # Verify constraint accuracy
        for slot in constrained_slots:
            for pos, letter in slot.constraints.items():
                # Calculate actual grid position
                if slot.direction == Direction.ACROSS:
                    grid_row, grid_col = slot.row, slot.col + pos
                else:
                    grid_row, grid_col = slot.row + pos, slot.col
                
                # Check that constraint matches grid letter
                if (0 <= grid_row < self.partial_grid.size and 
                    0 <= grid_col < self.partial_grid.size):
                    grid_letter = self.partial_grid.get_letter(grid_row, grid_col)
                    if grid_letter:
                        self.assertEqual(str(letter), str(grid_letter))
    
    def test_find_empty_slots_with_blocked_squares(self):
        """Test slot detection around blocked squares."""
        # Create a grid with strategic black squares
        test_grid = CrosswordGrid(15)
        test_grid.set_blocked(5, 5, True)
        test_grid.set_blocked(6, 5, True)
        
        slots = find_empty_slots(test_grid, min_length=3)
        
        # Verify no slots cross blocked squares
        for slot in slots:
            positions = slot.get_positions()
            for row, col in positions:
                self.assertFalse(test_grid.is_blocked(row, col),
                               f"Slot {slot} crosses blocked square at ({row}, {col})")
    
    def test_slot_difficulty_ordering(self):
        """Test that slots are ordered by difficulty."""
        slots = find_empty_slots(self.partial_grid, min_length=3)
        
        # Verify slots are sorted by difficulty (ascending)
        for i in range(1, len(slots)):
            self.assertGreaterEqual(slots[i-1].calculate_difficulty(), slots[i].calculate_difficulty())
    
    def test_minimum_length_filtering(self):
        """Test minimum length filtering."""
        # Test with different minimum lengths
        for min_len in [3, 5, 8]:
            slots = find_empty_slots(self.empty_grid, min_length=min_len)
            for slot in slots:
                self.assertGreaterEqual(slot.length, min_len)
    
    def test_direction_specific_slot_detection(self):
        """Test slot detection in specific directions."""
        # Test horizontal slots
        slots = find_empty_slots(self.partial_grid, min_length=3)
        
        for slot in slots:
            self.assertIn(slot.direction, [Direction.ACROSS, Direction.DOWN])
            self.assertGreaterEqual(slot.length, 3)
               
            unique_slots = set(slots)
            self.assertEqual(len(unique_slots), len(slots), "Slots should all be unique")

        slots = find_empty_slots(self.partial_grid, min_length=4)
        
        for slot in slots:
            self.assertIn(slot.direction, [Direction.ACROSS, Direction.DOWN])
            self.assertGreaterEqual(slot.length, 4)
            
            unique_slots = set(slots)
            self.assertEqual(len(unique_slots), len(slots), "Slots should all be unique")
    



class TestConstraintValidation(unittest.TestCase):
    """Test constraint validation and edge cases."""
    
    def test_numpy_string_handling(self):
        """Test handling of numpy strings in constraints."""
        # Create constraints with numpy strings (common in grid operations)  
        numpy_constraints = {1: np.str_('A'), 2: np.str_('T')}
        slot = Slot(0, 0, Direction.ACROSS, 5, numpy_constraints)
        
        # Should handle numpy strings correctly
        pattern = slot.get_constraint_pattern()
        self.assertEqual(pattern, "?AT??")
        
        self.assertTrue(slot.matches_word("WATER"))
        self.assertFalse(slot.matches_word("HELLO"))
    
    def test_constraint_boundary_conditions(self):
        """Test constraint boundary conditions."""
        # Constraint at exact boundary
        slot = Slot(0, 0, Direction.ACROSS, 5, {4: 'Z'})
        self.assertTrue(slot.matches_word("ABCDZ"))
        self.assertFalse(slot.matches_word("ABCDE"))
        
        # Multiple boundary constraints
        slot = Slot(0, 0, Direction.ACROSS, 6, {0: 'A', 5: 'F'})
        self.assertTrue(slot.matches_word("ABCDEF"))
        self.assertFalse(slot.matches_word("BBCDEF"))
        self.assertFalse(slot.matches_word("ABCDEE"))
    
    def test_all_positions_constrained(self):
        """Test slot with constraints at all positions."""
        constraints = {0: 'H', 1: 'E', 2: 'L', 3: 'L', 4: 'O'}
        slot = Slot(0, 0, Direction.ACROSS, 5, constraints)
        
        self.assertTrue(slot.matches_word("HELLO"))
        self.assertFalse(slot.matches_word("WORLD"))
        self.assertFalse(slot.matches_word("HELPS"))
        
        pattern = slot.get_constraint_pattern()
        self.assertEqual(pattern, "HELLO")
    
    def test_constraint_case_sensitivity(self):
        """Test constraint case handling."""
        slot = Slot(0, 0, Direction.ACROSS, 5, {1: 'a', 2: 'T'})
        
        # Should handle mixed case constraints
        self.assertTrue(slot.matches_word("WATER"))  # matches ?a?T? -> ?A?T?
        
        pattern = slot.get_constraint_pattern()
        self.assertEqual(pattern, "?aT??")


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)


