"""
Comprehensive Unit Tests for Crossword Package

Tests all major components: Grid, Validator, Creator, Player, and Puzzle classes.
Run with: python -m pytest tests/test_crossword.py -v 
Or run `make test`
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossword_package import (
    CrosswordGrid, CrosswordValidator, CrosswordCreator, 
    CrosswordPlayer, CrosswordPuzzle, Direction, WordPlacement, 
    WordDataManager
)

# Create a test word data manager with limited data for testing
class TestWordDataManager(WordDataManager):
    """Test word data manager with hardcoded data for unit tests."""
    
    def __init__(self):
        """Initialize with test data instead of loading CSV."""
        self.csv_file_path = "test_data.csv"
        self.word_clues = []
        self.word_to_clues = {}
        self._loaded = True
        
        # Test words with clues
        test_data = [
            ("HELLO", "Greeting"),
            ("WORLD", "Earth"),
            ("HELP", "Assistance"),
            ("HOUSE", "Dwelling"),
            ("APPLE", "Red fruit"),
            ("BANANA", "Yellow fruit"),
            ("TEST", "Examination"),
            ("EXAMPLE", "I don't understand. Give me an __"),
        ]
        
        from crossword_package.word_data import WordClue
        for word, clue in test_data:
            word_clue = WordClue(word, clue)
            self.word_clues.append(word_clue)
            if word not in self.word_to_clues:
                self.word_to_clues[word] = []
            self.word_to_clues[word].append(word_clue)

class TestCrosswordGrid(unittest.TestCase):
    """Test cases for CrosswordGrid class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grid = CrosswordGrid(5)  # 5x5 grid for testing
    
    def test_initialization(self):
        """Test grid initialization"""
        self.assertEqual(self.grid.size, 5)
        self.assertEqual(len(self.grid.blocked_cells), 0)
        self.assertEqual(len(self.grid.given_cells), 0)
        self.assertEqual(len(self.grid.user_cells), 0)
    
    def test_valid_position(self):
        """Test position validation"""
        self.assertTrue(self.grid.is_valid_position(0, 0))
        self.assertTrue(self.grid.is_valid_position(4, 4))
        self.assertFalse(self.grid.is_valid_position(-1, 0))
        self.assertFalse(self.grid.is_valid_position(5, 0))
        self.assertFalse(self.grid.is_valid_position(0, 5))
    
    def test_set_get_letter(self):
        """Test letter setting and getting"""
        self.grid.set_letter(2, 2, 'A')
        self.assertEqual(self.grid.get_letter(2, 2), 'A')
        
        # Test lowercase conversion
        self.grid.set_letter(1, 1, 'b')
        self.assertEqual(self.grid.get_letter(1, 1), 'B')
        
        # Test invalid position
        self.assertEqual(self.grid.get_letter(-1, 0), '')
    
    def test_blocked_cells(self):
        """Test blocked cell functionality"""
        self.grid.set_blocked(2, 2, True)
        self.assertTrue(self.grid.is_blocked(2, 2))
        
        # Should not be able to place letter in blocked cell
        with self.assertRaises(ValueError):
            self.grid.set_letter(2, 2, 'A')
        
        # Unblock the cell
        self.grid.set_blocked(2, 2, False)
        self.assertFalse(self.grid.is_blocked(2, 2))
        self.grid.set_letter(2, 2, 'A')  # Should work now
    
    def test_given_vs_user_cells(self):
        """Test distinction between given and user cells"""
        self.grid.set_letter(1, 1, 'A', is_given=True)
        self.grid.set_letter(2, 2, 'B', is_given=False)
        
        self.assertIn((1, 1), self.grid.given_cells)
        self.assertIn((2, 2), self.grid.user_cells)
        
        # Should not be able to remove given letter
        with self.assertRaises(ValueError):
            self.grid.remove_letter(1, 1)
        
        # Should be able to remove user letter
        self.grid.remove_letter(2, 2)
        self.assertEqual(self.grid.get_letter(2, 2), '')
    
    def test_clear_user_input(self):
        """Test clearing user input while keeping given letters"""
        self.grid.set_letter(1, 1, 'A', is_given=True)
        self.grid.set_letter(2, 2, 'B', is_given=False)
        
        self.grid.clear_user_input()
        
        self.assertEqual(self.grid.get_letter(1, 1), 'A')  # Given letter remains
        self.assertEqual(self.grid.get_letter(2, 2), '')   # User letter cleared
    
    def test_grid_copy(self):
        """Test grid copying functionality"""
        self.grid.set_letter(1, 1, 'A')
        self.grid.set_blocked(2, 2, True)
        
        copied_grid = self.grid.copy()
        
        self.assertEqual(copied_grid.get_letter(1, 1), 'A')
        self.assertTrue(copied_grid.is_blocked(2, 2))
        self.assertEqual(copied_grid.size, self.grid.size)
        
        # Ensure it's a deep copy
        self.grid.set_letter(3, 3, 'Z')
        self.assertEqual(copied_grid.get_letter(3, 3), '')
    
    def test_empty_and_filled_cells(self):
        """Test empty and filled cell detection"""
        self.grid.set_letter(1, 1, 'A')
        self.grid.set_blocked(2, 2, True)
        
        empty_cells = self.grid.get_empty_cells()
        filled_cells = self.grid.get_filled_cells()
        
        self.assertNotIn((1, 1), empty_cells)  # Has letter
        self.assertNotIn((2, 2), empty_cells)  # Blocked
        self.assertIn((0, 0), empty_cells)     # Empty
        
        self.assertIn((1, 1), filled_cells)    # Has letter
        self.assertNotIn((2, 2), filled_cells) # Blocked
        self.assertNotIn((0, 0), filled_cells) # Empty

class TestWordPlacement(unittest.TestCase):
    """Test cases for WordPlacement class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.across_word = WordPlacement("HELLO", 2, 1, Direction.ACROSS, "Greeting")
        self.down_word = WordPlacement("WORLD", 1, 3, Direction.DOWN, "Earth")
    
    def test_word_placement_creation(self):
        """Test WordPlacement creation"""
        self.assertEqual(self.across_word.word, "HELLO")
        self.assertEqual(self.across_word.row, 2)
        self.assertEqual(self.across_word.col, 1)
        self.assertEqual(self.across_word.direction, Direction.ACROSS)
        self.assertEqual(self.across_word.clue, "Greeting")
    
    def test_get_positions(self):
        """Test position calculation"""
        positions = self.across_word.get_positions()
        expected = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]
        self.assertEqual(positions, expected)
        
        positions = self.down_word.get_positions()
        expected = [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]
        self.assertEqual(positions, expected)
    
    def test_get_end_position(self):
        """Test end position calculation"""
        self.assertEqual(self.across_word.get_end_position(), (2, 5))
        self.assertEqual(self.down_word.get_end_position(), (5, 3))
    
    def test_get_letter_at_position(self):
        """Test getting letter at specific position"""
        self.assertEqual(self.across_word.get_letter_at_position(2, 1), 'H')
        self.assertEqual(self.across_word.get_letter_at_position(2, 2), 'E')
        self.assertEqual(self.across_word.get_letter_at_position(2, 5), 'O')
        self.assertEqual(self.across_word.get_letter_at_position(3, 1), '')  # Not in word
        
        self.assertEqual(self.down_word.get_letter_at_position(1, 3), 'W')
        self.assertEqual(self.down_word.get_letter_at_position(3, 3), 'R')

class TestCrosswordValidator(unittest.TestCase):
    """Test cases for CrosswordValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grid = CrosswordGrid(10)
        # Use words that actually intersect: HELLO and HELP share L
        # HELLO across at (2,1): H(2,1) E(2,2) L(2,3) L(2,4) O(2,5)
        # HELP down at (0,3): H(0,3) E(1,3) L(2,3) P(3,3)
        # They intersect at (2,3) where both have 'L'
        self.word1 = WordPlacement("HELLO", 2, 1, Direction.ACROSS)
        self.word2 = WordPlacement("HELP", 0, 3, Direction.DOWN)
    
    def test_can_place_word(self):
        """Test word placement validation"""
        # Should be able to place word in empty grid
        self.assertTrue(CrosswordValidator.can_place_word(
            self.grid, "HELLO", 2, 1, Direction.ACROSS))
        
        # Should not be able to place word out of bounds
        self.assertFalse(CrosswordValidator.can_place_word(
            self.grid, "HELLO", 2, 8, Direction.ACROSS))
        
        # Place a letter that conflicts
        self.grid.set_letter(2, 3, 'X')
        self.assertFalse(CrosswordValidator.can_place_word(
            self.grid, "HELLO", 2, 1, Direction.ACROSS))
        
        # Place a letter that matches
        self.grid.set_letter(2, 3, 'L')
        self.assertTrue(CrosswordValidator.can_place_word(
            self.grid, "HELLO", 2, 1, Direction.ACROSS))
    
    def test_get_intersections(self):
        """Test intersection detection"""
        intersections = CrosswordValidator.get_intersections(self.word1, self.word2)
        expected = [(2, 3)]  # HELLO and HELP intersect at (2, 3) with 'L'
        self.assertEqual(intersections, expected)
    
    def test_validate_intersections(self):
        """Test intersection validation"""
        # These words should have valid intersection (L from HELLO, L from HELP)
        self.assertTrue(CrosswordValidator.validate_intersections(
            self.grid, [self.word1, self.word2]))
        
        # Create words that DO intersect properly - use HELLO and WORLD that intersect at O
        word_a = WordPlacement("HELLO", 2, 1, Direction.ACROSS)  # H-E-L-L-O at row 2, cols 1-5
        word_b = WordPlacement("WORLD", 1, 5, Direction.DOWN)    # W-O-R-L-D at col 5, rows 1-5
        # They intersect at (2,5) where HELLO has 'O' and WORLD has 'O' - this should work!
        self.assertTrue(CrosswordValidator.validate_intersections(
            self.grid, [word_a, word_b]))
        
        # Create conflicting words - HELLO and HOUSE conflict at (2,3): L vs U
        bad_word = WordPlacement("HOUSE", 1, 3, Direction.DOWN)  # H-O-U-S-E, U at (3,3) conflicts with L
        self.assertFalse(CrosswordValidator.validate_intersections(
            self.grid, [self.word1, bad_word]))
    
    def test_get_valid_placements(self):
        """Test finding valid placements"""
        placements = CrosswordValidator.get_valid_placements(self.grid, "TEST")
        
        # Should find many valid placements in empty grid
        self.assertGreater(len(placements), 0)
        
        # Block some cells and test again
        for i in range(5):
            self.grid.set_blocked(i, i, True)
        
        new_placements = CrosswordValidator.get_valid_placements(self.grid, "TEST")
        self.assertLess(len(new_placements), len(placements))
    
    def test_count_word_intersections(self):
        """Test intersection counting"""
        count = CrosswordValidator.count_word_intersections([self.word1, self.word2])
        self.assertEqual(count, 1)  # One intersection between HELLO and HELP
        
        # No intersections with single word
        count = CrosswordValidator.count_word_intersections([self.word1])
        self.assertEqual(count, 0)
    
    def test_validate_grid_connectivity(self):
        """Test grid connectivity validation"""
        # Two intersecting words should be connected
        self.assertTrue(CrosswordValidator.validate_grid_connectivity(
            [self.word1, self.word2]))
        
        # Single word is connected
        self.assertTrue(CrosswordValidator.validate_grid_connectivity([self.word1]))
        
        # No words is connected
        self.assertTrue(CrosswordValidator.validate_grid_connectivity([]))
        
        # Two non-intersecting words should not be connected
        isolated_word = WordPlacement("ISOLATED", 8, 8, Direction.ACROSS)
        self.assertFalse(CrosswordValidator.validate_grid_connectivity(
            [self.word1, isolated_word]))

class TestCrosswordCreator(unittest.TestCase):
    """Test cases for CrosswordCreator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grid = CrosswordGrid(10)
        self.test_word_manager = TestWordDataManager()
        self.creator = CrosswordCreator(self.grid, self.test_word_manager)
    
    def test_place_word(self):
        """Test word placement"""
        success = self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.assertTrue(success)
        self.assertEqual(len(self.creator.word_placements), 1)
        self.assertEqual(self.grid.get_letter(2, 2), 'H')
        self.assertEqual(self.grid.get_letter(2, 6), 'O')
    
    def test_place_conflicting_word(self):
        """Test placing conflicting words"""
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        
        # Try to place conflicting word
        success = self.creator.place_word("HOUSE", 1, 4, Direction.DOWN)
        self.assertFalse(success)  # Should fail due to H vs L conflict
    
    def test_place_intersecting_word(self):
        """Test placing intersecting words"""
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        
        # Place intersecting word - HELLO at (2,2-6), WORLD at (1,6) down intersects at (2,6) with O
        success = self.creator.place_word("WORLD", 1, 6, Direction.DOWN)
        self.assertTrue(success)
        self.assertEqual(len(self.creator.word_placements), 2)
    
    def test_remove_word(self):
        """Test word removal"""
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        placement = self.creator.word_placements[0]
        
        success = self.creator.remove_word(placement)
        self.assertTrue(success)
        self.assertEqual(len(self.creator.word_placements), 0)
        self.assertEqual(self.grid.get_letter(2, 2), '')
    
    def test_remove_word_with_intersection(self):
        """Test removing word that intersects with another"""
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.creator.place_word("WORLD", 1, 6, Direction.DOWN)
        
        # Remove first word - intersection letter should remain
        placement = self.creator.word_placements[0]
        self.creator.remove_word(placement)
        
        self.assertEqual(len(self.creator.word_placements), 1)
        self.assertEqual(self.grid.get_letter(2, 6), 'O')  # Intersection remains
        self.assertEqual(self.grid.get_letter(2, 2), '')   # Non-intersection removed
    
    def test_get_puzzle_statistics(self):
        """Test puzzle statistics"""
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.creator.place_word("WORLD", 1, 6, Direction.DOWN)
        
        stats = self.creator.get_puzzle_statistics()
        
        self.assertEqual(stats['word_count'], 2)
        self.assertGreater(stats['filled_cells'], 0)
        self.assertGreater(stats['intersection_count'], 0)
        self.assertTrue(stats['is_connected'])
    
    def test_clear_puzzle(self):
        """Test puzzle clearing"""
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.creator.clear_puzzle()
        
        self.assertEqual(len(self.creator.word_placements), 0)
        self.assertEqual(self.creator.grid.get_letter(2, 2), '')  # Check creator's grid, not self.grid

class TestCrosswordPlayer(unittest.TestCase):
    """Test cases for CrosswordPlayer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grid = CrosswordGrid(10)
        self.test_word_manager = TestWordDataManager()
        self.creator = CrosswordCreator(self.grid, self.test_word_manager)
        self.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.creator.place_word("WORLD", 1, 6, Direction.DOWN)  # Intersects at (2,6) with O
        self.player = CrosswordPlayer(self.creator)
    
    def test_player_initialization(self):
        """Test player initialization"""
        self.assertEqual(self.player.mistakes, 0)
        self.assertEqual(len(self.player.word_placements), 2)
        
        # Play grid should be cleared of letters but keep structure
        self.assertEqual(self.player.play_grid.get_letter(2, 2), '')
        self.assertEqual(self.player.solution_grid.get_letter(2, 2), 'H')
    
    def test_make_guess(self):
        """Test making guesses"""
        # Correct guess
        correct = self.player.make_guess(2, 2, 'H')
        self.assertTrue(correct)
        self.assertEqual(self.player.mistakes, 0)
        self.assertEqual(self.player.play_grid.get_letter(2, 2), 'H')
        
        # Incorrect guess
        correct = self.player.make_guess(2, 3, 'X')
        self.assertFalse(correct)
        self.assertEqual(self.player.mistakes, 1)
    
    def test_make_guess_given_cell(self):
        """Test that you can't modify given cells"""
        # Set a given cell
        self.player.play_grid.set_letter(1, 1, 'G', is_given=True)
        
        with self.assertRaises(ValueError):
            self.player.make_guess(1, 1, 'A')
    
    def test_remove_guess(self):
        """Test removing guesses"""
        self.player.make_guess(2, 2, 'H')
        self.player.remove_guess(2, 2)
        self.assertEqual(self.player.play_grid.get_letter(2, 2), '')
    
    def test_check_word_completion(self):
        """Test word completion checking"""
        word_placement = self.player.word_placements[0]  # HELLO
        
        # Word not complete initially
        self.assertFalse(self.player.check_word_completion(word_placement))
        
        # Fill in the word correctly
        for i, letter in enumerate("HELLO"):
            self.player.make_guess(2, 2 + i, letter)
        
        self.assertTrue(self.player.check_word_completion(word_placement))
    
    def test_is_complete(self):
        """Test puzzle completion"""
        self.assertFalse(self.player.is_complete())
        
        # Fill in all letters correctly
        for placement in self.player.word_placements:
            for i, letter in enumerate(placement.word):
                row, col = placement.get_positions()[i]
                self.player.make_guess(row, col, letter)
        
        self.assertTrue(self.player.is_complete())
    
    def test_get_progress(self):
        """Test progress tracking"""
        progress = self.player.get_progress()
        
        self.assertEqual(progress['completed_words'], 0)
        self.assertEqual(progress['mistakes'], 0)
        
        # Make some progress
        self.player.make_guess(2, 2, 'H')
        
        progress = self.player.get_progress()
        self.assertGreater(progress['completion_percentage'], 0)
    
    def test_get_clue_at_position(self):
        """Test getting clues at a specific position"""
        # Test position with one word (HELLO at 2,2)
        clues = self.player.get_clue_at_position(2, 2)
        self.assertEqual(len(clues), 1)
        clue = clues[0]
        self.assertEqual(clue["direction"], 'across')
        self.assertEqual(clue["clue"], "Greeting")  # Expected clue from TestWordDataManager
        self.assertIn("positions", clue)
        
        # Test position with two words (intersection at 2,6)
        clues = self.player.get_clue_at_position(2, 6)
        self.assertEqual(len(clues), 2)
        
        # Should have both across and down clues
        directions = [clue["direction"] for clue in clues]
        self.assertIn('across', directions)
        self.assertIn('down', directions)
        
        # Test position with no words
        clues = self.player.get_clue_at_position(0, 0)
        self.assertEqual(len(clues), 0)
    
    def test_get_clue_at_position_with_custom_clues(self):
        """Test getting clues when word placements have custom clues"""
        # Set custom clues
        self.player.word_placements[0].clue = "A greeting"
        self.player.word_placements[1].clue = "Our planet"
        
        clues = self.player.get_clue_at_position(2, 2)
        self.assertEqual(len(clues), 1)
        clue = clues[0]
        self.assertEqual(clue["clue"], "A greeting")
        
        clues = self.player.get_clue_at_position(2, 6)
        clue_texts = [clue["clue"] for clue in clues]
        self.assertIn("A greeting", clue_texts)
        self.assertIn("Our planet", clue_texts)
    
    def test_get_clue_for_word(self):
        """Test getting clue for a specific word placement"""
        word_placement = self.player.word_placements[0]
        
        # Test with clue from TestWordDataManager
        clue = self.player.get_clue_for_word(word_placement)
        self.assertEqual(clue, "Greeting")  # Expected clue from TestWordDataManager
        
        # Test with custom clue override
        word_placement.clue = "A friendly greeting"
        clue = self.player.get_clue_for_word(word_placement)
        self.assertEqual(clue, "A friendly greeting")
        
        # Test with empty clue (should fall back to default)
        word_placement.clue = ""
        clue = self.player.get_clue_for_word(word_placement)
        self.assertIn('Word', clue)
    
    def test_find_word_by_start_position(self):
        """Test finding words by their start position and direction"""
        # Find HELLO (starts at 2,2 across)
        word = self.player.find_word_by_start_position(2, 2, 'across')
        self.assertIsNotNone(word)
        self.assertEqual(word.word, 'HELLO')
        self.assertEqual(word.direction.value, 'across')
        
        # Find WORLD (starts at 1,6 down)
        word = self.player.find_word_by_start_position(1, 6, 'down')
        self.assertIsNotNone(word)
        self.assertEqual(word.word, 'WORLD')
        self.assertEqual(word.direction.value, 'down')
        
        # Test non-existent position
        word = self.player.find_word_by_start_position(0, 0, 'across')
        self.assertIsNone(word)
        
        # Test wrong direction at existing position
        word = self.player.find_word_by_start_position(2, 2, 'down')
        self.assertIsNone(word)
        
        # Test case insensitive direction matching
        word = self.player.find_word_by_start_position(2, 2, 'ACROSS')
        self.assertIsNone(word)  # Should be exact match
    
    def test_get_word_at_position(self):
        """Test getting all words at a specific position"""
        # Test position with one word
        words = self.player.get_word_at_position(2, 2)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].word, 'HELLO')
        
        # Test intersection position with two words
        words = self.player.get_word_at_position(2, 6)
        self.assertEqual(len(words), 2)
        word_names = [w.word for w in words]
        self.assertIn('HELLO', word_names)
        self.assertIn('WORLD', word_names)
        
        # Test position with no words
        words = self.player.get_word_at_position(0, 0)
        self.assertEqual(len(words), 0)
        
        # Test edge position of a word
        words = self.player.get_word_at_position(2, 3)  # Middle of HELLO
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].word, 'HELLO')
    
    def test_clue_functionality_integration(self):
        """Test integration of all clue functionality together"""
        # Set up custom clues
        hello_placement = self.player.find_word_by_start_position(2, 2, 'across')
        world_placement = self.player.find_word_by_start_position(1, 6, 'down')
        
        hello_placement.clue = "Common greeting"
        world_placement.clue = "The Earth"
        
        # Test getting clue for found word
        clue = self.player.get_clue_for_word(hello_placement)
        self.assertEqual(clue, "Common greeting")
        
        # Test getting clues at intersection
        clues = self.player.get_clue_at_position(2, 6)
        self.assertEqual(len(clues), 2)
        
        # Verify both clues are present
        clue_texts = [clue["clue"] for clue in clues]
        self.assertIn("Common greeting", clue_texts)
        self.assertIn("The Earth", clue_texts)
        
        # Verify directions are correct
        directions = [clue["direction"] for clue in clues]
        self.assertIn('across', directions)
        self.assertIn('down', directions)
        
        # Verify positions are included
        for clue in clues:
            self.assertIn("positions", clue)
            self.assertIsInstance(clue["positions"], list)
    
    def test_get_crossword_clues(self):
        """Test clue organization"""
        clues = self.player.get_crossword_clues()
        
        self.assertIn('across', clues)
        self.assertIn('down', clues)
        self.assertEqual(len(clues['across']), 1)  # HELLO
        self.assertEqual(len(clues['down']), 1)    # WORLD
    
    def test_reset_puzzle(self):
        """Test puzzle reset"""
        self.player.make_guess(2, 2, 'H')
        
        self.player.reset_puzzle()
        
        self.assertEqual(self.player.mistakes, 0)
        self.assertEqual(self.player.play_grid.get_letter(2, 2), '')

class TestCrosswordPuzzle(unittest.TestCase):
    """Test cases for CrosswordPuzzle orchestrator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_word_manager = TestWordDataManager()
        self.puzzle = CrosswordPuzzle(10, self.test_word_manager)
    
    def test_puzzle_initialization(self):
        """Test puzzle initialization"""
        self.assertEqual(self.puzzle.grid.size, 10)
        self.assertFalse(self.puzzle.is_created)
        self.assertIsNone(self.puzzle.player)
    
    def test_start_game_without_puzzle(self):
        """Test that starting game without puzzle raises error"""
        with self.assertRaises(ValueError):
            self.puzzle.start_game()
    
    def test_start_game_with_puzzle(self):
        """Test starting game with created puzzle"""
        # Create a simple puzzle manually
        self.puzzle.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.puzzle.is_created = True
        
        player = self.puzzle.start_game()
        
        self.assertIsNotNone(player)
        self.assertIsInstance(player, CrosswordPlayer)
        self.assertEqual(self.puzzle.player, player)
    
    def test_get_puzzle_info(self):
        """Test puzzle information retrieval"""
        info = self.puzzle.get_puzzle_info()
        
        self.assertIn('is_created', info)
        self.assertIn('grid_size', info)
        self.assertIn('has_active_game', info)
        
        # Create puzzle and test again
        self.puzzle.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.puzzle.is_created = True
        
        info = self.puzzle.get_puzzle_info()
        self.assertTrue(info['is_created'])
    
    def test_reset_game(self):
        """Test game reset"""
        # Setup game
        self.puzzle.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.puzzle.is_created = True
        player = self.puzzle.start_game()
        
        # Make some moves
        player.make_guess(2, 2, 'H')
        
        # Reset
        self.puzzle.reset_game()
        
        self.assertEqual(player.mistakes, 0)
    
    def test_save_load_puzzle_state(self):
        """Test saving and loading puzzle state"""
        # Create puzzle
        self.puzzle.creator.place_word("HELLO", 2, 2, Direction.ACROSS)
        self.puzzle.is_created = True
        
        # Save state
        state = self.puzzle.save_puzzle_state()
        
        self.assertIn('is_created', state)
        self.assertIn('word_placements', state)
        
        # Create new puzzle and load state
        new_puzzle = CrosswordPuzzle(10, self.test_word_manager)
        success = new_puzzle.load_puzzle_state(state)
        
        self.assertTrue(success)
        self.assertTrue(new_puzzle.is_created)
        self.assertEqual(len(new_puzzle.creator.word_placements), 1)

class TestIntegration(unittest.TestCase):
    """Integration tests for the entire system"""
    
    def test_full_workflow(self):
        """Test complete workflow from creation to solving"""
        # Create puzzle
        test_word_manager = TestWordDataManager()
        puzzle = CrosswordPuzzle(8, test_word_manager)
        
        # Manually create a simple puzzle
        creator = puzzle.get_creator()
        success1 = creator.place_word("HELLO", 3, 1, Direction.ACROSS)
        success2 = creator.place_word("WORLD", 2, 3, Direction.DOWN)
        
        if success1 and success2:
            puzzle.is_created = True
            
            # Start game
            player = puzzle.start_game()
            
            # Play some moves
            player.make_guess(3, 1, 'H')  # Correct
            player.make_guess(3, 2, 'X')  # Incorrect
            
            # Check progress
            progress = player.get_progress()
            self.assertGreater(progress['completion_percentage'], 0)
            self.assertEqual(progress['mistakes'], 1)
            
            # Complete the puzzle
            for placement in player.word_placements:
                for i, letter in enumerate(placement.word):
                    row, col = placement.get_positions()[i]
                    if player.play_grid.get_letter(row, col) == '':
                        player.make_guess(row, col, letter)
            
            self.assertTrue(player.is_complete())

if __name__ == '__main__':
    # Run specific test class if provided as argument
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        if hasattr(sys.modules[__name__], test_class):
            suite = unittest.TestLoader().loadTestsFromTestCase(
                getattr(sys.modules[__name__], test_class))
            unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            print(f"Test class {test_class} not found")
    else:
        # Run all tests
        unittest.main(verbosity=2) 