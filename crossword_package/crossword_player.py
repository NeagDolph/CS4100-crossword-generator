"""
Crossword Player Module

Handles crossword puzzle gameplay and solving.
Manages user input, progress tracking, and clue access.
"""

from typing import Dict, List, Set, Tuple, Optional
from .crossword_grid import CrosswordGrid
from .crossword_creator import CrosswordCreator
from .word_placement import WordPlacement

class CrosswordPlayer:
    """
    Handles crossword puzzle gameplay and solving.
    Manages user input, progress tracking, and clue access.
    """
    
    def __init__(self, creator: CrosswordCreator):
        """
        Initialize the crossword player with a completed puzzle.
        
        Args:
            creator: CrosswordCreator instance with a completed puzzle
        """
        # Create a copy of the creator's grid for the solution
        self.solution_grid = creator.grid.copy()
        
        # Create an empty grid for gameplay, preserving only structure (blocked cells)
        self.play_grid = CrosswordGrid(creator.grid.size)
        # Copy blocked cells from the solution
        for row, col in creator.grid.blocked_cells:
            self.play_grid.set_blocked(row, col, True)
        
        self.word_placements = creator.word_placements.copy()
        self.mistakes = 0
        self.start_time = None
        self.end_time = None
        self.guess_history: List[Tuple[int, int, str, bool]] = []  # (row, col, letter, correct)
    
    def make_guess(self, row: int, col: int, letter: str) -> bool:
        """
        Make a guess at a position.
        
        Args:
            row: Row position
            col: Column position
            letter: Letter to guess
            
        Returns:
            True if guess is correct, False otherwise
            
        Raises:
            ValueError: If trying to modify given letters or invalid position
        """
        if not self.play_grid.is_valid_position(row, col):
            raise ValueError(f"Invalid position: ({row}, {col})")
        
        if (row, col) in self.play_grid.given_cells:
            raise ValueError("Cannot modify given letters")
        
        if self.play_grid.is_blocked(row, col):
            raise ValueError("Cannot place letter in blocked cell")
        
        letter = letter.upper()
        self.play_grid.set_letter(row, col, letter, is_given=False)
        
        # Check if guess is correct
        correct_letter = self.solution_grid.get_letter(row, col)
        is_correct = letter == correct_letter
        
        if not is_correct:
            self.mistakes += 1
        
        # Record the guess
        self.guess_history.append((row, col, letter, is_correct))
        
        return is_correct
    
    def remove_guess(self, row: int, col: int):
        """
        Remove a guess at a position.
        
        Args:
            row: Row position
            col: Column position
            
        Raises:
            ValueError: If trying to remove given letters
        """
        if (row, col) in self.play_grid.given_cells:
            raise ValueError("Cannot remove given letters")
        
        self.play_grid.remove_letter(row, col)
    
    def check_word_completion(self, word_placement: WordPlacement) -> bool:
        """
        Check if a word is completely and correctly filled.
        
        Args:
            word_placement: WordPlacement to check
            
        Returns:
            True if word is correctly completed, False otherwise
        """
        for row, col in word_placement.get_positions():
            play_letter = self.play_grid.get_letter(row, col)
            solution_letter = self.solution_grid.get_letter(row, col)
            
            if play_letter != solution_letter or not play_letter:
                return False
        
        return True
    
    def get_completed_words(self) -> List[WordPlacement]:
        """
        Get list of words that are completely and correctly filled.
        
        Returns:
            List of completed WordPlacement objects
        """
        completed = []
        for word_placement in self.word_placements:
            if self.check_word_completion(word_placement):
                completed.append(word_placement)
        return completed
    
    def get_partially_filled_words(self) -> List[WordPlacement]:
        """
        Get list of words that are partially filled.
        
        Returns:
            List of partially filled WordPlacement objects
        """
        partially_filled = []
        for word_placement in self.word_placements:
            filled_count = 0
            total_count = len(word_placement.word)
            
            for row, col in word_placement.get_positions():
                if self.play_grid.get_letter(row, col):
                    filled_count += 1
            
            if 0 < filled_count < total_count:
                partially_filled.append(word_placement)
        
        return partially_filled
    
    def is_complete(self) -> bool:
        """
        Check if the puzzle is completely and correctly solved.
        
        Returns:
            True if puzzle is complete, False otherwise
        """
        # Get all positions that should have letters (from word placements)
        word_positions = set()
        for word_placement in self.word_placements:
            word_positions.update(word_placement.get_positions())
        
        # Check only the positions that should have letters
        for row, col in word_positions:
            play_letter = self.play_grid.get_letter(row, col)
            solution_letter = self.solution_grid.get_letter(row, col)
            
            if play_letter != solution_letter or not play_letter:
                return False
        return True
    
    def get_progress(self) -> Dict[str, float]:
        """
        Get solving progress statistics.
        
        Returns:
            Dictionary with progress information
        """
        total_cells = 0
        filled_cells = 0
        correct_cells = 0
        
        for row in range(self.play_grid.size):
            for col in range(self.play_grid.size):
                if not self.play_grid.is_blocked(row, col):
                    total_cells += 1
                    play_letter = self.play_grid.get_letter(row, col)
                    solution_letter = self.solution_grid.get_letter(row, col)
                    
                    if play_letter:
                        filled_cells += 1
                        if play_letter == solution_letter:
                            correct_cells += 1
        
        completed_words = len(self.get_completed_words())
        total_words = len(self.word_placements)
        
        return {
            'total_cells': total_cells,
            'filled_cells': filled_cells,
            'correct_cells': correct_cells,
            'total_words': total_words,
            'completed_words': completed_words,
            'mistakes': self.mistakes,
            'completion_percentage': (correct_cells / total_cells * 100) if total_cells > 0 else 0,
            'word_completion_percentage': (completed_words / total_words * 100) if total_words > 0 else 0
        }
    
    def get_incorrect_guesses(self) -> List[Tuple[int, int, str]]:
        """
        Get list of incorrect guesses made.
        
        Returns:
            List of (row, col, letter) tuples for incorrect guesses
        """
        incorrect = []
        for row, col, letter, correct in self.guess_history:
            if not correct:
                incorrect.append((row, col, letter))
        return incorrect
    
    def validate_current_state(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Validate the current state and return any errors.
        
        Returns:
            Dictionary with error types and positions
        """
        errors = {
            'incorrect_letters': [],
            'conflicting_letters': []
        }
        
        # Check for incorrect letters
        for row in range(self.play_grid.size):
            for col in range(self.play_grid.size):
                if not self.play_grid.is_blocked(row, col):
                    play_letter = self.play_grid.get_letter(row, col)
                    solution_letter = self.solution_grid.get_letter(row, col)
                    
                    if play_letter and play_letter != solution_letter:
                        errors['incorrect_letters'].append((row, col))
        
        return errors
    
    def get_word_at_position(self, row: int, col: int) -> List[WordPlacement]:
        """
        Get all words that pass through a specific position.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            List of WordPlacement objects that contain this position
        """
        words = []
        for word_placement in self.word_placements:
            if (row, col) in word_placement.get_positions():
                words.append(word_placement)
        return words
    
    def get_clue_at_position(self, row: int, col: int) -> List[Dict[str, any]]:
        """
        Get clues for all words that pass through a specific position.
        
        Args:
            row: Row position
            col: Column position
            
        Returns:
            List of dictionaries with keys: 'direction', 'clue', 'positions'
            for words at this position
        """
        clues = []
        for word_placement in self.get_word_at_position(row, col):
            clue_text = word_placement.clue if word_placement.clue else f"Word {word_placement.number}"
            clues.append({
                "direction": word_placement.direction.value,
                "clue": clue_text,
                "positions": word_placement.get_positions()
            })
        return clues
    
    def get_clue_for_word(self, word_placement: WordPlacement) -> str:
        """
        Get the clue text for a specific word placement.
        
        Args:
            word_placement: WordPlacement to get clue for
            
        Returns:
            Clue text for the word
        """
        return word_placement.clue if word_placement.clue else f"Word {word_placement.number}"
    
    def find_word_by_start_position(self, row: int, col: int, direction: str) -> Optional[WordPlacement]:
        """
        Find a word by its starting position and direction.
        
        Args:
            row: Starting row position
            col: Starting column position
            direction: 'across' or 'down'
            
        Returns:
            WordPlacement if found, None otherwise
        """
        for word_placement in self.word_placements:
            if (word_placement.row == row and 
                word_placement.col == col and 
                word_placement.direction.value == direction):
                return word_placement
        return None
    
    def get_crossword_clues(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Get organized clues for the crossword.
        
        Returns:
            Dictionary with 'across' and 'down' keys containing lists of (number, clue, word) tuples
        """
        clues = {'across': [], 'down': []}
        
        for i, word_placement in enumerate(self.word_placements):
            # Use placement index as clue number if not set
            clue_number = word_placement.number if word_placement.number > 0 else i + 1
            clue_text = word_placement.clue if word_placement.clue else f"Word {clue_number}"
            
            if word_placement.direction.value == 'across':
                clues['across'].append((clue_number, clue_text, word_placement.word))
            else:
                clues['down'].append((clue_number, clue_text, word_placement.word))
        
        # Sort by clue number
        clues['across'].sort(key=lambda x: x[0])
        clues['down'].sort(key=lambda x: x[0])
        
        return clues
    
    def reset_puzzle(self):
        """Reset the puzzle to its initial state."""
        # Create a fresh empty grid for gameplay, preserving only structure (blocked cells)
        self.play_grid = CrosswordGrid(self.solution_grid.size)
        # Copy blocked cells from the solution
        for row, col in self.solution_grid.blocked_cells:
            self.play_grid.set_blocked(row, col, True)
        
        self.mistakes = 0
        self.guess_history.clear()
        self.start_time = None
        self.end_time = None 