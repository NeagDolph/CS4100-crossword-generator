"""
Crossword Puzzle Module

High-level orchestrator that manages the entire crossword puzzle lifecycle.
Coordinates between creation and gameplay phases.
"""

from typing import Optional, List
from .crossword_grid import CrosswordGrid
from .crossword_creator import CrosswordCreator
from .crossword_player import CrosswordPlayer
from .word_data import WordDataManager
from .word_placement import WordPlacement, Direction

class CrosswordPuzzle:
    """
    High-level orchestrator that manages the entire crossword puzzle lifecycle.
    Coordinates between creation and gameplay phases.
    """
    
    def __init__(self, size: int = 15, word_data_manager: Optional[WordDataManager] = None, 
                 csv_file_path: str = None):
        """
        Initialize a crossword puzzle.
        
        Args:
            size: Size of the crossword grid
            word_data_manager: Optional pre-configured WordDataManager
            csv_file_path: Path to CSV file with word-clue data (used if word_data_manager not provided)
        """
        self.grid = CrosswordGrid(size)
        
        # Initialize word data manager
        if word_data_manager:
            self.word_data_manager = word_data_manager
        else:
            if csv_file_path:
                self.word_data_manager = WordDataManager(csv_file_path)
            else:
                self.word_data_manager = WordDataManager()
        
        self.creator = CrosswordCreator(self.grid, self.word_data_manager)
        self.player: Optional[CrosswordPlayer] = None
        self.is_created = False
    
    def get_word_statistics(self) -> dict:
        """
        Get statistics about available words.
        
        Returns:
            Dictionary with word statistics
        """
        return self.word_data_manager.get_statistics()
    
    def get_random_words(self, count: int = 50, min_length: int = 3, max_length: int = 15) -> List[str]:
        """
        Get random words for manual puzzle creation.
        
        Args:
            count: Number of words to return
            min_length: Minimum word length
            max_length: Maximum word length
            
        Returns:
            List of random words
        """
        return self.word_data_manager.get_random_words(count, min_length, max_length)
    
    def get_clue_for_word(self, word: str) -> Optional[str]:
        """
        Get a clue for a specific word.
        
        Args:
            word: Word to get clue for
            
        Returns:
            Clue for the word, or None if not found
        """
        return self.word_data_manager.get_clue_for_word(word)
    
    def generate_puzzle_csp(self, max_iterations: int = 25000,
                           place_blocked_squares: bool = True,
                           target_fill: float = 90.0,
                           random_seed: Optional[int] = None,
                           preferred_length: int = 6) -> bool:
        """
        Generate a crossword puzzle automatically using advanced CSP techniques, aiming for high fill rate.
        
        Uses sophisticated backtracking and constraint propagation to achieve target fill percentage
        with intelligent state exploration and conflict resolution.
        
        Args:
            max_iterations: Maximum solver iterations before giving up (default: 25000)
            place_blocked_squares: Whether to place strategic black squares (default: True)
            target_fill: Target fill percentage (default: 90.0%)
            random_seed: Random seed for reproducible results (default: None for random)
            preferred_length: Preferred word length for quality and difficulty scoring (default: 6)
            
        Returns:
            True if puzzle generation was successful (reached target or reasonable fill), False otherwise
        """
        success = self.creator.generate_puzzle_csp(
            max_iterations=max_iterations, 
            place_blocked_squares=place_blocked_squares,
            target_fill=target_fill,
            random_seed=random_seed,
            preferred_length=preferred_length
        )
        
        if success:
            self.is_created = True
            
        return success

    def start_game(self) -> CrosswordPlayer:
        """
        Start a gameplay session.
        
        Returns:
            CrosswordPlayer instance for gameplay
            
        Raises:
            ValueError: If no puzzle has been created
        """
        if not self.is_created:
            raise ValueError("Must create puzzle before starting game")
        
        self.player = CrosswordPlayer(self.creator)
        return self.player
    
    def get_creator(self) -> CrosswordCreator:
        """
        Get the creator for manual puzzle building.
        
        Returns:
            CrosswordCreator instance
        """
        return self.creator
    
    def get_player(self) -> Optional[CrosswordPlayer]:
        """
        Get the current player instance.
        
        Returns:
            CrosswordPlayer if game is active, None otherwise
        """
        return self.player
    
    def get_solution_display(self) -> str:
        """
        Get the solution grid for display.
        
        Returns:
            String representation of solution grid
            
        Raises:
            ValueError: If no puzzle has been created
        """
        if not self.is_created:
            raise ValueError("No puzzle has been created")
        
        return str(self.creator.grid)
    
    def get_puzzle_info(self) -> dict:
        """
        Get comprehensive information about the puzzle.
        
        Returns:
            Dictionary with puzzle information
        """
        info = {
            'is_created': self.is_created,
            'grid_size': self.grid.size,
            'has_active_game': self.player is not None
        }
        
        if self.is_created:
            stats = self.creator.get_puzzle_statistics()
            info.update(stats)
            
            if self.player:
                progress = self.player.get_progress()
                info['game_progress'] = progress
                info['is_complete'] = self.player.is_complete()
        
        return info