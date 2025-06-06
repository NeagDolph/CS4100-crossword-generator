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

class CrosswordPuzzle:
    """
    High-level orchestrator that manages the entire crossword puzzle lifecycle.
    Coordinates between creation and gameplay phases.
    """
    
    def __init__(self, size: int = 15, word_data_manager: Optional[WordDataManager] = None, 
                 csv_file_path: str = "nytcrosswords.csv"):
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
            self.word_data_manager = WordDataManager(csv_file_path)
        
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
    
    def search_words(self, pattern: str) -> List[str]:
        """
        Search for words containing a pattern.
        
        Args:
            pattern: Pattern to search for
            
        Returns:
            List of matching words
        """
        return self.word_data_manager.search_words_by_pattern(pattern)
    
    def get_clue_for_word(self, word: str) -> Optional[str]:
        """
        Get a clue for a specific word.
        
        Args:
            word: Word to get clue for
            
        Returns:
            Clue for the word, or None if not found
        """
        return self.word_data_manager.get_clue_for_word(word)
    
    def create_custom_puzzle(self) -> bool:
        """
        Start creating a custom puzzle manually.
        This allows manual word placement through the creator.
        
        Returns:
            True to indicate custom creation mode is enabled
        """
        self.creator.clear_puzzle()
        self.is_created = False
        return True
    
    def finalize_custom_puzzle(self) -> bool:
        """
        Finalize a custom puzzle after manual creation.
        
        Returns:
            True if puzzle is valid and finalized, False otherwise
        """
        if len(self.creator.word_placements) == 0:
            return False
        
        # Validate the custom puzzle
        if self.creator.validate_puzzle():
            self.is_created = True
            return True
        
        return False
    
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
    
    def get_current_grid_display(self) -> str:
        """
        Get the current grid state for display.
        
        Returns:
            String representation of current grid
        """
        if self.player:
            return str(self.player.play_grid)
        else:
            return str(self.creator.grid)
    
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
    
    def reset_game(self):
        """Reset the current game while keeping the puzzle."""
        if self.player:
            self.player.reset_puzzle()
    
    def save_puzzle_state(self) -> dict:
        """
        Save the current puzzle state for later restoration.
        
        Returns:
            Dictionary containing puzzle state
        """
        state = {
            'is_created': self.is_created,
            'grid_size': self.grid.size,
            'csv_file_path': self.word_data_manager.csv_file_path,
            'word_placements': [
                {
                    'word': wp.word,
                    'row': wp.row,
                    'col': wp.col,
                    'direction': wp.direction.value,
                    'clue': wp.clue,
                    'number': wp.number
                }
                for wp in self.creator.word_placements
            ]
        }
        
        if self.player:
            state['game_state'] = {
                'mistakes': self.player.mistakes,
                'hints_used': self.player.hints_used,
                'guess_history': self.player.guess_history.copy()
            }
        
        return state
    
    def load_puzzle_state(self, state: dict) -> bool:
        """
        Load a previously saved puzzle state.
        
        Args:
            state: Dictionary containing puzzle state
            
        Returns:
            True if state was successfully loaded, False otherwise
        """
        try:
            # Recreate grid and word data manager
            self.grid = CrosswordGrid(state['grid_size'])
            csv_path = state.get('csv_file_path', 'nytcrosswords.csv')
            self.word_data_manager = WordDataManager(csv_path)
            self.creator = CrosswordCreator(self.grid, self.word_data_manager)
            
            # Recreate word placements
            from .word_placement import Direction, WordPlacement
            
            for wp_data in state['word_placements']:
                direction = Direction.ACROSS if wp_data['direction'] == 'across' else Direction.DOWN
                
                # Place the word on the grid with its clue
                self.creator.place_word(
                    wp_data['word'],
                    wp_data['row'],
                    wp_data['col'],
                    direction,
                    wp_data['clue']
                )
                
                # Update the number if present
                if 'number' in wp_data and wp_data['number'] > 0:
                    # Find the just-placed word and update its number
                    for placement in self.creator.word_placements:
                        if (placement.word == wp_data['word'] and 
                            placement.row == wp_data['row'] and 
                            placement.col == wp_data['col'] and 
                            placement.direction == direction):
                            placement.number = wp_data['number']
                            break
            
            self.is_created = state['is_created']
            
            # Restore game state if present
            if 'game_state' in state and self.is_created:
                self.player = CrosswordPlayer(self.creator)
                game_state = state['game_state']
                self.player.mistakes = game_state['mistakes']
                self.player.hints_used = game_state['hints_used']
                self.player.guess_history = game_state['guess_history']
            
            return True
        
        except Exception:
            return False
    
    def export_for_print(self) -> dict:
        """
        Export puzzle in a format suitable for printing.
        
        Returns:
            Dictionary with printable puzzle information
            
        Raises:
            ValueError: If no puzzle has been created
        """
        if not self.is_created:
            raise ValueError("No puzzle has been created")
        
        # Create empty grid for puzzle
        empty_grid = CrosswordGrid(self.grid.size)
        
        # Copy blocked cells and structure
        for row, col in self.creator.grid.blocked_cells:
            empty_grid.set_blocked(row, col, True)
        
        # Get clues
        clues = {}
        if self.player:
            clues = self.player.get_crossword_clues()
        else:
            # Generate clues from creator
            clues = {'across': [], 'down': []}
            for i, wp in enumerate(self.creator.word_placements):
                clue_num = wp.number if wp.number > 0 else i + 1
                clue_text = wp.clue if wp.clue else f"Word {clue_num}"
                
                if wp.direction.value == 'across':
                    clues['across'].append((clue_num, clue_text, wp.word))
                else:
                    clues['down'].append((clue_num, clue_text, wp.word))
        
        return {
            'puzzle_grid': str(empty_grid),
            'solution_grid': str(self.creator.grid),
            'clues': clues,
            'statistics': self.creator.get_puzzle_statistics()
        } 