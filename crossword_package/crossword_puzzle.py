"""
Crossword Puzzle Module

High-level orchestrator that manages the entire crossword puzzle lifecycle.
Coordinates between creation and gameplay phases.
"""

from typing import Optional, List
from .crossword_grid import CrosswordGrid
from .crossword_creator import CrosswordCreator
from .crossword_player import CrosswordPlayer

class CrosswordPuzzle:
    """
    High-level orchestrator that manages the entire crossword puzzle lifecycle.
    Coordinates between creation and gameplay phases.
    """
    
    def __init__(self, size: int = 15, available_words: List[str] = None):
        """
        Initialize a crossword puzzle.
        
        Args:
            size: Size of the crossword grid
            available_words: List of words available for puzzle creation
        """
        self.grid = CrosswordGrid(size)
        self.creator = CrosswordCreator(self.grid, available_words or [])
        self.player: Optional[CrosswordPlayer] = None
        self.is_created = False
    
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
            ],
            'available_words': self.creator.available_words.copy()
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
            # Recreate grid and creator
            self.grid = CrosswordGrid(state['grid_size'])
            self.creator = CrosswordCreator(self.grid, state['available_words'])
            
            # Recreate word placements
            from .word_placement import Direction, WordPlacement
            
            for wp_data in state['word_placements']:
                direction = Direction.ACROSS if wp_data['direction'] == 'across' else Direction.DOWN
                word_placement = WordPlacement(
                    wp_data['word'],
                    wp_data['row'],
                    wp_data['col'],
                    direction,
                    wp_data['clue'],
                    wp_data['number']
                )
                
                # Place the word on the grid
                self.creator.place_word(
                    wp_data['word'],
                    wp_data['row'],
                    wp_data['col'],
                    direction,
                    wp_data['clue']
                )
            
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