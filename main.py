"""
Simple Crossword Player Demo

Demonstrates the crossword player functionality with manually created puzzles.
"""

from crossword_package import (
    CrosswordGrid, CrosswordCreator, CrosswordValidator, 
    WordPlacement, Direction, WORDS, GRID_SIZE, CrosswordPuzzle, CrosswordPlayer
)

def create_manual_puzzle_demo():
    """Create a simple crossword puzzle manually for demonstration."""
    print("Creating a manual crossword puzzle for demonstration...")
    
    # Create a puzzle
    puzzle = CrosswordPuzzle(10, WORDS)
    creator = puzzle.get_creator()
    
    # Manually place some words
    print("Placing words manually...")
    success1 = creator.place_word("HELLO", 2, 2, Direction.ACROSS, "Greeting")
    success2 = creator.place_word("WORLD", 1, 6, Direction.DOWN, "Earth")  # Intersects at (2,6) with O
    
    if success1 and success2:
        print("Successfully placed words!")
        puzzle.is_created = True
        
        print("Grid with words:")
        print(creator.grid)
        
        # Display word placements
        print("\nWord placements:")
        for i, placement in enumerate(creator.word_placements):
            print(f"{i+1}. {placement.word} at ({placement.row}, {placement.col}) {placement.direction.value}")
        
        return puzzle
    else:
        print("Failed to place words manually")
        return None

def test_gameplay_system(puzzle: CrosswordPuzzle) -> None:
    """Test the gameplay system with a manually created puzzle."""
    print("\n" + "="*50)
    print("Testing gameplay system...")
    
    # Start a game
    player = puzzle.start_game()
    print("Player's view (empty grid):")
    print(player.play_grid)
    
    if puzzle.creator.word_placements:
        first_word = puzzle.creator.word_placements[0]
        print(f"\nMaking a guess for first letter of '{first_word.word}':")
        try:
            correct = player.make_guess(first_word.row, first_word.col, first_word.word[0])
            print(f"Guess was {'correct' if correct else 'incorrect'}")
            print("Updated grid:")
            print(player.play_grid)
            print(f"Progress: {player.get_progress()}")
            
            print(f"\nGetting a hint for second letter...")
            clue = player.get_clue_at_position(first_word.row, first_word.col + 1)
            print(f"Clue for {first_word.word}: {clue}")
            print("Updated grid:")
            print(player.play_grid)
            
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    # Create a manual puzzle for demonstration
    puzzle = create_manual_puzzle_demo()
    
    if puzzle:
        # Test the gameplay system
        test_gameplay_system(puzzle)
        
        print("\n" + "="*50)
        print("Demo completed successfully!")
        print("Note: Automatic puzzle generation has been removed.")
        print("Use manual word placement methods to create puzzles.")
    else:
        print("Failed to create demo puzzle!")
