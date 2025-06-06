"""
Enhanced Crossword Player Demo

Demonstrates the crossword player functionality with CSV-loaded word-clue pairs.
"""

from crossword_package import (
    CrosswordGrid, CrosswordCreator, CrosswordValidator, 
    WordPlacement, Direction, GRID_SIZE, CrosswordPuzzle, CrosswordPlayer,
    WordDataManager, word_data_manager
)

def show_word_data_stats():
    """Display statistics about the loaded word data."""
    print("Loading word-clue data from CSV...")
    stats = word_data_manager.get_statistics()
    
    print(f"Word Data Statistics:")
    print(f"  Total entries: {stats['total_entries']:,}")
    print(f"  Unique words: {stats['unique_words']:,}")
    print(f"  Word length range: {stats['min_word_length']}-{stats['max_word_length']}")
    print(f"  Average word length: {stats['avg_word_length']:.1f}")
    print()

def create_csv_based_puzzle_demo():
    """Create a crossword puzzle using words and clues from a CSV file."""
    print("Creating a demo crossword puzzle with loaded words and clues...")
    
    # Create a puzzle with CSV data
    puzzle = CrosswordPuzzle(12)  # 12x12 grid
    creator = puzzle.get_creator()
    
    # Get some random words of different lengths
    print("Selecting words from CSV database...")
    short_words = puzzle.get_random_words(count=20, min_length=3, max_length=5)
    medium_words = puzzle.get_random_words(count=15, min_length=6, max_length=8)
    long_words = puzzle.get_random_words(count=10, min_length=9, max_length=12)
    
    print(f"Available short words (3-5 letters): {short_words[:5]}...")
    print(f"Available medium words (6-8 letters): {medium_words[:5]}...")
    print(f"Available long words (9-12 letters): {long_words[:3]}...")
    print()
    
    # Try to place a few words manually with their clues
    placed_words = []
    
    # Try a long word first
    if long_words:
        word = long_words[0]
        clue = puzzle.get_clue_for_word(word)
        success = creator.place_word_with_auto_clue(word, 6, 2, Direction.ACROSS)
        if success:
            placed_words.append((word, clue))
            print(f"Placed '{word}' ACROSS at (6,2)")
            print(f"  Clue: {clue}")
    
    # Try to place an intersecting word
    if medium_words and placed_words:
        for word in medium_words:
            # Try to place it intersecting with the first word
            success = creator.place_word_with_auto_clue(word, 3, 5, Direction.DOWN)
            if success:
                clue = puzzle.get_clue_for_word(word)
                placed_words.append((word, clue))
                print(f"Placed '{word}' DOWN at (3,5)")
                print(f"  Clue: {clue}")
                break
    
    # Try to place another intersecting word
    if short_words and len(placed_words) >= 2:
        for word in short_words:
            success = creator.place_word_with_auto_clue(word, 7, 6, Direction.DOWN)
            if success:
                clue = puzzle.get_clue_for_word(word)
                placed_words.append((word, clue))
                print(f"Placed '{word}' DOWN at (7,6)")
                print(f"  Clue: {clue}")
                break
    
    if placed_words:
        puzzle.is_created = True
        print(f"\nSuccessfully created puzzle with {len(placed_words)} words!")
        
        print("\nPuzzle grid:")
        print(creator.grid)
        
        print("\nWord placements with clues:")
        for i, placement in enumerate(creator.word_placements):
            print(f"{i+1}. {placement.word} at ({placement.row}, {placement.col}) {placement.direction.value}")
            print(f"   Clue: {placement.clue}")
        
        return puzzle
    else:
        print("Failed to place any words")
        return None

def demonstrate_word_search():
    """Demonstrate word search functionality."""
    print("\n" + "="*50)
    print("Demonstrating word search functionality...")
    
    # Search for words containing specific patterns
    patterns = ['CAT', 'HOUSE', 'LOVE']
    
    for pattern in patterns:
        matches = word_data_manager.search_words_by_pattern(pattern)
        print(f"\nWords containing '{pattern}': {matches[:10]}{'...' if len(matches) > 10 else ''}")
        if matches:
            # Show clues for first few matches
            for word in matches[:3]:
                clue = word_data_manager.get_clue_for_word(word)
                print(f"  {word}: {clue}")

def test_csv_gameplay_system(puzzle: CrosswordPuzzle) -> None:
    """Test the gameplay system with a CSV-created puzzle."""
    print("\n" + "="*50)
    print("Testing gameplay system with CSV-loaded clues...")
    
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
            
            print(f"\nGetting clues at position ({first_word.row}, {first_word.col}):")
            clues = player.get_clue_at_position(first_word.row, first_word.col)
            print(f"Clues: {clues}")
            
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    print("Crossword Demo with Loaded Word-Clue Data")
    print("=" * 50)
    
    # Show word data statistics
    show_word_data_stats()
    
    # Demonstrate word search
    demonstrate_word_search()
    
    # Create a CSV-based puzzle
    puzzle = create_csv_based_puzzle_demo()
    
    if puzzle:
        # Test the gameplay system
        test_csv_gameplay_system(puzzle)
        
        print("\n" + "="*50)
        print("Demo completed successfully!")
        print(f"Total words available in database: {word_data_manager.get_statistics()['unique_words']:,}")
    else:
        print("Failed to create demo puzzle!")
