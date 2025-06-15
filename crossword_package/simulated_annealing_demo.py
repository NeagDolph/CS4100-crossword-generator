"""
Example usage of the Simulated Annealing Crossword Generator

This example demonstrates how to use the SA solver to generate crossword puzzles
from scratch, similar to the CSP solver but using temperature-based optimization.
"""

from crossword_puzzle import CrosswordPuzzle
from word_data import WordDataManager
from simulated_annealing_generator import SimulatedAnnealingSolver
import random

def generate_crossword_with_sa(grid_size=15, max_iterations=10000, target_fill=75.0, 
                              random_seed=None, csv_file="nytcrosswords.csv"):
    """
    Generate a crossword puzzle using simulated annealing.
    
    Args:
        grid_size: Size of the crossword grid (default 15x15)
        max_iterations: Maximum number of SA iterations
        target_fill: Target fill percentage (default 75%)
        random_seed: Random seed for reproducible results
        csv_file: Path to word-clue CSV file
        
    Returns:
        CrosswordPuzzle instance with generated crossword
    """
    print(f"Generating {grid_size}x{grid_size} crossword using Simulated Annealing")
    print(f"Target fill: {target_fill}%, Max iterations: {max_iterations}")
    
    # Initialize puzzle and word data
    try:
        word_data_manager = WordDataManager(csv_file)
        puzzle = CrosswordPuzzle(grid_size, word_data_manager)
    except Exception as e:
        print(f"Error loading word data: {e}")
        print("Using fallback word data...")
        # Fallback: create a simple word list for testing
        word_data_manager = create_fallback_word_data()
        puzzle = CrosswordPuzzle(grid_size, word_data_manager)
    
    # Get creator and initialize SA solver
    creator = puzzle.get_creator()
    sa_solver = SimulatedAnnealingSolver(word_data_manager, preferred_length=6)
    
    # Configure SA parameters
    sa_solver.initial_temperature = 100.0
    sa_solver.final_temperature = 0.01
    sa_solver.cooling_rate = 0.995
    
    # Adjust perturbation weights for better generation
    sa_solver.perturbation_weights = {
        sa_solver.PerturbationType.ADD_WORD: 0.6,      # Prefer adding words
        sa_solver.PerturbationType.REMOVE_WORD: 0.15,  # Sometimes remove words
        sa_solver.PerturbationType.SWAP_WORD: 0.15,    # Sometimes swap words
        sa_solver.PerturbationType.RELOCATE_WORD: 0.1  # Occasionally relocate
    }
    
    # Run the solver
    success = sa_solver.solve(
        creator=creator,
        max_iterations=max_iterations,
        target_fill=target_fill,
        place_blocked_squares=True,
        random_seed=random_seed
    )
    
    if success:
        puzzle.finalize_custom_puzzle()
        print("\n" + "="*50)
        print("CROSSWORD GENERATION SUCCESSFUL!")
        print("="*50)
        
        # Print statistics
        stats = creator.get_puzzle_statistics()
        sa_stats = sa_solver.get_statistics()
        
        print(f"Final Statistics:")
        print(f"  Fill Percentage: {stats['fill_percentage']:.1f}%")
        print(f"  Words Placed: {stats['word_count']}")
        print(f"  Intersections: {stats['intersection_count']}")
        print(f"  Connected: {stats['is_connected']}")
        print(f"  Blocked Cells: {stats['blocked_cells']}")
        
        print(f"\nSA Solver Statistics:")
        print(f"  Accepted Moves: {sa_stats['accepted_moves']}")
        print(f"  Rejected Moves: {sa_stats['rejected_moves']}")
        print(f"  Acceptance Rate: {sa_stats['acceptance_rate']:.1f}%")
        print(f"  Final Fitness: {sa_stats['current_fitness']:.1f}")
        print(f"  Best Fitness: {sa_stats['best_fitness']:.1f}")
        print(f"  Final Temperature: {sa_stats['final_temperature']:.6f}")
        
        # Display the crossword
        print(f"\nGenerated Crossword:")
        print(puzzle.get_current_grid_display())
        
        # Show word list
        print(f"\nWords in Puzzle:")
        for i, wp in enumerate(creator.word_placements, 1):
            direction = "Across" if wp.direction.value == "across" else "Down"
            clue = wp.clue if wp.clue else f"Clue for {wp.word}"
            print(f"  {i}. {wp.word} ({direction}) - {clue}")
            
    else:
        print("\nCrossword generation was not fully successful.")
        print("You may want to try:")
        print("- Increasing max_iterations")
        print("- Reducing target_fill percentage")
        print("- Using a different random seed")
        print("- Adjusting SA parameters")
        
        # Still show what was generated
        stats = creator.get_puzzle_statistics()
        if stats['word_count'] > 0:
            print(f"\nPartial result achieved:")
            print(f"  Fill: {stats['fill_percentage']:.1f}%")
            print(f"  Words: {stats['word_count']}")
            print(puzzle.get_current_grid_display())
    
    return puzzle

def create_fallback_word_data():
    """Create a simple word data manager for testing when CSV is not available."""
    from word_data import WordDataManager, WordClue
    
    # Create a basic word list for testing
    test_words = [
        ("CAT", "Feline pet"),
        ("DOG", "Canine companion"),
        ("BIRD", "Flying animal"),
        ("FISH", "Swimming creature"),
        ("TREE", "Woody plant"),
        ("HOUSE", "Dwelling place"),
        ("WATER", "H2O"),
        ("LIGHT", "Illumination"),
        ("MUSIC", "Sound art"),
        ("BOOK", "Reading material"),
        ("COMPUTER", "Electronic device"),
        ("PHONE", "Communication device"),
        ("GARDEN", "Outdoor space"),
        ("KITCHEN", "Cooking room"),
        ("BEDROOM", "Sleeping room"),
        ("WINDOW", "Glass opening"),
        ("DOOR", "Entry way"),
        ("CHAIR", "Seating furniture"),
        ("TABLE", "Flat surface"),
        ("FLOWER", "Blooming plant"),
        ("BREAD", "Baked food"),
        ("CHEESE", "Dairy product"),
        ("APPLE", "Red fruit"),
        ("ORANGE", "Citrus fruit"),
        ("BANANA", "Yellow fruit"),
        ("PIZZA", "Italian dish"),
        ("PASTA", "Italian noodles"),
        ("SALAD", "Green dish"),
        ("COFFEE", "Morning drink"),
        ("TEA", "Hot beverage"),
        ("JUICE", "Fruit drink"),
        ("MILK", "Dairy drink"),
        ("BREAD", "Staple food"),
        ("BUTTER", "Dairy spread"),
        ("SUGAR", "Sweet substance"),
        ("SALT", "Seasoning"),
        ("PEPPER", "Spice"),
        ("ONION", "Vegetable"),
        ("GARLIC", "Aromatic bulb"),
        ("TOMATO", "Red vegetable"),
        ("CARROT", "Orange vegetable"),
        ("POTATO", "Starchy tuber"),
        ("LETTUCE", "Leafy green"),
        ("SPINACH", "Dark green"),
        ("BROCCOLI", "Green vegetable"),
        ("CHICKEN", "Poultry"),
        ("BEEF", "Red meat"),
        ("PORK", "Pig meat"),
        ("FISH", "Seafood"),
        ("SHRIMP", "Shellfish"),
        ("LOBSTER", "Crustacean")
    ]
    
    class MockWordDataManager:
        def __init__(self):
            self.word_clues = [WordClue(word, clue) for word, clue in test_words]
            self.word_to_clues = {}
            for wc in self.word_clues:
                if wc.word not in self.word_to_clues:
                    self.word_to_clues[wc.word] = []
                self.word_to_clues[wc.word].append(wc)
            self._loaded = True
            self.csv_file_path = "fallback_data"
        
        def ensure_loaded(self):
            pass
        
        def get_all_words(self):
            return list(self.word_to_clues.keys())
        
        def get_words_by_length(self, min_length=3, max_length=15):
            return [word for word in self.word_to_clues.keys() 
                   if min_length <= len(word) <= max_length]
        
        def get_random_words(self, count=50, min_length=3, max_length=15):
            available = self.get_words_by_length(min_length, max_length)
            return random.sample(available, min(count, len(available)))
        
        def get_clue_for_word(self, word):
            word = word.upper()
            if word in self.word_to_clues:
                return self.word_to_clues[word][0].clue
            return None
        
        def get_statistics(self):
            words = list(self.word_to_clues.keys())
            return {
                'total_entries': len(self.word_clues),
                'unique_words': len(words),
                'min_word_length': min(len(w) for w in words),
                'max_word_length': max(len(w) for w in words),
                'avg_word_length': sum(len(w) for w in words) / len(words)
            }
    
    return MockWordDataManager()

def run_sa_experiments():
    """Run multiple SA experiments with different parameters."""
    print("Running Simulated Annealing Crossword Generation Experiments")
    print("="*60)
    
    # Experiment 1: Small grid, high fill target
    print("\nExperiment 1: 9x9 grid, 80% fill target")
    puzzle1 = generate_crossword_with_sa(
        grid_size=9,
        max_iterations=5000,
        target_fill=80.0,
        random_seed=42
    )
    
    # Experiment 2: Standard grid, moderate fill target
    print("\nExperiment 2: 15x15 grid, 70% fill target")
    puzzle2 = generate_crossword_with_sa(
        grid_size=15,
        max_iterations=8000,
        target_fill=70.0,
        random_seed=123
    )
    
    # Experiment 3: Large grid, lower fill target
    print("\nExperiment 3: 19x19 grid, 60% fill target")
    puzzle3 = generate_crossword_with_sa(
        grid_size=19,
        max_iterations=12000,
        target_fill=60.0,
        random_seed=456
    )
    
    return [puzzle1, puzzle2, puzzle3]

def compare_sa_parameters():
    """Compare different SA parameter settings."""
    print("Comparing Simulated Annealing Parameter Settings")
    print("="*50)
    
    base_params = {
        'grid_size': 11,
        'max_iterations': 6000,
        'target_fill': 75.0,
        'random_seed': 789
    }
    
    # Test different cooling schedules
    cooling_schedules = ['exponential', 'linear', 'logarithmic', 'adaptive']
    
    for schedule in cooling_schedules:
        print(f"\nTesting {schedule} cooling schedule:")
        
        # Create puzzle
        word_data_manager = create_fallback_word_data()
        puzzle = CrosswordPuzzle(base_params['grid_size'], word_data_manager)
        creator = puzzle.get_creator()
        
        # Create SA solver with specific cooling schedule
        sa_solver = SimulatedAnnealingSolver(word_data_manager)
        sa_solver.cooling_schedule = getattr(sa_solver.CoolingSchedule, schedule.upper())
        
        # Run solver
        success = sa_solver.solve(
            creator=creator,
            max_iterations=base_params['max_iterations'],
            target_fill=base_params['target_fill'],
            random_seed=base_params['random_seed']
        )
        
        # Report results
        stats = creator.get_puzzle_statistics()
        sa_stats = sa_solver.get_statistics()
        
        print(f"  Fill: {stats['fill_percentage']:.1f}%")
        print(f"  Words: {stats['word_count']}")
        print(f"  Acceptance Rate: {sa_stats['acceptance_rate']:.1f}%")
        print(f"  Final Fitness: {sa_stats['current_fitness']:.1f}")

if __name__ == "__main__":
    print("Simulated Annealing Crossword Generator")
    print("="*40)
    
    # Simple single generation example
    print("\nGenerating a single crossword puzzle:")
    puzzle = generate_crossword_with_sa(
        grid_size=13,
        max_iterations=7000,
        target_fill=75.0,
        random_seed=2024
    )
    
    # Uncomment to run experiments
    # run_sa_experiments()
    
    # Uncomment to compare parameters
    # compare_sa_parameters()
    
    print("\nGeneration complete!")