#!/usr/bin/env python3
"""
Demo for Constraint Satisfaction Programming (CSP) crossword generator that automatically creates crossword puzzles
with different difficulty levels
"""

import time

from crossword_package.crossword_puzzle import CrosswordPuzzle
from crossword_package.word_placement import Direction, WordPlacement

def demo_generation(difficulty: str, preferred_length: int, target_fill: float, grid_size: int = 15):
    """Demonstrate CSP crossword generation with specified difficulty parameters."""
    print("=" * 60)
    print(f"DEMO: {difficulty.upper()} CSP Crossword Generation")
    print("=" * 60)
    
    # Create a puzzle instance
    puzzle = CrosswordPuzzle(size=grid_size)
    
    print(f"Difficulty: {difficulty}")
    print(f"Target fill: {target_fill}%")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Generating crossword using CSP...")
    
    start_time = time.time()
    
    # Generate puzzle using CSP with difficulty parameters
    success = puzzle.generate_puzzle_csp(
        max_iterations=25000,
        target_fill=target_fill,
        preferred_length=preferred_length
    )
    
    generation_time = time.time() - start_time
    
    if success:
        print(f"\nGenerated {difficulty} puzzle in {generation_time:.2f} seconds")
    else:
        print(f"\nGeneration failed after {generation_time:.2f} seconds")

    print("\nGenerated Crossword Grid:")
    print(puzzle.get_solution_display())
    
    # Show puzzle info
    info = puzzle.get_puzzle_info()
    print(f"\nPuzzle Statistics:")
    print(f"- Words placed: {info['word_count']}")
    print(f"- Grid fill: {info['fill_percentage']:.1f}%")
    print(f"- Intersections: {info['intersection_count']}")
    print(f"- Connected: {info['is_connected']}")
    
    # Show word length distribution
    creator = puzzle.get_creator()
    word_lengths = [len(placement.word) for placement in creator.word_placements]
    avg_length = sum(word_lengths) / len(word_lengths)
    print(f"- Average word length: {avg_length:.1f}")
    print(f"- Word length range: {min(word_lengths)}-{max(word_lengths)}")
    
    # Show all word placements with their locations and directions
    print(f"\nAll Word Placements:")
    for placement in creator.word_placements:
        direction = "ACROSS" if placement.direction == Direction.ACROSS else "DOWN"
        print(f"  {placement.word} ({direction}) at ({placement.row},{placement.col}) - Length: {len(placement.word)}")
    
    return success

def main():
    """Run all CSP demonstration scenarios with different difficulty levels."""
    print("CSP Crossword Generator - Easy, Medium and Hard puzzle generation")
    print("=" * 60)

    # Difficulty configurations
    difficulties = [
        ("EASY", 4, 60.0, 8),     # 4-letter words, 60% fill, 8x8 grid
        ("MEDIUM", 6, 70.0, 11),   # 6-letter words, 70% fill, 11x11 grid
        ("HARD", 8, 80.0, 14)      # 8-letter words, 80% fill, 14x14 grid
    ]
    
    results = []
    
    for difficulty, preferred_length, target_fill, grid_size in difficulties:
        print("\n" + "=" * 80)
        success = demo_generation(difficulty, preferred_length, target_fill, grid_size)
        results.append((difficulty, success))
        time.sleep(1)

    for difficulty, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{difficulty} puzzle generation: {status}")

if __name__ == "__main__":
    main() 