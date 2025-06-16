from crossword_puzzle import CrosswordPuzzle

def main():
    # Create a new puzzle instance (loads nytcrosswords.csv by default)
    puzzle = CrosswordPuzzle()

    # Generate crossword automatically using CSP
    success = puzzle.get_creator().generate_puzzle_with_csp()

    if success:
        print("Puzzle successfully generated with CSP!")
        print("\nGenerated Grid:\n")
        print(puzzle.get_solution_display())
        print("\nPuzzle Stats:")
        print(puzzle.get_puzzle_info())
    else:
        print("Failed to generate puzzle.")

if __name__ == "__main__":
    main()
