
from crossword_grid_2 import CrosswordGrid
from crossword_csp_solver import CrosswordCSP
from word_data_2 import load_words

def print_solution(grid, assignment):
    for slot, word in assignment.items():
        grid.place_word(slot, word)
    grid.display()

def main():
    # Create a basic 5x5 grid with pre-defined black squares
    grid_data = [
        ["", "", "", "", ""],
        ["", "#", "", "#", ""],
        ["", "", "", "", ""],
        ["", "#", "", "#", ""],
        ["", "", "", "", ""]
    ]
    grid = CrosswordGrid(grid_data)

    # Load word list
    word_list = load_words("/Users/kelseynihezagirwe/Desktop/CS4100-crossword-generator/nytcrosswords.csv")
    print(f"Loaded {len(word_list)} words")


    for slot in grid.get_word_slots():
        print(f"{slot.direction.name} slot at ({slot.row}, {slot.col}) length={slot.length}")

    # Solve the puzzle
    csp = CrosswordCSP(grid, word_list)
    solution = csp.solve()

    if solution:
        print("\nSolved crossword:")
        print_solution(grid, solution)
    else:
        print("\nNo solution found.")

if __name__ == "__main__":
    main()
