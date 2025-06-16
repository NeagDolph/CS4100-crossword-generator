from word_placement import WordPlacement, Direction

class CrosswordGrid:
    def __init__(self, grid_data):
        self.grid = grid_data
        self.rows = len(grid_data)
        self.cols = len(grid_data[0]) if self.rows > 0 else 0

    def get_word_slots(self):
        slots = []
        # Horizontal slots
        for r in range(self.rows):
            c = 0
            while c < self.cols:
                if self.grid[r][c] != "#":
                    start = c
                    while c < self.cols and self.grid[r][c] != "#":
                        c += 1
                    if c - start > 1:
                        slots.append(WordPlacement("", r, start, Direction.ACROSS, length=c - start))

                else:
                    c += 1

        # Vertical slots
        for c in range(self.cols):
            r = 0
            while r < self.rows:
                if self.grid[r][c] != "#":
                    start = r
                    while r < self.rows and self.grid[r][c] != "#":
                        r += 1
                    if r - start > 1:
                        slots.append(WordPlacement("", start, c, Direction.DOWN, length=r - start))

                else:
                    r += 1

        return slots

    def get_overlap(self, wp1, wp2):
        # Find intersection coordinates and index in each word
        if wp1.direction == wp2.direction:
            return None

        if wp1.direction == Direction.ACROSS:
            across, down = wp1, wp2
        else:
            across, down = wp2, wp1

        if (down.col >= across.col and
            down.col < across.col + len(across.word) and
            across.row >= down.row and
            across.row < down.row + len(down.word)):

            i = down.col - across.col
            j = across.row - down.row
            return (i, j)
        return None

    def place_word(self, slot, word):
        r, c = slot.row, slot.col
        for i, ch in enumerate(word):
            if slot.direction == Direction.ACROSS:
                self.grid[r][c + i] = ch
            else:
                self.grid[r + i][c] = ch

    def display(self):
        for row in self.grid:
            print(" ".join(ch if ch else "_" for ch in row))
