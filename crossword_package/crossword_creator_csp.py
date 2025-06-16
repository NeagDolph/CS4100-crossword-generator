import random
from typing import List, Optional, Tuple
from .crossword_grid import CrosswordGrid
from .crossword_validator import CrosswordValidator
from .word_placement import WordPlacement, Direction
from .word_data import WordDataManager

class CrosswordCreator:
    def __init__(self, grid: CrosswordGrid, word_data_manager: Optional[WordDataManager] = None):
        self.grid = grid
        self.word_placements: List[WordPlacement] = []
        self.word_data_manager = word_data_manager

    def place_word(self, word: str, row: int, col: int, direction: Direction, clue: str = "") -> bool:
        if not CrosswordValidator.can_place_word(self.grid, word, row, col, direction):
            return False

        if not clue and self.word_data_manager:
            clue = self.word_data_manager.get_clue_for_word(word) or ""

        placement = WordPlacement(word.upper(), row, col, direction, clue)
        test_placements = self.word_placements + [placement]

        if not CrosswordValidator.validate_intersections(self.grid, test_placements):
            return False

        for i, letter in enumerate(word.upper()):
            if direction == Direction.ACROSS:
                self.grid.set_letter(row, col + i, letter, is_given=True)
            else:
                self.grid.set_letter(row + i, col, letter, is_given=True)

        self.word_placements.append(placement)
        return True

    def place_word_with_auto_clue(self, word: str, row: int, col: int, direction: Direction) -> bool:
        clue = ""
        if self.word_data_manager:
            clue = self.word_data_manager.get_clue_for_word(word) or ""
        return self.place_word(word, row, col, direction, clue)

    def remove_word_by_position(self, row: int, col: int) -> bool:
        for placement in self.word_placements:
            if (row, col) in placement.get_positions():
                return self.remove_word(placement)
        return False

    def remove_word(self, placement: WordPlacement) -> bool:
        if placement not in self.word_placements:
            return False

        positions_to_clear = set(placement.get_positions())
        for other in self.word_placements:
            if other != placement:
                positions_to_clear -= set(other.get_positions())

        for row, col in positions_to_clear:
            self.grid.set_letter(row, col, '')
            self.grid.given_cells.discard((row, col))

        self.word_placements.remove(placement)
        return True

    def clear_puzzle(self):
        self.grid = CrosswordGrid(self.grid.size)
        self.word_placements.clear()

    def validate_puzzle(self) -> bool:
        return (CrosswordValidator.validate_intersections(self.grid, self.word_placements) and
                CrosswordValidator.validate_grid_connectivity(self.word_placements))

    def get_puzzle_statistics(self) -> dict:
        filled_cells = len(self.grid.get_filled_cells())
        total_cells = self.grid.size * self.grid.size
        blocked_cells = len(self.grid.blocked_cells)

        return {
            'word_count': len(self.word_placements),
            'filled_cells': filled_cells,
            'total_cells': total_cells,
            'blocked_cells': blocked_cells,
            'fill_percentage': (filled_cells / (total_cells - blocked_cells) * 100) if total_cells > blocked_cells else 0,
            'intersection_count': CrosswordValidator.count_word_intersections(self.word_placements),
            'is_connected': CrosswordValidator.validate_grid_connectivity(self.word_placements)
        }

    def generate_puzzle_with_csp(self) -> bool:
        self.clear_puzzle()
        slots = self.identify_slots()
        return self._backtrack({}, slots)

    def _backtrack(self, assignment: dict, slots: List[Tuple[int, int, Direction]]) -> bool:
        if len(assignment) == len(slots):
            return True

        # MRV heuristic
        slots = sorted(slots, key=lambda slot: len(self._get_fitting_words(slot)))
        slot = slots[0]
        row, col, direction = slot

        for word in self._get_fitting_words(slot):
            if word in assignment.values():
                continue

            if self.place_word_with_auto_clue(word, row, col, direction):
                assignment[slot] = word
                result = self._backtrack(assignment, [s for s in slots if s != slot])
                if result:
                    return True

                self.remove_word_by_position(row, col)
                del assignment[slot]

        return False

    def identify_slots(self) -> List[Tuple[int, int, Direction]]:
        slots = []
        for row in range(self.grid.size):
            for col in range(self.grid.size):
                if (col == 0 or self.grid.get_letter(row, col - 1) == '') and self.grid.get_letter(row, col) == '':
                    length = self._measure_word_length(row, col, Direction.ACROSS)
                    if length >= 3:
                        slots.append((row, col, Direction.ACROSS))
                if (row == 0 or self.grid.get_letter(row - 1, col) == '') and self.grid.get_letter(row, col) == '':
                    length = self._measure_word_length(row, col, Direction.DOWN)
                    if length >= 3:
                        slots.append((row, col, Direction.DOWN))
        return slots

    def _measure_word_length(self, row: int, col: int, direction: Direction) -> int:
        count = 0
        while row < self.grid.size and col < self.grid.size and self.grid.get_letter(row, col) == '':
            count += 1
            if direction == Direction.ACROSS:
                col += 1
            else:
                row += 1
        return count

    def _get_fitting_words(self, slot: Tuple[int, int, Direction]) -> List[str]:
        row, col, direction = slot
        max_length = self._measure_word_length(row, col, direction)
        return self.word_data_manager.get_words_by_length(max_length, max_length) if self.word_data_manager else []
