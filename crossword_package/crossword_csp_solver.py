from collections import defaultdict
from word_placement import WordPlacement, Direction
from word_data import get_word_list_by_length
from crossword_grid import CrosswordGrid

class CrosswordCSP:
    def __init__(self, grid: CrosswordGrid, word_list):
        self.grid = grid
        self.variables = self.grid.get_word_slots()  # List of WordPlacement slots
        self.domains = {var: get_word_list_by_length(word_list, var.length()) for var in self.variables}
        self.overlaps = self._find_overlaps()
        self.neighbors = self._build_neighbors()

    def _find_overlaps(self):
        # Map (var1, var2) -> (i, j) where word[i] == other_word[j]
        overlaps = {}
        for i, var1 in enumerate(self.variables):
            for j, var2 in enumerate(self.variables):
                if var1 == var2:
                    continue
                overlap = self.grid.get_overlap(var1, var2)
                if overlap:
                    overlaps[(var1, var2)] = overlap
        return overlaps

    def _build_neighbors(self):
        neighbors = defaultdict(list)
        for (var1, var2), _ in self.overlaps.items():
            neighbors[var1].append(var2)
        return neighbors

    def is_consistent(self, var, value, assignment):
        for neighbor in self.neighbors[var]:
            if neighbor not in assignment:
                continue
            i, j = self.overlaps[(var, neighbor)]
            if value[i] != assignment[neighbor][j]:
                return False
        return True

    def select_unassigned_variable(self, assignment):
        # Minimum Remaining Values heuristic (MRV)
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda var: len(self.domains[var]))

    def order_domain_values(self, var, assignment):
        return self.domains[var]

    def backtrack(self, assignment):
        if len(assignment) == len(self.variables):
            return assignment

        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result:
                    return result
                del assignment[var]

        return None

    def solve(self):
        return self.backtrack({})
