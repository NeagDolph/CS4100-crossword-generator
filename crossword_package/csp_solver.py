"""
CSP Solver Module for Crossword Generation

Implements constraint satisfaction programming techniques for automatic crossword generation.
Based on academic research in crossword compilation using CSP methods.
Enhanced with multi-objective heuristics for optimal word placement.
"""

import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from .crossword_creator import CrosswordCreator
from .crossword_grid import CrosswordGrid
from .word_placement import WordPlacement, Direction
from .crossword_validator import CrosswordValidator
from .word_data import WordDataManager
from .util import exponential_random_choice, frequency_score

@dataclass  
class Slot(WordPlacement):
    """
    Represents a slot (sequence of empty cells) for word placement.
    Inherits from WordPlacement and adds constraint information.
    
    Attributes:
        constraints: Dictionary mapping position to required letter
        length: Length of the slot (derived from word length)
    """
    constraints: Dict[int, str] = field(default_factory=dict)
    
    def __init__(self, row: int, col: int, direction: Direction, length: int, constraints: Dict[int, str] = None):
        # Create a placeholder word of the right length for WordPlacement
        placeholder_word = '?' * length
        super().__init__(placeholder_word, row, col, direction)
        self.constraints = constraints or {}
        self.length = length
    
    def calculate_difficulty(self, preferred_length: int = 6) -> float:
        """Calculate difficulty score based on constraints, length, and preferred length"""
        base_difficulty = len(self.constraints) * 3.5 # More constraints = harder

        # Calculate difficulty based on letter frequency
        letter_freq = frequency_score(''.join(str(letter) for letter in self.constraints.values()))
        
        # Calculate average letter frequency for constrained positions
        if self.constraints:
            avg_freq = letter_freq / len(self.constraints)
            base_difficulty += 0.5 * avg_freq

        # Length penalty based on distance from preferred length
        length_penalty = abs(self.length - preferred_length) * 0.5

        return 0 if base_difficulty == 0 else base_difficulty + length_penalty
    
    def get_constraint_pattern(self) -> str:
        """Get constraint pattern as string with '?' for unknown positions."""
        pattern = ['?'] * self.length
        for pos, letter in self.constraints.items():
            if 0 <= pos < self.length:
                pattern[pos] = str(letter)  # Convert numpy strings to regular strings
        return ''.join(pattern)
    
    def matches_word(self, word: str) -> bool:
        """Check if a word matches this slot's constraints.
        Words must be long enough to satisfy at least one constraint."""
        if len(word) > self.length:
            return False  # Word can't be longer than slot
        
        # If slot has constraints (represents intersections), word must reach at least one constraint
        if self.constraints:
            min_constraint_pos = min(self.constraints.keys())
            if len(word) <= min_constraint_pos:
                return False  # Word must be long enough to a single constraint
        
        # Check all constraints that the word should satisfy
        for pos, required_letter in self.constraints.items():
            if pos < len(word):  # Only check constraint if position exists in word
                if word[pos].upper() != str(required_letter).upper():
                    return False
        return True
    
    def __str__(self) -> str:
        return f"Slot(row={self.row}, col={self.col}, direction={self.direction.value}, length={self.length})"
    
    def __hash__(self):
        return hash((self.row, self.col, self.direction, self.length, tuple(sorted(self.constraints.items()))))
    
    def __eq__(self, other):
        if not isinstance(other, Slot):
            return False
        return self.row == other.row and self.col == other.col and self.direction == other.direction and self.length == other.length and self.constraints == other.constraints

class Heuristics:
    """
    Heuristic system for crossword CSP solving.
    Balances intersections, fill percentage, and spatial distribution.
    """
    
    def __init__(self):
        self.weights = {
            'intersection_potential': 2,
            'fill_efficiency': 0.25,
            'feasibility': 0.15,
            'constraint_satisfaction': 0.2,
            'length': 1.1
        }
    
    def evaluate_slot(self, slot: Slot, compatible_words: List[str], 
                     current_placements: List[WordPlacement], preferred_length: int = 6) -> float:
        """
        Evaluate a slot using multi-objective heuristic.
        
        Args:
            slot: Slot to evaluate
            compatible_words: List of words that can fit in this slot
            current_placements: Current word placements
            all_slots: All available slots
            grid: Current grid state
            
        Returns:
            Heuristic score (higher is better)
        """
        if not compatible_words or len(compatible_words) == 0:
            return 0.0
        
        # Intersection Potential Score
        intersection_score = self._calculate_intersection_potential(
            slot, compatible_words, current_placements)
        
        #Feasibility Score
        feasibility_score = min(10.0, len(compatible_words) / 5.0)
        
        # Constraint Satisfaction Score
        constraint_score = len(slot.constraints) * 2.0
        
        # Prefer variation in word lengths
        length_deviations = [abs(preferred_length - len(word)) for word in compatible_words]
        word_length_deviation_bonus = 2.0 / (1.0 + np.std(length_deviations))

        # Prefer average word length be close to preferred length
        average_word_length = sum(len(word) for word in compatible_words) / len(compatible_words)
        average_word_length_bonus = 15.0 / (1.0 + abs(preferred_length - average_word_length))
        
        # Weighted combination
        total_score = (
            self.weights['intersection_potential'] * intersection_score +
            self.weights['feasibility'] * feasibility_score +
            self.weights['constraint_satisfaction'] * constraint_score +
            self.weights['length'] * (word_length_deviation_bonus + average_word_length_bonus)
        )
        
        return total_score
    
    def _calculate_intersection_potential(self, slot: Slot, 
                                        compatible_words: List[str],
                                        current_placements: List[WordPlacement]) -> float:
        """Calculate potential for creating beneficial intersections."""
        if not current_placements:
            return 5.0  # Base score for first words
        
        intersection_potential = 0.0
        slot_positions = set(slot.get_positions())
        intersection_count = 0
        
        for placement in current_placements:
            placement_positions = set(placement.get_positions())
            intersections = slot_positions & placement_positions
            
            if intersections:
                intersection_count += len(intersections)
                # Score based on number of potential intersections
                intersection_potential += len(intersections) * 3.0
                
                # Bonus for intersections that create letter constraints
                for pos in intersections:
                    letter = placement.get_letter_at_position(pos[0], pos[1])
                    if letter:
                        # Check how many compatible words have this letter at intersection
                        matching_words = 0
                        for word in compatible_words[:10]:
                            if slot.direction == Direction.ACROSS:
                                word_pos = pos[1] - slot.col
                            else:
                                word_pos = pos[0] - slot.row
                            
                            if 0 <= word_pos < len(word) and word[word_pos] == letter:
                                matching_words += 1
                        
                        # Higher score if more words can satisfy this constraint
                        intersection_potential += (matching_words / max(1, len(compatible_words))) * 10.0
        
        return min(20.0, intersection_potential)


class FastWordIndex:
    """
    Fast word index for constraint-based word lookup.
    Uses length-based indexing with quality scoring for efficient word selection.
    """
    
    def __init__(self, word_data_manager: WordDataManager, preferred_length: int = 6):
        self.word_data_manager = word_data_manager
        self.word_data_manager.ensure_loaded()
        self.preferred_length = preferred_length
        
        # Length-based index for quick filtering
        self.length_index: Dict[int, List[str]] = defaultdict(list)
        
        # Word quality index for better selection
        self.word_quality: Dict[str, float] = {}
        
        # Position-letter index for efficient constraint matching
        # Maps (position, letter) -> set of words that have that letter at that position
        self.position_letter_index: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
        
        self._build_indexes()
    
    def _build_indexes(self):
        """Build all word indexes for fast lookup."""
        all_words = self.word_data_manager.get_all_words()
        
        for word in all_words:
            word_len = len(word)
            self.length_index[word_len].append(word)
            
            # Calculate word quality score
            self.word_quality[word] = self._calculate_word_quality(word, self.preferred_length)
            
            # Build position-letter index
            word_upper = word.upper()
            for pos, letter in enumerate(word_upper):
                self.position_letter_index[(pos, letter)].add(word)
    
    def _calculate_word_quality(self, word: str, preferred_length: int = 6) -> float:
        """Calculate quality score for word selection. Higher score = better match to preferred length and more common letters."""
        base_quality = frequency_score(word)
        length_penalty = 2 * abs(len(word) - preferred_length)  # Prefer words closer to preferred length
        return base_quality - length_penalty
    
    def find_compatible_words(self, slot: Slot, max_results: int = 100, sort_by_quality: bool = True) -> List[str]:
        """
        Find words compatible with the given slot constraints.
        Words must be long enough to satisfy at least one intersection constraint.
        
        Args:
            slot: Slot to find words for
            max_results: Maximum number of results to return
            sort_by_quality: Whether to sort by word quality
            
        Returns:
            List of compatible words
        """
        min_word_length = 3
        
        # If no constraints, just return words by length
        if not slot.constraints:
            candidates = []
            for length in range(min_word_length, slot.length + 1):
                candidates.extend(self.length_index.get(length, []))
            
            if sort_by_quality:
                candidates.sort(key=lambda w: self._calculate_word_quality(w, self.preferred_length), reverse=True)
            return candidates[:max_results]
        
        # Use constraint-based filtering for efficiency
        # Sort constraints by position (lowest to highest)
        sorted_constraints = sorted(slot.constraints.items(), key=lambda x: x[0])
        
        # Initialize compatible words set with words that satisfy the first constraint
        compatible_words = None
        min_required_length = max(pos + 1 for pos, _ in sorted_constraints)  # Minimum length to satisfy all constraints
        
        for pos, required_letter in sorted_constraints:
            # Get all words that have the required letter at this position
            words_with_letter = self.position_letter_index.get((pos, str(required_letter).upper()), set())
            
            # Filter by length requirements:
            # - Word must be at least long enough to reach this constraint position
            # - Word must not be longer than the slot
            valid_length_words = {w for w in words_with_letter 
                                if min_word_length <= len(w) <= slot.length and len(w) > pos}
            
            if compatible_words is None:
                # First constraint - initialize the set
                compatible_words = valid_length_words
            else:
                # Subsequent constraints - intersect with previous results
                compatible_words = compatible_words.intersection(valid_length_words)
            
            # Early exit if no words satisfy constraints so far
            if not compatible_words:
                return []
        
        # Convert to list for sorting and slicing
        compatible = list(compatible_words) if compatible_words else []
        
        if sort_by_quality:
            # Sort by quality first, then by length
            compatible.sort(key=lambda w: (self._calculate_word_quality(w, self.preferred_length), len(w)), reverse=True)
        else:
            # Just sort by length (longer words preferred)
            compatible.sort(key=len, reverse=True)
        
        return compatible[:max_results]

def _find_slots_in_direction(grid: CrosswordGrid, direction: Direction, min_length: int = 3) -> List[Slot]:
    """
    Helper function to find slots in a specific direction.
    Explicitly splits slots between black squares
    """
    slots: List[Slot] = []
    is_horizontal = direction == Direction.ACROSS
    
    for outer in range(grid.size):
        # Find all black square positions in this row/column
        black_positions = []
        for inner in range(grid.size):
            row, col = (outer, inner) if is_horizontal else (inner, outer)
            if grid.is_blocked(row, col):
                black_positions.append(inner)
        
        # Create segments between black squares (including start and end)
        segment_starts = [0] + [pos + 1 for pos in black_positions]
        segment_ends = black_positions + [grid.size]
        
        # Process each segment to create slots
        for start, end in zip(segment_starts, segment_ends):
            segment_length = end - start
            
            # Skip segments that are too short
            if segment_length < min_length:
                continue
            
            # Build constraints for this segment
            constraints = {}
            for pos in range(start, end):
                row, col = (outer, pos) if is_horizontal else (pos, outer)
                letter = grid.get_letter(row, col)
                if letter:
                    constraints[pos - start] = str(letter)
            
            # Create slot for this segment
            start_row, start_col = (outer, start) if is_horizontal else (start, outer)
            slot = Slot(start_row, start_col, direction, segment_length, constraints)
            slots.append(slot)
    
    return slots

def find_empty_slots(grid: CrosswordGrid, min_length: int = 3, preferred_length: int = 6) -> List[Slot]:
    """
    Find all slots in the grid that can accommodate new words.
    
    Args:
        grid: Crossword grid to analyze
        min_length: Minimum slot length to consider
        
    Returns:
        List of slots that can accommodate words, sorted by difficulty
    """
    slots: List[Slot] = []

    # Find horizontal and vertical slots
    slots.extend(_find_slots_in_direction(grid, Direction.ACROSS, min_length))
    slots.extend(_find_slots_in_direction(grid, Direction.DOWN, min_length))
        
    # Sort by difficulty (harder/longer slots first, then easier ones)
    slots.sort(key=lambda s: s.calculate_difficulty(preferred_length=preferred_length), reverse=True)
    
    return slots

def find_intersecting_slots(grid: CrosswordGrid, existing_placements: List[WordPlacement], 
                           min_length: int = 3) -> List[Slot]:
    """
    Find slots that would create intersections with existing words.
    
    Args:
        grid: Crossword grid to analyze
        existing_placements: Current word placements
        min_length: Minimum slot length to consider
        
    Returns:
        List of slots that would create intersections with existing words
    """
    if not existing_placements:
        # If no words are placed, return all slots (for first word)
        return find_empty_slots(grid, min_length)
    
    all_slots = find_empty_slots(grid, min_length)
    intersecting_slots = []
    
    # Get all positions occupied by existing words
    existing_positions = set()
    for placement in existing_placements:
        existing_positions.update(placement.get_positions())
    
    # Check each slot to see if it would intersect with existing words
    for slot in all_slots:
        slot_positions = set(slot.get_positions())
        
        # Check if this slot would share at least one position with existing words
        if slot_positions & existing_positions:
            intersecting_slots.append(slot)
    
    return intersecting_slots

@dataclass
class SolverState:
    """Represents a solver state for backtracking."""
    word_placements: List[WordPlacement]
    blocked_cells: Set[Tuple[int, int]]  # Store blocked cell positions
    fill_percentage: float
    fitness_score: float
    iteration: int
    conflicts: int = 0
    
    def __hash__(self):
        # Create hash based on placed words and their positions
        word_sigs: List[str] = []
        for wp in self.word_placements:
            sig = f"{wp.word}-{wp.row}-{wp.col}-{wp.direction.value}"
            word_sigs.append(sig)
        return hash(tuple(sorted(word_sigs)))

class CSPSolver:
    """
    Main CSP solver for crossword generation with enhanced heuristics.
    """
    
    def __init__(self, word_data_manager: WordDataManager, preferred_length: int = 6):
        self.word_data_manager = word_data_manager
        self.preferred_length = preferred_length
        self.word_index = FastWordIndex(word_data_manager, preferred_length)
        self.best_state: Optional[SolverState] = None
        self.backtrack_count = 0  # Track number of backtracks
        
        # Enhanced heuristics for word placement
        self.heuristic = Heuristics()
        
        # Crossword fitness scoring weights - balance quality aspects
        self.fitness_weights = {
            'intersectionality': 1,  # High weight: intersections are crucial for quality
            'fill_efficiency': 1,    # High weight: space utilization important
            'word_density': 0.4,       # Medium weight: more words = richer puzzle
        }
    
    def solve(self, creator: CrosswordCreator, max_iterations: int = 25000, max_consecutive_failures: int = 5,
              target_fill: float = 90.0, place_blocked_squares: bool = True, random_seed: Optional[int] = None) -> bool:
        """
        Solve crossword using CSP (Constraint Satisfaction Programming)
        
        Args:
            creator: CrosswordCreator instance to work with
            max_iterations: Maximum solver iterations
            max_consecutive_failures: Max failures before aggressive backtracking
            target_fill: Target fill percentage
            random_seed: Random seed for reproducible results (optional)
            
        Returns:
            True if successfully filled grid to target, False otherwise
        """
        top_n = 3 # Number of scored slots to consider
        
        # Set random seed for reproducible results
        if random_seed is not None:
            random.seed(random_seed)
        else:
            random.seed()
        
        iterations = 0
        consecutive_failures = 0
        used_words = set()  # Track used words to avoid duplicates
        
        # Track existing words to avoid duplicates (in case grid isn't empty)
        for placement in creator.word_placements:
            used_words.add(placement.word.upper())
        
        # Initialize state tracking with fitness scoring
        current_fill = self._calculate_fill_percentage(creator)
        initial_fitness = self.calculate_puzzle_fitness_score(creator)
        self.best_state = SolverState(
            creator.word_placements.copy(), 
            creator.grid.blocked_cells.copy(),
            current_fill, 
            initial_fitness['total_fitness'], 
            0
        )
        
        print(f"Starting CSP solver with {max_iterations} iterations")
        print(f"Target Fill: {target_fill}%")
        print(f"Random seed: {random_seed}" if random_seed is not None else '')
        print(f"intersections={self.heuristic.weights['intersection_potential']:.0f}, "
              f"constraints={self.heuristic.weights['constraint_satisfaction']:.1f}")
        print("")
        
        while iterations < max_iterations:
            iterations += 1

            current_fill = self._calculate_fill_percentage(creator)
            
            # Periodic status reporting
            if iterations % 500 == 0:
                print(f"Status at iteration {iterations}: "
                      f"fill={current_fill:.1f}%, "
                      f"words={len(creator.word_placements)}, "
                      f"intersections={creator.get_puzzle_statistics()['intersection_count']}, "
                      f"backtracks={self.backtrack_count}")
            
            # Use find_intersecting_slots to ensure connectivity
            empty_slots = find_intersecting_slots(creator.grid, creator.word_placements, min_length=3)
        
            current_fitness = self.calculate_puzzle_fitness_score(creator)
            total_fitness = current_fitness['total_fitness']
            
            # Update best state if fitness is higher and puzzle is connected
            if total_fitness > self.best_state.fitness_score and current_fitness['stats_used']['is_connected']:
                self.best_state = SolverState(
                    creator.word_placements.copy(), 
                    creator.grid.blocked_cells.copy(),
                    current_fill, 
                    total_fitness, 
                    iterations
                )
                self.backtrack_count = 0
                consecutive_failures = 0
                print(f"Iteration {iterations}: New best fitness {total_fitness:.1f} "
                      f"(fill={current_fill:.1f}%, words={len(creator.word_placements)}, "
                      f"intersections={current_fitness['stats_used']['intersection_count']})")
            

            if current_fill >= target_fill and current_fitness['stats_used']['is_connected']:
                print(f"[SUCCESS] Target achieved! Fill: {current_fill:.1f}% in {iterations} iterations")
                break
            
            # Check if we have any empty slots to work with
            if len(empty_slots) == 0:
                if len(creator.word_placements) > 0:
                    # We have no intersecting slots available
                    all_empty_slots = find_empty_slots(creator.grid, min_length=3)
                    if len(all_empty_slots) > 0:
                        print(f"[ERROR] {len(all_empty_slots)} empty slots exist but none intersect with existing words")
                consecutive_failures += 1
                continue
            
            # Simple backtracking on consecutive failures only
            if consecutive_failures >= max_consecutive_failures:
                backtrack_success = self._backtrack(creator, used_words)
                if backtrack_success:
                    consecutive_failures = 0
                    self.backtrack_count += 1
                    
                    # Prevent infinite loops with excessive backtracks
                    if self.backtrack_count >= max_iterations // 25:
                        print(f"Too many backtracks ({self.backtrack_count})")
                        break
                    continue
                else:
                    print("Backtracking failed - no more words to remove")
                    break
            
            slot_words = {}

            for slot in empty_slots:
                compatible_words = self.word_index.find_compatible_words(slot, max_results=100)
                available_words = [word for word in compatible_words if word.upper() not in used_words]
                slot_words[slot] = available_words

            def scoreSlot(slot: Slot) -> float:
                available_words = slot_words[slot]
                if available_words and len(available_words) > 0:
                    score = self.heuristic.evaluate_slot(slot, available_words, creator.word_placements, self.preferred_length)
                    return score
                return 0
            
            filtered_slots = [slot for slot in empty_slots if len(slot_words[slot]) > 0]

            if len(filtered_slots) == 0:
                consecutive_failures += 1
                continue
            
            # Sort slots by score and select top N
            filtered_slots.sort(key=scoreSlot, reverse=True)
            selected_slots = filtered_slots[:top_n]

            selected_slot = random.choice(selected_slots)
            selected_words = slot_words[selected_slot][:top_n*2]

            selected_word = random.choice(selected_words)
                  
            if not selected_slot or not selected_word:
                consecutive_failures += 1
                continue
            
            # Validate that the word will create actual intersections before placing
            if not validate_word_creates_intersections(selected_word.upper(), selected_slot, creator.word_placements):
                consecutive_failures += 1
                continue
            
            # Try to place the best available word for the selected slot and blocks the end if enabled
            placed = False
            if creator.place_word_in_slot(selected_word.upper(), selected_slot, blocked_number=len(creator.grid.blocked_cells), place_blocked_squares=place_blocked_squares, blocked_ratio=0.1):
                placed = True
                consecutive_failures = 0
                used_words.add(selected_word.upper())
                continue
        
            if not placed:
                consecutive_failures += 1

        final_stats = creator.get_puzzle_statistics()
        final_fitness = self.calculate_puzzle_fitness_score(creator)

         # Restore best state
        if len(creator.word_placements) > 1:
            # If current state is worse than best state, restore it
            if final_fitness['total_fitness'] < self.best_state.fitness_score:
                print(f"\nAttempting to restore best state. final fitness score: {final_fitness['total_fitness']:.1f}, best fitness score: {self.best_state.fitness_score:.1f}")
                self._restore_best_state(creator, used_words)
                final_stats = creator.get_puzzle_statistics()
                final_fitness = self.calculate_puzzle_fitness_score(creator)
                print(f"Restored state: {final_stats['fill_percentage']:.1f}% fill, fitness: {final_fitness['total_fitness']:.1f}, connected: {final_stats['is_connected']}")
        
        return True

    def _calculate_fill_percentage(self, creator: CrosswordCreator) -> float:
        """Calculate accurate fill percentage accounting for blocked squares."""
        stats = creator.get_puzzle_statistics()
        return stats['fill_percentage']
    
    def calculate_puzzle_fitness_score(self, creator: CrosswordCreator) -> Dict[str, float]:
        """
        Calculate comprehensive fitness score for the crossword puzzle.
        
        This method evaluates puzzle quality across multiple dimensions:
        - Intersectionality: How well words intersect with each other
        - Fill Efficiency: How well the grid space is utilized  
        - Word Density: Number of words relative to grid size
        - Grid Efficiency: Quality of word interconnections
        
        Args:
            creator: CrosswordCreator instance to evaluate
            
        Returns:
            Dictionary with detailed scoring breakdown and total fitness score
        """
        stats = creator.get_puzzle_statistics()
        grid_size = creator.grid.size
        
        # Hard requirement: puzzle must be connected (or have ≤1 word)
        connectivity_multiplier = 1.0 if stats['word_count'] <= 1 or stats['is_connected'] else 0.0
        
        # 1. Intersectionality Score (0-100)
        # More intersections = better word interplay, with diminishing returns
        intersection_count = stats['intersection_count']
        word_count = max(1, stats['word_count'])  # Avoid division by zero
        
        # Target: ~1.5-2 intersections per word is excellent
        intersection_ratio = intersection_count / word_count
        intersectionality_score = intersection_ratio * 25 + intersection_count
        
        # 3. Word Density Score (0-100)
        # More words = richer puzzle, normalized by grid size
        usable_cells = stats['total_cells'] - stats['blocked_cells']
        word_density_score = abs(1 - ((word_count / usable_cells) * self.preferred_length)) * 100
        

        # Calculate weighted total fitness score
        total_fitness = (
            intersectionality_score * self.fitness_weights['intersectionality'] +
            stats['fill_percentage'] * self.fitness_weights['fill_efficiency'] +
            word_density_score * self.fitness_weights['word_density']
        ) * connectivity_multiplier
        
        return {
            'total_fitness': total_fitness,
            'connectivity_multiplier': connectivity_multiplier,
            'intersectionality_score': intersectionality_score,
            'word_density_score': word_density_score,
            'stats_used': {
                'word_count': word_count,
                'intersection_count': intersection_count,
                'fill_percentage': stats['fill_percentage'],
                'is_connected': stats['is_connected'],
                'usable_cells': usable_cells
            }
        }
    
    def _backtrack(self, creator: CrosswordCreator, used_words: Set[str]) -> bool:
        """
        Variable backtracking - remove different amounts each time to break cycles.
        Simply removes the most recently added words to ensure we return to a valid state.
        """
        if len(creator.word_placements) <= 1:
            print("Cannot backtrack: only 1 or fewer words placed")
            return False
        
        num_to_remove = exponential_random_choice(len(creator.word_placements), lambd=0.3)
        
        # Get the last N word placements (most recently added)
        words_to_remove = creator.word_placements[-num_to_remove:]
        removed_count = 0
        
        # Remove words from most recent to oldest
        for word_placement in reversed(words_to_remove):
            removed_word = word_placement.word.upper()
            success = creator.remove_word(word_placement)
            if success:
                used_words.discard(removed_word)
                removed_count += 1
        
        return removed_count > 0

    
    def get_puzzle_fitness_breakdown(self, creator: CrosswordCreator) -> Dict[str, float]:
        """
        Get detailed fitness breakdown for external analysis.
        
        Args:
            creator: CrosswordCreator instance to evaluate
            
        Returns:
            Dictionary with comprehensive fitness scoring breakdown
        """
        return self.calculate_puzzle_fitness_score(creator)
    
    def _restore_best_state(self, creator: CrosswordCreator, used_words: Set[str]):
        """Restore the best fitness state found so far."""
        print(f"Restoring best state: fitness={self.best_state.fitness_score:.1f}, fill={self.best_state.fill_percentage:.1f}%")
        
        # Clear current state completely
        creator.grid = CrosswordGrid(creator.grid.size)
        creator.word_placements.clear()
        used_words.clear()
        
        # Restore blocked cells first
        for row, col in self.best_state.blocked_cells:
            creator.grid.set_blocked(row, col, True)
        
        # Then restore word placements
        for placement in self.best_state.word_placements:
            success = creator.place_word_placement(placement)
            if success:
                used_words.add(placement.word.upper())
            elif placement.word:  # Only log if word is not empty
                print(f"Warning: Failed to restore placement for '{placement.word}' at ({placement.row},{placement.col}) {placement.direction.value}")

def validate_word_creates_intersections(word: str, slot: Slot, existing_placements: List[WordPlacement]) -> bool:
    """
    Validate that placing a word in a slot will create actual intersections with existing words.
    
    Args:
        word: Word to be placed
        slot: Slot where word will be placed
        existing_placements: Current word placements in the grid
        
    Returns:
        True if word creates valid intersections, False otherwise
    """
    if not existing_placements:
        return True  # First word doesn't need intersections
    
    # Create a temporary placement for the word
    temp_placement = WordPlacement(word.upper(), slot.row, slot.col, slot.direction)
    word_positions = set(temp_placement.get_positions())
    
    # Check if word actually intersects with any existing word
    intersections_found = 0
    for existing_placement in existing_placements:
        existing_positions = set(existing_placement.get_positions())
        intersection_positions = word_positions & existing_positions
        
        if intersection_positions:
            # Validate that letters match at intersection points
            for row, col in intersection_positions:
                word_letter = temp_placement.get_letter_at_position(row, col)
                existing_letter = existing_placement.get_letter_at_position(row, col)
                if word_letter == existing_letter and word_letter:
                    intersections_found += 1
                else:
                    return False  # Letters don't match at intersection
    
    return intersections_found > 0  # Must have at least one valid intersection