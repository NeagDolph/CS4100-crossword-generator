"""
Simulated Annealing Solver Module for Crossword Generation

Implements simulated annealing optimization techniques for automatic crossword generation.
Uses temperature-based acceptance criteria and perturbation operations for word placement.
Enhanced with multi-objective fitness evaluation and adaptive cooling schedules.
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy

import numpy as np

from .crossword_creator import CrosswordCreator
from .crossword_grid import CrosswordGrid
from .word_placement import WordPlacement, Direction
from .crossword_validator import CrosswordValidator
from .word_data import WordDataManager
from .util import exponential_random_choice, frequency_score

# Reuse helper classes and functions from CSP solver
from .csp_solver import (
    Slot, FastWordIndex, find_empty_slots, find_intersecting_slots,
    validate_word_creates_intersections
)

class PerturbationType(Enum):
    """Types of perturbation operations in simulated annealing."""
    ADD_WORD = "add_word"
    REMOVE_WORD = "remove_word"
    SWAP_WORD = "swap_word"
    RELOCATE_WORD = "relocate_word"

class CoolingSchedule(Enum):
    """Different cooling schedule strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"

@dataclass
class SAState:
    """Represents a state in simulated annealing optimization."""
    word_placements: List[WordPlacement]
    blocked_cells: Set[Tuple[int, int]]
    fill_percentage: float
    fitness_score: float
    temperature: float
    iteration: int
    energy: float = 0.0  # Energy = -fitness_score (minimization problem)
    
    def __post_init__(self):
        self.energy = -self.fitness_score  # Convert to minimization problem
    
    def copy(self):
        """Create a deep copy of the state."""
        return SAState(
            word_placements=copy.deepcopy(self.word_placements),
            blocked_cells=self.blocked_cells.copy(),
            fill_percentage=self.fill_percentage,
            fitness_score=self.fitness_score,
            temperature=self.temperature,
            iteration=self.iteration,
            energy=self.energy
        )
    
    def __hash__(self):
        # Create hash based on placed words and their positions
        word_sigs: List[str] = []
        for wp in self.word_placements:
            sig = f"{wp.word}-{wp.row}-{wp.col}-{wp.direction.value}"
            word_sigs.append(sig)
        return hash(tuple(sorted(word_sigs)))

class SAFitnessEvaluator:
    """
    Fitness evaluator for simulated annealing crossword generation.
    Evaluates crossword quality across multiple dimensions.
    """
    
    def __init__(self, preferred_length: int = 6):
        self.preferred_length = preferred_length
        self.weights = {
            'connectivity': 50.0,      # Hard requirement: puzzle must be connected
            'intersections': 2.0,      # More intersections = better
            'fill_efficiency': 1.0,    # Better space utilization
            'word_count': 0.5,         # More words = richer puzzle
            'length_diversity': 0.3,   # Variety in word lengths
            'compactness': 0.2,        # Prefer compact layouts
        }
    
    def evaluate_fitness(self, creator: CrosswordCreator) -> float:
        """
        Calculate comprehensive fitness score for the crossword puzzle.
        
        Args:
            creator: CrosswordCreator instance to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        stats = creator.get_puzzle_statistics()
        
        # Hard requirement: puzzle must be connected (or have ≤1 word)
        if stats['word_count'] > 1 and not stats['is_connected']:
            return 0.0  # Invalid puzzle
        
        connectivity_score = self.weights['connectivity'] if stats['is_connected'] else 0.0
        
        # Intersection score - reward intersections but with diminishing returns
        intersection_count = stats['intersection_count']
        word_count = max(1, stats['word_count'])
        intersection_score = min(50.0, intersection_count * self.weights['intersections'])
        
        # Fill efficiency score
        fill_score = stats['fill_percentage'] * self.weights['fill_efficiency']
        
        # Word count score - more words up to a reasonable limit
        optimal_words = max(10, stats['total_cells'] // (self.preferred_length * 2))
        word_count_score = min(20.0, (word_count / optimal_words) * 20.0) * self.weights['word_count']
        
        # Length diversity score
        if word_count > 1:
            word_lengths = [len(wp.word) for wp in creator.word_placements]
            length_std = np.std(word_lengths) if len(word_lengths) > 1 else 0
            diversity_score = min(10.0, length_std * 2) * self.weights['length_diversity']
        else:
            diversity_score = 0.0
        
        # Compactness score - prefer words clustered together
        compactness_score = self._calculate_compactness(creator) * self.weights['compactness']
        
        total_fitness = (
            connectivity_score +
            intersection_score +
            fill_score +
            word_count_score +
            diversity_score +
            compactness_score
        )
        
        return total_fitness
    
    def _calculate_compactness(self, creator: CrosswordCreator) -> float:
        """Calculate how compact the word layout is."""
        if len(creator.word_placements) <= 1:
            return 5.0
        
        # Get all occupied positions
        occupied_positions = set()
        for wp in creator.word_placements:
            occupied_positions.update(wp.get_positions())
        
        if not occupied_positions:
            return 0.0
        
        # Calculate bounding box
        rows = [pos[0] for pos in occupied_positions]
        cols = [pos[1] for pos in occupied_positions]
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        occupied_count = len(occupied_positions)
        
        # Compactness = occupied cells / bounding box area
        compactness = occupied_count / bbox_area if bbox_area > 0 else 0
        return compactness * 10.0  # Scale to reasonable range

class SimulatedAnnealingSolver:
    """
    Main simulated annealing solver for crossword generation.
    Uses temperature-based optimization with multiple perturbation operations.
    """
    
    def __init__(self, word_data_manager: WordDataManager, preferred_length: int = 6):
        self.word_data_manager = word_data_manager
        self.preferred_length = preferred_length
        self.word_index = FastWordIndex(word_data_manager, preferred_length)
        self.fitness_evaluator = SAFitnessEvaluator(preferred_length)
        
        # Simulated annealing parameters
        self.initial_temperature = 100.0
        self.final_temperature = 0.01
        self.cooling_schedule = CoolingSchedule.EXPONENTIAL
        self.cooling_rate = 0.995
        
        # Perturbation weights (sum should be 1.0)
        self.perturbation_weights = {
            PerturbationType.ADD_WORD: 0.5,
            PerturbationType.REMOVE_WORD: 0.2,
            PerturbationType.SWAP_WORD: 0.2,
            PerturbationType.RELOCATE_WORD: 0.1
        }
        
        # State tracking
        self.current_state: Optional[SAState] = None
        self.best_state: Optional[SAState] = None
        self.accepted_moves = 0
        self.rejected_moves = 0
    
    def solve(self, creator: CrosswordCreator, max_iterations: int = 10000, 
              target_fill: float = 80.0, place_blocked_squares: bool = True,
              random_seed: Optional[int] = None) -> bool:
        """
        Generate crossword using simulated annealing optimization.
        
        Args:
            creator: CrosswordCreator instance to work with
            max_iterations: Maximum solver iterations
            target_fill: Target fill percentage
            place_blocked_squares: Whether to place blocked squares
            random_seed: Random seed for reproducible results
            
        Returns:
            True if successfully generated crossword, False otherwise
        """
        # Set random seed for reproducible results
        if random_seed is not None:
            random.seed(random_seed)
        else:
            random.seed()
        
        # Initialize state
        initial_fitness = self.fitness_evaluator.evaluate_fitness(creator)
        initial_fill = self._calculate_fill_percentage(creator)
        
        self.current_state = SAState(
            word_placements=creator.word_placements.copy(),
            blocked_cells=creator.grid.blocked_cells.copy(),
            fill_percentage=initial_fill,
            fitness_score=initial_fitness,
            temperature=self.initial_temperature,
            iteration=0
        )
        
        self.best_state = self.current_state.copy()
        self.accepted_moves = 0
        self.rejected_moves = 0
        used_words = {wp.word.upper() for wp in creator.word_placements}
        
        print(f"Starting Simulated Annealing solver with {max_iterations} iterations")
        print(f"Initial temperature: {self.initial_temperature}")
        print(f"Target Fill: {target_fill}%")
        print(f"Cooling schedule: {self.cooling_schedule.value}")
        print("")
        
        for iteration in range(max_iterations):
            # Update temperature
            temperature = self._update_temperature(iteration, max_iterations)
            self.current_state.temperature = temperature
            self.current_state.iteration = iteration
            
            # Periodic status reporting
            if iteration % 1000 == 0 and iteration > 0:
                acceptance_rate = self.accepted_moves / (self.accepted_moves + self.rejected_moves) * 100
                print(f"Iteration {iteration}: temp={temperature:.3f}, "
                      f"fill={self.current_state.fill_percentage:.1f}%, "
                      f"fitness={self.current_state.fitness_score:.1f}, "
                      f"words={len(self.current_state.word_placements)}, "
                      f"acceptance={acceptance_rate:.1f}%")
            
            # Check termination conditions
            if self.current_state.fill_percentage >= target_fill:
                print(f"[SUCCESS] Target achieved! Fill: {self.current_state.fill_percentage:.1f}% "
                      f"in {iteration} iterations")
                break
            
            if temperature < self.final_temperature:
                print(f"[TERMINATION] Final temperature reached at iteration {iteration}")
                break
            
            # Generate neighbor state through perturbation
            neighbor_creator = self._create_neighbor_state(creator, used_words, place_blocked_squares)
            if neighbor_creator is None:
                continue
            
            # Evaluate neighbor state
            neighbor_fitness = self.fitness_evaluator.evaluate_fitness(neighbor_creator)
            neighbor_fill = self._calculate_fill_percentage(neighbor_creator)
            
            neighbor_state = SAState(
                word_placements=neighbor_creator.word_placements.copy(),
                blocked_cells=neighbor_creator.grid.blocked_cells.copy(),
                fill_percentage=neighbor_fill,
                fitness_score=neighbor_fitness,
                temperature=temperature,
                iteration=iteration
            )
            
            # Accept or reject the move
            if self._accept_move(self.current_state, neighbor_state):
                # Accept the move
                self._apply_state_to_creator(neighbor_state, creator)
                self.current_state = neighbor_state
                self.accepted_moves += 1
                
                # Update used words
                used_words.clear()
                used_words.update(wp.word.upper() for wp in creator.word_placements)
                
                # Update best state if better
                if neighbor_fitness > self.best_state.fitness_score:
                    self.best_state = neighbor_state.copy()
                    print(f"New best state at iteration {iteration}: "
                          f"fitness={neighbor_fitness:.1f}, fill={neighbor_fill:.1f}%")
            else:
                # Reject the move
                self.rejected_moves += 1
        
        # Restore best state
        if self.best_state.fitness_score > self.current_state.fitness_score:
            print(f"Restoring best state: fitness={self.best_state.fitness_score:.1f}")
            self._apply_state_to_creator(self.best_state, creator)
            self.current_state = self.best_state
        
        final_stats = creator.get_puzzle_statistics()
        acceptance_rate = self.accepted_moves / (self.accepted_moves + self.rejected_moves) * 100
        
        print(f"\nFinal Results:")
        print(f"Fill: {final_stats['fill_percentage']:.1f}%")
        print(f"Words: {final_stats['word_count']}")
        print(f"Intersections: {final_stats['intersection_count']}")
        print(f"Connected: {final_stats['is_connected']}")
        print(f"Final fitness: {self.current_state.fitness_score:.1f}")
        print(f"Acceptance rate: {acceptance_rate:.1f}%")
        
        return final_stats['fill_percentage'] >= target_fill * 0.8  # Accept if within 80% of target
    
    def _update_temperature(self, iteration: int, max_iterations: int) -> float:
        """Update temperature according to cooling schedule."""
        if self.cooling_schedule == CoolingSchedule.LINEAR:
            return self.initial_temperature * (1 - iteration / max_iterations)
        
        elif self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return self.initial_temperature * (self.cooling_rate ** iteration)
        
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return self.initial_temperature / (1 + math.log(1 + iteration))
        
        elif self.cooling_schedule == CoolingSchedule.ADAPTIVE:
            # Adaptive cooling based on acceptance rate
            recent_acceptance_rate = self.accepted_moves / max(1, self.accepted_moves + self.rejected_moves)
            if recent_acceptance_rate > 0.6:
                # Cool faster if accepting too many moves
                rate = self.cooling_rate * 0.99
            elif recent_acceptance_rate < 0.1:
                # Cool slower if rejecting too many moves
                rate = self.cooling_rate * 1.01
            else:
                rate = self.cooling_rate
            
            return self.initial_temperature * (rate ** iteration)
        
        return self.initial_temperature * (self.cooling_rate ** iteration)
    
    def _create_neighbor_state(self, creator: CrosswordCreator, used_words: Set[str], 
                              place_blocked_squares: bool) -> Optional[CrosswordCreator]:
        """
        Create a neighbor state by applying a random perturbation.
        
        Returns:
            New CrosswordCreator instance with perturbed state, or None if perturbation failed
        """
        # Create a copy of the creator to work with
        neighbor_creator = self._copy_creator(creator)
        
        # Choose perturbation type based on weights and current state
        perturbation_type = self._choose_perturbation_type(len(creator.word_placements))
        
        if perturbation_type == PerturbationType.ADD_WORD:
            return self._add_word_perturbation(neighbor_creator, used_words, place_blocked_squares)
        
        elif perturbation_type == PerturbationType.REMOVE_WORD:
            return self._remove_word_perturbation(neighbor_creator, used_words)
        
        elif perturbation_type == PerturbationType.SWAP_WORD:
            return self._swap_word_perturbation(neighbor_creator, used_words)
        
        elif perturbation_type == PerturbationType.RELOCATE_WORD:
            return self._relocate_word_perturbation(neighbor_creator, used_words, place_blocked_squares)
        
        return None
    
    def _choose_perturbation_type(self, word_count: int) -> PerturbationType:
        """Choose perturbation type based on current state and weights."""
        # Adjust weights based on current state
        adjusted_weights = self.perturbation_weights.copy()
        
        if word_count == 0:
            # Only add words if no words placed
            return PerturbationType.ADD_WORD
        
        elif word_count == 1:
            # Prefer adding words when only one word
            adjusted_weights[PerturbationType.ADD_WORD] = 0.8
            adjusted_weights[PerturbationType.REMOVE_WORD] = 0.1
            adjusted_weights[PerturbationType.SWAP_WORD] = 0.05
            adjusted_weights[PerturbationType.RELOCATE_WORD] = 0.05
        
        elif word_count < 5:
            # Prefer adding words when few words
            adjusted_weights[PerturbationType.ADD_WORD] = 0.6
            adjusted_weights[PerturbationType.REMOVE_WORD] = 0.15
            adjusted_weights[PerturbationType.SWAP_WORD] = 0.15
            adjusted_weights[PerturbationType.RELOCATE_WORD] = 0.1
        
        # Choose randomly based on adjusted weights
        types = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        return random.choices(types, weights=weights)[0]
    
    def _add_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str], 
                              place_blocked_squares: bool) -> Optional[CrosswordCreator]:
        """Add a new word to the crossword."""
        # Find available slots
        if len(creator.word_placements) == 0:
            # First word - can place anywhere
            empty_slots = find_empty_slots(creator.grid, min_length=3)
        else:
            # Must intersect with existing words
            empty_slots = find_intersecting_slots(creator.grid, creator.word_placements, min_length=3)
        
        if not empty_slots:
            return None
        
        # Try multiple slots
        random.shuffle(empty_slots)
        for slot in empty_slots[:5]:  # Try up to 5 slots
            compatible_words = self.word_index.find_compatible_words(slot, max_results=50)
            available_words = [word for word in compatible_words if word.upper() not in used_words]
            
            if available_words:
                word = random.choice(available_words[:10])  # Choose from top 10
                
                # Validate intersections if not first word
                if len(creator.word_placements) > 0:
                    if not validate_word_creates_intersections(word.upper(), slot, creator.word_placements):
                        continue
                
                # Try to place the word
                success = creator.place_word_in_slot(word.upper(), slot, 
                                                   blocked_number=len(creator.grid.blocked_cells),
                                                   place_blocked_squares=place_blocked_squares,
                                                   blocked_ratio=0.1)
                if success:
                    return creator
        
        return None
    
    def _remove_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Remove a word from the crossword."""
        if len(creator.word_placements) <= 1:
            return None  # Don't remove if only one word
        
        # Choose a word to remove (prefer words with fewer intersections)
        word_intersection_counts = []
        for wp in creator.word_placements:
            intersection_count = 0
            for other_wp in creator.word_placements:
                if wp != other_wp:
                    intersections = CrosswordValidator.get_intersections(wp, other_wp)
                    intersection_count += len(intersections)
            word_intersection_counts.append((wp, intersection_count))
        
        # Sort by intersection count (fewer intersections first)
        word_intersection_counts.sort(key=lambda x: x[1])
        
        # Remove one of the words with fewer intersections
        candidates = word_intersection_counts[:len(word_intersection_counts)//2 + 1]
        word_to_remove, _ = random.choice(candidates)
        
        success = creator.remove_word(word_to_remove)
        if success:
            used_words.discard(word_to_remove.word.upper())
            return creator
        
        return None
    
    def _swap_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Swap a word with a different word in the same position."""
        if not creator.word_placements:
            return None
        
        # Choose a random word to swap
        word_to_swap = random.choice(creator.word_placements)
        
        # Create slot from the word's position
        slot = Slot(word_to_swap.row, word_to_swap.col, word_to_swap.direction, 
                   len(word_to_swap.word))
        
        # Find constraints from intersecting words
        for other_wp in creator.word_placements:
            if other_wp != word_to_swap:
                intersections = CrosswordValidator.get_intersections(word_to_swap, other_wp)
                for row, col in intersections:
                    if word_to_swap.direction == Direction.ACROSS:
                        pos_in_word = col - word_to_swap.col
                    else:
                        pos_in_word = row - word_to_swap.row
                    
                    if 0 <= pos_in_word < len(word_to_swap.word):
                        intersection_letter = other_wp.get_letter_at_position(row, col)
                        if intersection_letter:
                            slot.constraints[pos_in_word] = intersection_letter
        
        # Find compatible words (excluding the current word)
        compatible_words = self.word_index.find_compatible_words(slot, max_results=50)
        available_words = [word for word in compatible_words 
                          if word.upper() not in used_words and word.upper() != word_to_swap.word.upper()]
        
        if available_words:
            # Remove the old word
            creator.remove_word(word_to_swap)
            used_words.discard(word_to_swap.word.upper())
            
            # Try to place a new word
            new_word = random.choice(available_words[:10])
            success = creator.place_word(new_word.upper(), slot.row, slot.col, slot.direction)
            
            if success:
                return creator
            else:
                # Restore the old word if placement failed
                creator.place_word(word_to_swap.word, word_to_swap.row, word_to_swap.col, 
                                 word_to_swap.direction, word_to_swap.clue)
                used_words.add(word_to_swap.word.upper())
        
        return None
    
    def _relocate_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str], 
                                   place_blocked_squares: bool) -> Optional[CrosswordCreator]:
        """Relocate a word to a different position."""
        if not creator.word_placements:
            return None
        
        # Choose a word to relocate
        word_to_relocate = random.choice(creator.word_placements)
        original_word = word_to_relocate.word
        original_clue = word_to_relocate.clue
        
        # Remove the word temporarily
        creator.remove_word(word_to_relocate)
        used_words.discard(word_to_relocate.word.upper())
        
        # Find new positions for the word
        if len(creator.word_placements) == 0:
            # If this was the only word, can place anywhere
            available_slots = find_empty_slots(creator.grid, min_length=len(original_word))
        else:
            # Must intersect with remaining words
            available_slots = find_intersecting_slots(creator.grid, creator.word_placements, 
                                                    min_length=len(original_word))
        
        # Filter slots that can fit the word
        suitable_slots = []
        for slot in available_slots:
            if slot.length >= len(original_word):
                # Create a temporary slot of exact word length
                temp_slot = Slot(slot.row, slot.col, slot.direction, len(original_word), slot.constraints)
                if temp_slot.matches_word(original_word):
                    suitable_slots.append(temp_slot)
        
        if suitable_slots:
            # Try to place at a new location
            new_slot = random.choice(suitable_slots)
            success = creator.place_word(original_word, new_slot.row, new_slot.col, 
                                       new_slot.direction, original_clue)
            if success:
                return creator
        
        # If relocation failed, restore the word at original position
        creator.place_word(original_word, word_to_relocate.row, word_to_relocate.col, 
                         word_to_relocate.direction, original_clue)
        used_words.add(original_word.upper())
        
        return None
    
    def _accept_move(self, current_state: SAState, neighbor_state: SAState) -> bool:
        """
        Decide whether to accept a move using simulated annealing criteria.
        
        Args:
            current_state: Current state
            neighbor_state: Proposed neighbor state
            
        Returns:
            True if move should be accepted, False otherwise
        """
        # Always accept improvements
        if neighbor_state.fitness_score > current_state.fitness_score:
            return True
        
        # For worse moves, accept probabilistically based on temperature
        if current_state.temperature <= 0:
            return False
        
        # Calculate acceptance probability using Boltzmann distribution
        delta_energy = neighbor_state.energy - current_state.energy  # Change in energy
        acceptance_probability = math.exp(-delta_energy / current_state.temperature)
        
        return random.random() < acceptance_probability
    
    def _calculate_fill_percentage(self, creator: CrosswordCreator) -> float:
        """Calculate fill percentage."""
        stats = creator.get_puzzle_statistics()
        return stats['fill_percentage']
    
    def _copy_creator(self, creator: CrosswordCreator) -> CrosswordCreator:
        """Create a deep copy of CrosswordCreator for perturbation."""
        new_grid = creator.grid.copy()
        new_creator = CrosswordCreator(new_grid, creator.word_data_manager)
        new_creator.word_placements = copy.deepcopy(creator.word_placements)
        return new_creator
    
    def _apply_state_to_creator(self, state: SAState, creator: CrosswordCreator):
        """Apply a state to a CrosswordCreator instance."""
        # Clear current state
        creator.grid = CrosswordGrid(creator.grid.size)
        creator.word_placements.clear()
        
        # Restore blocked cells
        for row, col in state.blocked_cells:
            creator.grid.set_blocked(row, col, True)
        
        # Restore word placements
        for placement in state.word_placements:
            success = creator.place_word(placement.word, placement.row, placement.col, 
                                       placement.direction, placement.clue)
            if not success:
                print(f"Warning: Failed to restore placement for '{placement.word}'")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get solver statistics."""
        total_moves = self.accepted_moves + self.rejected_moves
        acceptance_rate = self.accepted_moves / total_moves if total_moves > 0 else 0
        
        return {
            'accepted_moves': self.accepted_moves,
            'rejected_moves': self.rejected_moves,
            'total_moves': total_moves,
            'acceptance_rate': acceptance_rate,
            'best_fitness': self.best_state.fitness_score if self.best_state else 0,
            'current_fitness': self.current_state.fitness_score if self.current_state else 0,
            'final_temperature': self.current_state.temperature if self.current_state else 0
        }

# Extension methods for CrosswordCreator to support slot-based placement
def place_word_in_slot(self, word: str, slot: Slot, blocked_number: int = 0, 
                      place_blocked_squares: bool = True, blocked_ratio: float = 0.1) -> bool:
    """
    Place a word in a specific slot, optionally adding blocked squares.
    
    Args:
        word: Word to place
        slot: Slot to place word in
        blocked_number: Current number of blocked squares
        place_blocked_squares: Whether to add blocked squares
        blocked_ratio: Target ratio of blocked squares
        
    Returns:
        True if word was successfully placed, False otherwise
    """
    # Check if word fits in slot
    if len(word) > slot.length:
        return False
    
    # Validate constraints
    if not slot.matches_word(word):
        return False
    
    # Try to place the word
    success = self.place_word(word, slot.row, slot.col, slot.direction)
    
    if success and place_blocked_squares:
        # Optionally add blocked squares around the word
        self._add_strategic_blocked_squares(word, slot, blocked_ratio)
    
    return success

def place_word_placement(self, placement: WordPlacement) -> bool:
    """Place a WordPlacement directly."""
    return self.place_word(placement.word, placement.row, placement.col, 
                          placement.direction, placement.clue)

def _add_strategic_blocked_squares(self, word: str, slot: Slot, blocked_ratio: float):
    """Add blocked squares strategically around placed words."""
    total_cells = self.grid.size * self.grid.size
    current_blocked = len(self.grid.blocked_cells)
    target_blocked = int(total_cells * blocked_ratio)
    
    if current_blocked >= target_blocked:
        return
    
    # Add blocked squares at word boundaries sometimes
    if random.random() < 0.3:  # 30% chance
        word_length = len(word)
        
        if slot.direction == Direction.ACROSS:
            # Try to block before and after the word
            before_pos = (slot.row, slot.col - 1)
            after_pos = (slot.row, slot.col + word_length)
        else:
            # Try to block above and below the word
            before_pos = (slot.row - 1, slot.col)
            after_pos = (slot.row + word_length, slot.col)
        
        for pos in [before_pos, after_pos]:
            if (self.grid.is_valid_position(pos[0], pos[1]) and 
                not self.grid.is_blocked(pos[0], pos[1]) and
                self.grid.get_letter(pos[0], pos[1]) == ''):
                
                self.grid.set_blocked(pos[0], pos[1], True)
                current_blocked += 1
                if current_blocked >= target_blocked:
                    break

# Monkey patch the methods onto CrosswordCreator
CrosswordCreator.place_word_in_slot = place_word_in_slot
CrosswordCreator.place_word_placement = place_word_placement
CrosswordCreator._add_strategic_blocked_squares = _add_strategic_blocked_squares