#!/usr/bin/env python3
"""
A sophisticated crossword puzzle generator using simulated annealing optimization
to create high-quality crosswords at three difficulty levels (Easy, Medium, Hard).

OVERVIEW:
--------
This system generates crossword puzzles by treating crossword creation as an 
optimization problem. Using simulated annealing, it iteratively places, removes, 
swaps, and relocates words to maximize puzzle quality while meeting difficulty-
specific targets for word count, intersections, and grid fill percentage.

FEATURES:
---------
• Multi-Difficulty Support: Easy (9x9), Medium (13x13), Hard (17x17) grids
• Real Clue Integration: Uses external CSV database of cryptic crossword clues
• Smart Optimization: Intersection-focused placement with target achievement bonuses
• Quality Metrics: Tracks connectivity, density, intersection count, and fill percentage
• Progressive Complexity: Each difficulty level uses tailored optimization strategies
• Duplicate Prevention: Robust validation to ensure unique word placements
• Interactive Interface: Choose individual difficulties or run complete progression

"""

import random
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import copy
import csv
import os
import sys
import time

# Import the WordDataManager from your word_data.py
try:
    from word_data import WordDataManager as BaseWordDataManager, WordClue
except ImportError:
    print("Error: Could not import WordDataManager from word_data.py")
    print("Please ensure word_data.py is in the same directory as this script")
    sys.exit(1)

# Create a wrapper class that handles different CSV column names
class WordDataManagerWrapper(BaseWordDataManager):
    """Wrapper that adapts to different CSV column naming conventions."""
    
    def load_data(self) -> bool:
        """
        Load word-clue data from CSV file with flexible column naming.
        Handles both 'word'/'clue' and 'answer'/'clue' column formats.
        """
        if not os.path.exists(self.csv_file_path):
            print(f"Warning: CSV file {self.csv_file_path} not found")
            return False
        
        try:
            # Try different encodings to handle various CSV formats
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(self.csv_file_path, 'r', encoding=encoding) as file:
                        reader = csv.DictReader(file)
                        
                        # Check what columns are available
                        fieldnames = reader.fieldnames
                        print(f"CSV columns found: {fieldnames}")
                        
                        # Determine which column names to use
                        word_column = None
                        clue_column = None
                        
                        if 'word' in fieldnames:
                            word_column = 'word'
                        elif 'answer' in fieldnames:
                            word_column = 'answer'
                        
                        if 'clue' in fieldnames:
                            clue_column = 'clue'
                        
                        if not word_column or not clue_column:
                            print(f"Error: Could not find required columns. Available: {fieldnames}")
                            print("Expected: 'word' or 'answer' AND 'clue'")
                            return False
                        
                        print(f"Using columns: word='{word_column}', clue='{clue_column}'")
                        
                        for row in reader:
                            # Skip rows with missing data
                            if not row.get(word_column) or not row.get(clue_column):
                                continue
                                
                            word = row[word_column].strip()
                            # Skip words containing dashes or spaces
                            if '-' in word or ' ' in word:
                                continue
                            
                            word_clue = WordClue(
                                word=word,
                                clue=row[clue_column].strip(),
                                date=row.get('puzzle_date', '').strip() if row.get('puzzle_date') else None
                            )
                            
                            self.word_clues.append(word_clue)
                            
                            # Build lookup dictionary
                            word = word_clue.word
                            if word not in self.word_to_clues:
                                self.word_to_clues[word] = []
                            self.word_to_clues[word].append(word_clue)
                    
                    self._loaded = True
                    print(f"Loaded {len(self.word_clues)} word-clue pairs from {self.csv_file_path} (encoding: {encoding})")
                    return True
                    
                except UnicodeDecodeError:
                    # Try next encoding
                    continue
            
            # If all encodings failed
            print(f"Error: Could not decode CSV file with any supported encoding")
            return False
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return False

# Utility functions
def exponential_random_choice(max_val: int, lambd: float = 0.5) -> int:
    """Choose a random number from 1 to max_val with exponential distribution."""
    if max_val <= 1:
        return 1
    # Generate exponential random variable
    x = random.expovariate(lambd)
    # Scale and clamp to [1, max_val]
    result = int(x * max_val / 3) + 1  # Divide by 3 to adjust scale
    return min(max_val, max(1, result))

def frequency_score(word: str) -> float:
    """Calculate frequency score based on common letters."""
    common_letters = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 6.97, 'N': 6.75,
        'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
        'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
        'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
        'Q': 0.10, 'Z': 0.07
    }
    
    if not word:
        return 0.0
    
    total_score = 0.0
    for char in word.upper():
        total_score += common_letters.get(char, 0.1)  # Low score for uncommon letters
    
    return total_score / len(word)  # Average frequency

# Core classes
class Direction(Enum):
    """Enumeration for word direction in crossword puzzle"""
    ACROSS = "across"
    DOWN = "down"

@dataclass
class WordPlacement:
    """Represents a word placed in the crossword puzzle."""
    word: str
    row: int
    col: int
    direction: Direction
    clue: str = ""
    number: int = 0
    
    def get_end_position(self) -> Tuple[int, int]:
        """Get the ending position of this word."""
        if self.direction == Direction.ACROSS:
            return (self.row, self.col + len(self.word) - 1)
        else:
            return (self.row + len(self.word) - 1, self.col)
    
    def get_positions(self) -> List[Tuple[int, int]]:
        """Get all grid positions this word occupies."""
        positions = []
        for i in range(len(self.word)):
            if self.direction == Direction.ACROSS:
                positions.append((self.row, self.col + i))
            else:
                positions.append((self.row + i, self.col))
        return positions
    
    def get_letter_at_position(self, row: int, col: int) -> str:
        """Get the letter at a specific position if this word occupies it."""
        positions = self.get_positions()
        if (row, col) not in positions:
            return ""
        
        if self.direction == Direction.ACROSS:
            letter_index = col - self.col
        else:
            letter_index = row - self.row
            
        return self.word[letter_index] if 0 <= letter_index < len(self.word) else ""

class CrosswordGrid:
    """Core grid representation for crossword puzzles."""
    
    def __init__(self, size: int = 15):
        self.size = size
        self.grid = np.full((size, size), '', dtype='<U1')
        self.blocked_cells: Set[Tuple[int, int]] = set()
        self.given_cells: Set[Tuple[int, int]] = set()
        self.user_cells: Set[Tuple[int, int]] = set()
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def is_blocked(self, row: int, col: int) -> bool:
        """Check if a cell is blocked (black square)."""
        return (row, col) in self.blocked_cells
    
    def set_blocked(self, row: int, col: int, blocked: bool = True):
        """Set a cell as blocked or unblocked."""
        if not self.is_valid_position(row, col):
            raise ValueError(f"Invalid position: ({row}, {col})")
        
        if blocked:
            self.blocked_cells.add((row, col))
            self.grid[row, col] = '#'
            self.given_cells.discard((row, col))
            self.user_cells.discard((row, col))
        else:
            self.blocked_cells.discard((row, col))
            self.grid[row, col] = ''
    
    def get_letter(self, row: int, col: int) -> str:
        """Get letter at position."""
        if not self.is_valid_position(row, col):
            return ''
        return self.grid[row, col]
    
    def set_letter(self, row: int, col: int, letter: str, is_given: bool = False):
        """Set letter at position."""
        if not self.is_valid_position(row, col):
            raise ValueError(f"Invalid position: ({row}, {col})")
        if self.is_blocked(row, col):
            raise ValueError(f"Cannot place letter in blocked cell: ({row}, {col})")
        
        self.grid[row, col] = letter.upper() if letter else ''
        
        pos = (row, col)
        if is_given:
            self.given_cells.add(pos)
            self.user_cells.discard(pos)
        else:
            self.user_cells.add(pos)
            self.given_cells.discard(pos)
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if a cell is empty (not blocked, no letter)."""
        if not self.is_valid_position(row, col):
            return False
        return not self.is_blocked(row, col) and self.get_letter(row, col) == ''
    
    def get_filled_cells(self) -> Set[Tuple[int, int]]:
        """Get all filled cells in the grid."""
        filled_cells = set()
        for row in range(self.size):
            for col in range(self.size):
                if self.get_letter(row, col) and not self.is_blocked(row, col):
                    filled_cells.add((row, col))
        return filled_cells
    
    def copy(self):
        """Create a deep copy of the grid."""
        new_grid = CrosswordGrid(self.size)
        new_grid.grid = self.grid.copy()
        new_grid.blocked_cells = self.blocked_cells.copy()
        new_grid.given_cells = self.given_cells.copy()
        new_grid.user_cells = self.user_cells.copy()
        return new_grid
    
    def __str__(self):
        """Pretty print the grid."""
        lines = []
        for row in self.grid:
            line = ""
            for cell in row:
                if cell == '':
                    line += '.'
                elif cell == '#':
                    line += '█'
                else:
                    line += cell
                line += ' '
            lines.append(line)
        return "\n".join(lines)

class CrosswordValidator:
    """Utility class for validation operations."""
    
    @staticmethod
    def can_place_word(grid: CrosswordGrid, word: str, row: int, col: int, 
                      direction: Direction) -> bool:
        """Check if a word can be placed at the given position."""
        word = word.upper()
        
        # Check bounds
        if direction == Direction.ACROSS:
            if col + len(word) > grid.size:
                return False
        else:  # DOWN
            if row + len(word) > grid.size:
                return False
        
        # Check each position
        for i, letter in enumerate(word):
            if direction == Direction.ACROSS:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            # Can't place in blocked cells
            if grid.is_blocked(r, c):
                return False
            
            # Check for conflicts with existing letters
            existing = grid.get_letter(r, c)
            if existing and existing != letter:
                return False
        
        return True
    
    @staticmethod
    def get_intersections(placement1: WordPlacement, placement2: WordPlacement) -> List[Tuple[int, int]]:
        """Find intersection points between two word placements."""
        positions1 = set(placement1.get_positions())
        positions2 = set(placement2.get_positions())
        return list(positions1.intersection(positions2))
    
    @staticmethod
    def validate_intersections(grid: CrosswordGrid, placements: List[WordPlacement]) -> bool:
        """Validate that all word intersections have matching letters."""
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                intersections = CrosswordValidator.get_intersections(p1, p2)
                for row, col in intersections:
                    letter1_idx = None
                    letter2_idx = None
                    
                    if p1.direction == Direction.ACROSS:
                        letter1_idx = col - p1.col
                    else:
                        letter1_idx = row - p1.row
                    
                    if p2.direction == Direction.ACROSS:
                        letter2_idx = col - p2.col
                    else:
                        letter2_idx = row - p2.row
                    
                    if (0 <= letter1_idx < len(p1.word) and 
                        0 <= letter2_idx < len(p2.word)):
                        if p1.word[letter1_idx] != p2.word[letter2_idx]:
                            return False
        return True
    
    @staticmethod
    def validate_grid_connectivity(placements: List[WordPlacement]) -> bool:
        """Validate that all words in the grid are connected."""
        if len(placements) <= 1:
            return True
        
        # Build adjacency graph of word intersections
        connected_components = []
        
        for i, placement in enumerate(placements):
            component_indices = []
            for j, component in enumerate(connected_components):
                for other_idx in component:
                    if CrosswordValidator.get_intersections(placement, placements[other_idx]):
                        component_indices.append(j)
                        break
            
            if not component_indices:
                connected_components.append({i})
            else:
                new_component = {i}
                for idx in sorted(component_indices, reverse=True):
                    new_component.update(connected_components.pop(idx))
                connected_components.append(new_component)
        
        return len(connected_components) <= 1

    @staticmethod
    def count_word_intersections(placements: List[WordPlacement]) -> int:
        """Count the total number of intersections between all word placements."""
        intersection_count = 0
        for i, p1 in enumerate(placements):
            for p2 in placements[i+1:]:
                intersections = CrosswordValidator.get_intersections(p1, p2)
                intersection_count += len(intersections)
        return intersection_count

class CrosswordCreator:
    """Handles crossword puzzle creation and generation."""
    
    def __init__(self, grid: CrosswordGrid, word_data_manager):
        self.grid = grid
        self.word_placements: List[WordPlacement] = []
        self.word_data_manager = word_data_manager
    
    def place_word(self, word: str, row: int, col: int, direction: Direction, 
                   clue: str = "") -> bool:
        """Attempt to place a word on the grid."""
        if not CrosswordValidator.can_place_word(self.grid, word, row, col, direction):
            return False
        
        # Check for duplicate word placements at the same position
        word_upper = word.upper()
        for existing_placement in self.word_placements:
            if (existing_placement.word == word_upper and 
                existing_placement.row == row and 
                existing_placement.col == col and 
                existing_placement.direction == direction):
                return False  # Already placed at this exact position
        
        if not clue and self.word_data_manager:
            clue = self.word_data_manager.get_clue_for_word(word) or ""
        
        placement = WordPlacement(word_upper, row, col, direction, clue)
        
        test_placements = self.word_placements + [placement]
        if not CrosswordValidator.validate_intersections(self.grid, test_placements):
            return False
        
        # Place the word
        for i, letter in enumerate(word_upper):
            if direction == Direction.ACROSS:
                self.grid.set_letter(row, col + i, letter, is_given=True)
            else:
                self.grid.set_letter(row + i, col, letter, is_given=True)
        
        self.word_placements.append(placement)
        return True
    
    def remove_word(self, placement: WordPlacement) -> bool:
        """Remove a word from the grid."""
        if placement not in self.word_placements:
            return False
        
        positions_to_clear = set(placement.get_positions())
        
        for other_placement in self.word_placements:
            if other_placement != placement:
                positions_to_clear -= set(other_placement.get_positions())
        
        for row, col in positions_to_clear:
            self.grid.set_letter(row, col, '')
            self.grid.given_cells.discard((row, col))
        
        self.word_placements.remove(placement)
        return True
    
    def get_puzzle_statistics(self) -> dict:
        """Get statistics about the current puzzle."""
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

# Slot and solver classes
@dataclass  
class Slot(WordPlacement):
    """Represents a slot for word placement with constraints."""
    constraints: Dict[int, str] = field(default_factory=dict)
    
    def __init__(self, row: int, col: int, direction: Direction, length: int, constraints: Dict[int, str] = None):
        placeholder_word = '?' * length
        super().__init__(placeholder_word, row, col, direction)
        self.constraints = constraints or {}
        self.length = length
    
    def matches_word(self, word: str) -> bool:
        """Check if a word matches this slot's constraints."""
        if len(word) > self.length:
            return False
        
        if self.constraints:
            min_constraint_pos = min(self.constraints.keys())
            if len(word) <= min_constraint_pos:
                return False
        
        for pos, required_letter in self.constraints.items():
            if pos < len(word):
                if word[pos].upper() != str(required_letter).upper():
                    return False
        return True

class FastWordIndex:
    """Fast word index for constraint-based word lookup."""
    
    def __init__(self, word_data_manager, preferred_length: int = 6):
        self.word_data_manager = word_data_manager
        self.preferred_length = preferred_length
        self.length_index: Dict[int, List[str]] = defaultdict(list)
        self.position_letter_index: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
        self._build_indexes()
    
    def _build_indexes(self):
        """Build all word indexes for fast lookup."""
        all_words = self.word_data_manager.get_all_words()
        
        for word in all_words:
            word_len = len(word)
            self.length_index[word_len].append(word)
            
            word_upper = word.upper()
            for pos, letter in enumerate(word_upper):
                self.position_letter_index[(pos, letter)].add(word)
    
    def find_compatible_words(self, slot: Slot, max_results: int = 100) -> List[str]:
        """Find words compatible with the given slot constraints."""
        min_word_length = 3
        
        if not slot.constraints:
            candidates = []
            for length in range(min_word_length, slot.length + 1):
                candidates.extend(self.length_index.get(length, []))
            return candidates[:max_results]
        
        compatible_words = None
        
        for pos, required_letter in sorted(slot.constraints.items()):
            words_with_letter = self.position_letter_index.get((pos, str(required_letter).upper()), set())
            valid_length_words = {w for w in words_with_letter 
                                if min_word_length <= len(w) <= slot.length and len(w) > pos}
            
            if compatible_words is None:
                compatible_words = valid_length_words
            else:
                compatible_words = compatible_words.intersection(valid_length_words)
            
            if not compatible_words:
                return []
        
        compatible = list(compatible_words) if compatible_words else []
        compatible.sort(key=len, reverse=True)
        return compatible[:max_results]

def find_empty_slots(grid: CrosswordGrid, min_length: int = 3) -> List[Slot]:
    """Find all slots in the grid that can accommodate new words."""
    slots: List[Slot] = []
    
    # Find horizontal slots
    for row in range(grid.size):
        black_positions = []
        for col in range(grid.size):
            if grid.is_blocked(row, col):
                black_positions.append(col)
        
        segment_starts = [0] + [pos + 1 for pos in black_positions]
        segment_ends = black_positions + [grid.size]
        
        for start, end in zip(segment_starts, segment_ends):
            segment_length = end - start
            if segment_length < min_length:
                continue
            
            constraints = {}
            for pos in range(start, end):
                letter = grid.get_letter(row, pos)
                if letter:
                    constraints[pos - start] = str(letter)
            
            slot = Slot(row, start, Direction.ACROSS, segment_length, constraints)
            slots.append(slot)
    
    # Find vertical slots
    for col in range(grid.size):
        black_positions = []
        for row in range(grid.size):
            if grid.is_blocked(row, col):
                black_positions.append(row)
        
        segment_starts = [0] + [pos + 1 for pos in black_positions]
        segment_ends = black_positions + [grid.size]
        
        for start, end in zip(segment_starts, segment_ends):
            segment_length = end - start
            if segment_length < min_length:
                continue
            
            constraints = {}
            for pos in range(start, end):
                letter = grid.get_letter(pos, col)
                if letter:
                    constraints[pos - start] = str(letter)
            
            slot = Slot(start, col, Direction.DOWN, segment_length, constraints)
            slots.append(slot)
    
    return slots

def find_intersecting_slots(grid: CrosswordGrid, existing_placements: List[WordPlacement], 
                           min_length: int = 3) -> List[Slot]:
    """Find slots that would create intersections with existing words."""
    if not existing_placements:
        return find_empty_slots(grid, min_length)
    
    all_slots = find_empty_slots(grid, min_length)
    intersecting_slots = []
    
    existing_positions = set()
    for placement in existing_placements:
        existing_positions.update(placement.get_positions())
    
    for slot in all_slots:
        slot_positions = set(slot.get_positions())
        if slot_positions & existing_positions:
            intersecting_slots.append(slot)
    
    return intersecting_slots

# Simulated Annealing Classes
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
    energy: float = 0.0
    
    def __post_init__(self):
        self.energy = -self.fitness_score
    
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

class SAFitnessEvaluator:
    """Enhanced fitness evaluator with multi-objective optimization for crossword generation."""
    
    def __init__(self, preferred_length: int = 6, difficulty_config: Optional['DifficultyConfig'] = None):
        self.preferred_length = preferred_length
        self.difficulty_config = difficulty_config
        
        # Dynamic weights based on difficulty level
        if difficulty_config:
            if difficulty_config.name == "EASY":
                self.weights = {
                    'connectivity': 30.0,
                    'word_count': 15.0,        # Higher priority for word count
                    'intersections': 10.0,     # Moderate intersection priority
                    'fill_efficiency': 5.0,
                    'length_diversity': 2.0,
                    'compactness': 3.0,
                    'target_achievement': 25.0  # Bonus for meeting targets
                }
            elif difficulty_config.name == "MEDIUM":
                self.weights = {
                    'connectivity': 25.0,
                    'word_count': 12.0,
                    'intersections': 15.0,     # Higher intersection priority
                    'fill_efficiency': 8.0,
                    'length_diversity': 3.0,
                    'compactness': 2.0,
                    'target_achievement': 30.0
                }
            else:  # HARD
                self.weights = {
                    'connectivity': 20.0,
                    'word_count': 10.0,
                    'intersections': 20.0,     # Highest intersection priority
                    'fill_efficiency': 15.0,   # Higher fill priority
                    'length_diversity': 5.0,
                    'compactness': 5.0,
                    'target_achievement': 40.0  # Highest bonus for meeting targets
                }
        else:
            # Default weights
            self.weights = {
                'connectivity': 25.0,
                'intersections': 10.0,
                'fill_efficiency': 5.0,
                'word_count': 8.0,
                'length_diversity': 2.0,
                'compactness': 3.0,
                'target_achievement': 20.0
            }
    
    def evaluate_fitness(self, creator: CrosswordCreator) -> float:
        """Calculate comprehensive fitness score with target achievement bonuses."""
        stats = creator.get_puzzle_statistics()
        
        if stats['word_count'] > 1 and not stats['is_connected']:
            return 0.0
        
        # Base component scores
        connectivity_score = self.weights['connectivity'] if stats['is_connected'] else 0.0
        
        # Word count score with exponential rewards for hitting targets
        word_count = stats['word_count']
        if self.difficulty_config:
            target_words = self.difficulty_config.min_words_target
            if word_count >= target_words:
                word_count_score = self.weights['word_count'] * 2.0  # Double reward for meeting target
            else:
                word_count_score = self.weights['word_count'] * (word_count / target_words)
        else:
            word_count_score = self.weights['word_count'] * min(2.0, word_count / 10.0)
        
        # Intersection score with exponential rewards
        intersection_count = stats['intersection_count']
        if self.difficulty_config:
            target_intersections = self.difficulty_config.min_intersections_target
            if intersection_count >= target_intersections:
                intersection_score = self.weights['intersections'] * 2.0  # Double reward
            else:
                intersection_score = self.weights['intersections'] * (intersection_count / target_intersections)
        else:
            intersection_score = self.weights['intersections'] * min(2.0, intersection_count / 20.0)
        
        # Fill efficiency score
        fill_percentage = stats['fill_percentage']
        if self.difficulty_config:
            target_fill = self.difficulty_config.target_fill
            if fill_percentage >= target_fill:
                fill_score = self.weights['fill_efficiency'] * 2.0  # Double reward
            else:
                fill_score = self.weights['fill_efficiency'] * (fill_percentage / target_fill)
        else:
            fill_score = self.weights['fill_efficiency'] * (fill_percentage / 50.0)
        
        # Length diversity score
        if word_count > 1:
            word_lengths = [len(wp.word) for wp in creator.word_placements]
            length_std = np.std(word_lengths) if len(word_lengths) > 1 else 0
            diversity_score = min(self.weights['length_diversity'], length_std * 0.5)
        else:
            diversity_score = 0.0
        
        # Compactness score
        compactness_score = self._calculate_compactness(creator) * self.weights['compactness']
        
        # Target achievement bonus
        target_achievement_score = 0.0
        if self.difficulty_config:
            targets_met = 0
            if word_count >= self.difficulty_config.min_words_target:
                targets_met += 1
            if intersection_count >= self.difficulty_config.min_intersections_target:
                targets_met += 1
            if fill_percentage >= self.difficulty_config.target_fill:
                targets_met += 1
            
            # Exponential bonus for meeting multiple targets
            if targets_met == 3:
                target_achievement_score = self.weights['target_achievement'] * 3.0
            elif targets_met == 2:
                target_achievement_score = self.weights['target_achievement'] * 1.5
            elif targets_met == 1:
                target_achievement_score = self.weights['target_achievement'] * 0.5
        
        total_fitness = (
            connectivity_score +
            word_count_score +
            intersection_score +
            fill_score +
            diversity_score +
            compactness_score +
            target_achievement_score
        )
        
        return total_fitness
    
    def _calculate_compactness(self, creator: CrosswordCreator) -> float:
        """Calculate how compact the word layout is."""
        if len(creator.word_placements) <= 1:
            return 5.0
        
        occupied_positions = set()
        for wp in creator.word_placements:
            occupied_positions.update(wp.get_positions())
        
        if not occupied_positions:
            return 0.0
        
        rows = [pos[0] for pos in occupied_positions]
        cols = [pos[1] for pos in occupied_positions]
        
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        bbox_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        occupied_count = len(occupied_positions)
        
        compactness = occupied_count / bbox_area if bbox_area > 0 else 0
        return compactness * 10.0

class SimulatedAnnealingSolver:
    """Main simulated annealing solver for crossword generation."""
    
    def __init__(self, word_data_manager, preferred_length: int = 6, difficulty_config: Optional['DifficultyConfig'] = None):
        self.word_data_manager = word_data_manager
        self.preferred_length = preferred_length
        self.difficulty_config = difficulty_config
        self.word_index = FastWordIndex(word_data_manager, preferred_length)
        self.fitness_evaluator = SAFitnessEvaluator(preferred_length, difficulty_config)
        
        self.initial_temperature = 100.0
        self.final_temperature = 0.01
        self.cooling_schedule = CoolingSchedule.EXPONENTIAL
        self.cooling_rate = 0.995
        
        # Adaptive perturbation weights based on difficulty
        if difficulty_config:
            if difficulty_config.name == "EASY":
                self.perturbation_weights = {
                    PerturbationType.ADD_WORD: 0.7,      # Focus on adding words
                    PerturbationType.REMOVE_WORD: 0.1,
                    PerturbationType.SWAP_WORD: 0.15,
                    PerturbationType.RELOCATE_WORD: 0.05
                }
            elif difficulty_config.name == "MEDIUM":
                self.perturbation_weights = {
                    PerturbationType.ADD_WORD: 0.5,
                    PerturbationType.REMOVE_WORD: 0.15,
                    PerturbationType.SWAP_WORD: 0.25,    # More swapping for intersections
                    PerturbationType.RELOCATE_WORD: 0.1
                }
            else:  # HARD
                self.perturbation_weights = {
                    PerturbationType.ADD_WORD: 0.4,
                    PerturbationType.REMOVE_WORD: 0.2,
                    PerturbationType.SWAP_WORD: 0.3,     # Highest swapping for complexity
                    PerturbationType.RELOCATE_WORD: 0.1
                }
        else:
            self.perturbation_weights = {
                PerturbationType.ADD_WORD: 0.5,
                PerturbationType.REMOVE_WORD: 0.2,
                PerturbationType.SWAP_WORD: 0.2,
                PerturbationType.RELOCATE_WORD: 0.1
            }
        
        self.current_state: Optional[SAState] = None
        self.best_state: Optional[SAState] = None
        self.accepted_moves = 0
        self.rejected_moves = 0
        
        # Difficulty-specific targets (can be set externally)
        self.min_words_target = 0
        self.min_intersections_target = 0
    
    def solve(self, creator: CrosswordCreator, max_iterations: int = 5000, 
              target_fill: float = 70.0, random_seed: Optional[int] = None) -> bool:
        """Generate crossword using simulated annealing optimization."""
        if random_seed is not None:
            random.seed(random_seed)
        
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
        print(f"Target Fill: {target_fill}%")
        print("")
        
        for iteration in range(max_iterations):
            temperature = self._update_temperature(iteration, max_iterations)
            self.current_state.temperature = temperature
            self.current_state.iteration = iteration
            
            if iteration % 500 == 0 and iteration > 0:
                total_moves = self.accepted_moves + self.rejected_moves
                acceptance_rate = (self.accepted_moves / total_moves * 100) if total_moves > 0 else 0.0
                print(f"Iteration {iteration}: temp={temperature:.3f}, "
                      f"fill={self.current_state.fill_percentage:.1f}%, "
                      f"fitness={self.current_state.fitness_score:.1f}, "
                      f"words={len(self.current_state.word_placements)}, "
                      f"acceptance={acceptance_rate:.1f}%")
            
            # Check for early success based on difficulty-specific criteria
            if (self.current_state.fill_percentage >= target_fill and 
                len(self.current_state.word_placements) >= getattr(self, 'min_words_target', 0)):
                print(f"[SUCCESS] All targets achieved! Fill: {self.current_state.fill_percentage:.1f}%, "
                      f"Words: {len(self.current_state.word_placements)} in {iteration} iterations")
                break
            elif self.current_state.fill_percentage >= target_fill:
                print(f"[SUCCESS] Fill target achieved! Fill: {self.current_state.fill_percentage:.1f}% "
                      f"in {iteration} iterations")
                break
            
            if temperature < self.final_temperature:
                print(f"[TERMINATION] Final temperature reached at iteration {iteration}")
                break
            
            neighbor_creator = self._create_neighbor_state(creator, used_words)
            if neighbor_creator is None:
                continue
            
            # Ensure no duplicates in the neighbor state
            neighbor_creator = self._clean_duplicate_placements(neighbor_creator)
            
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
            
            if self._accept_move(self.current_state, neighbor_state):
                self._apply_state_to_creator(neighbor_state, creator)
                self.current_state = neighbor_state
                self.accepted_moves += 1
                
                used_words.clear()
                used_words.update(wp.word.upper() for wp in creator.word_placements)
                
                if neighbor_fitness > self.best_state.fitness_score:
                    self.best_state = neighbor_state.copy()
                    print(f"New best state at iteration {iteration}: "
                          f"fitness={neighbor_fitness:.1f}, fill={neighbor_fill:.1f}%")
            else:
                self.rejected_moves += 1
        
        if self.best_state.fitness_score > self.current_state.fitness_score:
            self._apply_state_to_creator(self.best_state, creator)
            self.current_state = self.best_state
        
        # Final cleanup to ensure no duplicates remain
        creator = self._clean_duplicate_placements(creator)
        
        # Update final statistics
        final_stats = creator.get_puzzle_statistics()
        print(f"\nFinal cleanup complete:")
        print(f"  Unique words: {final_stats['word_count']}")
        print(f"  Total intersections: {final_stats['intersection_count']}")
        print(f"  Fill percentage: {final_stats['fill_percentage']:.1f}%")
        
        return True
    
    def _update_temperature(self, iteration: int, max_iterations: int) -> float:
        """Update temperature according to cooling schedule."""
        if self.cooling_schedule == CoolingSchedule.LINEAR:
            return self.initial_temperature * (1 - iteration / max_iterations)
        elif self.cooling_schedule == CoolingSchedule.EXPONENTIAL:
            return self.initial_temperature * (self.cooling_rate ** iteration)
        elif self.cooling_schedule == CoolingSchedule.LOGARITHMIC:
            return self.initial_temperature / (1 + math.log(1 + iteration))
        else:
            return self.initial_temperature * (self.cooling_rate ** iteration)
    
    def _create_neighbor_state(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Create a neighbor state by applying a random perturbation."""
        neighbor_creator = self._copy_creator(creator)
        
        # Validate that the copied creator doesn't have duplicates
        neighbor_creator = self._clean_duplicate_placements(neighbor_creator)
        
        # Update used_words to reflect the actual state
        used_words.clear()
        used_words.update(wp.word.upper() for wp in neighbor_creator.word_placements)
        
        perturbation_type = self._choose_perturbation_type(len(neighbor_creator.word_placements))
        
        if perturbation_type == PerturbationType.ADD_WORD:
            return self._add_word_perturbation(neighbor_creator, used_words)
        elif perturbation_type == PerturbationType.REMOVE_WORD:
            return self._remove_word_perturbation(neighbor_creator, used_words)
        elif perturbation_type == PerturbationType.SWAP_WORD:
            return self._swap_word_perturbation(neighbor_creator, used_words)
        elif perturbation_type == PerturbationType.RELOCATE_WORD:
            return self._relocate_word_perturbation(neighbor_creator, used_words)
        
        return None
    
    def _clean_duplicate_placements(self, creator: CrosswordCreator) -> CrosswordCreator:
        """Remove any duplicate word placements from the creator."""
        unique_placements = []
        seen_placements = set()
        
        for placement in creator.word_placements:
            placement_key = (placement.word, placement.row, placement.col, placement.direction)
            if placement_key not in seen_placements:
                unique_placements.append(placement)
                seen_placements.add(placement_key)
            else:
                print(f"Removing duplicate placement: {placement.word} at ({placement.row},{placement.col})")
        
        if len(unique_placements) != len(creator.word_placements):
            print(f"Cleaned {len(creator.word_placements) - len(unique_placements)} duplicate placements")
            
            # Rebuild the creator with only unique placements
            new_grid = CrosswordGrid(creator.grid.size)
            new_creator = CrosswordCreator(new_grid, creator.word_data_manager)
            
            for placement in unique_placements:
                success = new_creator.place_word(placement.word, placement.row, placement.col,
                                               placement.direction, placement.clue)
                if not success:
                    print(f"Warning: Could not re-place {placement.word} during cleanup")
            
            return new_creator
        
        return creator
    
    def _choose_perturbation_type(self, word_count: int) -> PerturbationType:
        """Choose perturbation type based on current state and weights."""
        adjusted_weights = self.perturbation_weights.copy()
        
        if word_count == 0:
            return PerturbationType.ADD_WORD
        elif word_count == 1:
            adjusted_weights[PerturbationType.ADD_WORD] = 0.8
            adjusted_weights[PerturbationType.REMOVE_WORD] = 0.1
            adjusted_weights[PerturbationType.SWAP_WORD] = 0.05
            adjusted_weights[PerturbationType.RELOCATE_WORD] = 0.05
        elif word_count < 5:
            adjusted_weights[PerturbationType.ADD_WORD] = 0.6
            adjusted_weights[PerturbationType.REMOVE_WORD] = 0.15
            adjusted_weights[PerturbationType.SWAP_WORD] = 0.15
            adjusted_weights[PerturbationType.RELOCATE_WORD] = 0.1
        
        types = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        return random.choices(types, weights=weights)[0]
    
    def _add_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Add a new word to the crossword."""
        if len(creator.word_placements) == 0:
            empty_slots = find_empty_slots(creator.grid, min_length=3)
        else:
            empty_slots = find_intersecting_slots(creator.grid, creator.word_placements, min_length=3)
        
        if not empty_slots:
            return None
        
        random.shuffle(empty_slots)
        attempts = 0
        max_attempts = min(50, len(empty_slots) * 10)  # Limit attempts to prevent infinite loops
        
        for slot in empty_slots[:5]:
            if attempts >= max_attempts:
                break
                
            compatible_words = self.word_index.find_compatible_words(slot, max_results=50)
            available_words = [word for word in compatible_words if word.upper() not in used_words]
            
            if available_words:
                word = random.choice(available_words[:10])
                
                # Double-check this word isn't already placed at this position
                placement_exists = any(
                    wp.word == word.upper() and wp.row == slot.row and 
                    wp.col == slot.col and wp.direction == slot.direction 
                    for wp in creator.word_placements
                )
                
                if not placement_exists:
                    success = creator.place_word(word.upper(), slot.row, slot.col, slot.direction)
                    if success:
                        return creator
                        
            attempts += 1
        
        return None
    
    def _remove_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Remove a word from the crossword."""
        if len(creator.word_placements) <= 1:
            return None
        
        word_to_remove = random.choice(creator.word_placements)
        success = creator.remove_word(word_to_remove)
        if success:
            used_words.discard(word_to_remove.word.upper())
            return creator
        
        return None
    
    def _swap_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Swap a word with a different word in the same position."""
        if not creator.word_placements:
            return None
        
        word_to_swap = random.choice(creator.word_placements)
        slot = Slot(word_to_swap.row, word_to_swap.col, word_to_swap.direction, len(word_to_swap.word))
        
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
        
        compatible_words = self.word_index.find_compatible_words(slot, max_results=50)
        available_words = [word for word in compatible_words 
                          if word.upper() not in used_words and word.upper() != word_to_swap.word.upper()]
        
        if available_words:
            creator.remove_word(word_to_swap)
            used_words.discard(word_to_swap.word.upper())
            
            new_word = random.choice(available_words[:10])
            success = creator.place_word(new_word.upper(), slot.row, slot.col, slot.direction)
            
            if success:
                return creator
            else:
                creator.place_word(word_to_swap.word, word_to_swap.row, word_to_swap.col, 
                                 word_to_swap.direction, word_to_swap.clue)
                used_words.add(word_to_swap.word.upper())
        
        return None
    
    def _relocate_word_perturbation(self, creator: CrosswordCreator, used_words: Set[str]) -> Optional[CrosswordCreator]:
        """Relocate a word to a different position."""
        if not creator.word_placements:
            return None
        
        word_to_relocate = random.choice(creator.word_placements)
        original_word = word_to_relocate.word
        original_clue = word_to_relocate.clue
        
        creator.remove_word(word_to_relocate)
        used_words.discard(word_to_relocate.word.upper())
        
        if len(creator.word_placements) == 0:
            available_slots = find_empty_slots(creator.grid, min_length=len(original_word))
        else:
            available_slots = find_intersecting_slots(creator.grid, creator.word_placements, 
                                                    min_length=len(original_word))
        
        suitable_slots = []
        for slot in available_slots:
            if slot.length >= len(original_word):
                temp_slot = Slot(slot.row, slot.col, slot.direction, len(original_word), slot.constraints)
                if temp_slot.matches_word(original_word):
                    suitable_slots.append(temp_slot)
        
        if suitable_slots:
            new_slot = random.choice(suitable_slots)
            success = creator.place_word(original_word, new_slot.row, new_slot.col, 
                                       new_slot.direction, original_clue)
            if success:
                return creator
        
        creator.place_word(original_word, word_to_relocate.row, word_to_relocate.col, 
                         word_to_relocate.direction, original_clue)
        used_words.add(original_word.upper())
        
        return None
    
    def _accept_move(self, current_state: SAState, neighbor_state: SAState) -> bool:
        """Decide whether to accept a move using simulated annealing criteria."""
        if neighbor_state.fitness_score > current_state.fitness_score:
            return True
        
        if current_state.temperature <= 0:
            return False
        
        delta_energy = neighbor_state.energy - current_state.energy
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
        # Clear existing state completely
        creator.grid = CrosswordGrid(creator.grid.size)
        creator.word_placements.clear()
        
        # Restore blocked cells
        for row, col in state.blocked_cells:
            creator.grid.set_blocked(row, col, True)
        
        # Track which words we've successfully placed to avoid duplicates
        placed_words = set()
        successful_placements = []
        
        for placement in state.word_placements:
            # Create a unique key for this placement
            placement_key = (placement.word, placement.row, placement.col, placement.direction)
            
            if placement_key not in placed_words:
                success = creator.place_word(placement.word, placement.row, placement.col, 
                                           placement.direction, placement.clue)
                if success:
                    placed_words.add(placement_key)
                    successful_placements.append(placement)
                else:
                    print(f"Warning: Failed to restore placement for '{placement.word}' at ({placement.row},{placement.col})")
        
        # Ensure the creator's word_placements matches what was actually placed
        if len(creator.word_placements) != len(successful_placements):
            print(f"Warning: State restoration mismatch. Expected {len(successful_placements)}, got {len(creator.word_placements)}")
            
            # Remove any duplicate placements that might have snuck in
            unique_placements = []
            seen_keys = set()
            for wp in creator.word_placements:
                key = (wp.word, wp.row, wp.col, wp.direction)
                if key not in seen_keys:
                    unique_placements.append(wp)
                    seen_keys.add(key)
            creator.word_placements = unique_placements
    
    def get_statistics(self) -> Dict[str, any]:
        """Get solver statistics."""
        total_moves = self.accepted_moves + self.rejected_moves
        acceptance_rate = (self.accepted_moves / total_moves) if total_moves > 0 else 0.0
        
        return {
            'accepted_moves': self.accepted_moves,
            'rejected_moves': self.rejected_moves,
            'total_moves': total_moves,
            'acceptance_rate': acceptance_rate,
            'best_fitness': self.best_state.fitness_score if self.best_state else 0,
            'current_fitness': self.current_state.fitness_score if self.current_state else 0,
            'final_temperature': self.current_state.temperature if self.current_state else 0
        }

@dataclass
class DifficultyConfig:
    """Configuration for different difficulty levels."""
    name: str
    grid_size: int
    target_fill: float
    max_iterations: int
    initial_temperature: float
    cooling_rate: float
    preferred_length: int
    min_words_target: int
    min_intersections_target: int
    
def get_difficulty_configs():
    """Get configurations for all difficulty levels."""
    return {
        'easy': DifficultyConfig(
            name="EASY",
            grid_size=9,
            target_fill=40.0,
            max_iterations=2000,
            initial_temperature=30.0,
            cooling_rate=0.98,
            preferred_length=5,
            min_words_target=6,
            min_intersections_target=8
        ),
        'medium': DifficultyConfig(
            name="MEDIUM", 
            grid_size=13,
            target_fill=55.0,
            max_iterations=4000,
            initial_temperature=60.0,
            cooling_rate=0.99,
            preferred_length=6,
            min_words_target=12,
            min_intersections_target=20
        ),
        'hard': DifficultyConfig(
            name="HARD",
            grid_size=17,
            target_fill=70.0,
            max_iterations=6000,
            initial_temperature=100.0,
            cooling_rate=0.995,
            preferred_length=7,
            min_words_target=20,
            min_intersections_target=35
        )
    }

def generate_crossword_by_difficulty(word_data_manager, config: DifficultyConfig, random_seed=None):
    """Generate a crossword puzzle for a specific difficulty level."""
    print("="*70)
    print(f"GENERATING {config.name} CROSSWORD PUZZLE")
    print("="*70)
    print(f"Grid Size: {config.grid_size}x{config.grid_size}")
    print(f"Target Fill: {config.target_fill}%")
    print(f"Target Words: {config.min_words_target}+")
    print(f"Target Intersections: {config.min_intersections_target}+")
    print(f"Max Iterations: {config.max_iterations}")
    print()
    
    # Initialize grid and creator
    grid = CrosswordGrid(config.grid_size)
    creator = CrosswordCreator(grid, word_data_manager)
    
    # Create and configure SA solver based on difficulty
    sa_solver = SimulatedAnnealingSolver(word_data_manager, preferred_length=config.preferred_length)
    sa_solver.initial_temperature = config.initial_temperature
    sa_solver.cooling_rate = config.cooling_rate
    
    # Set difficulty-specific targets for the solver
    sa_solver.min_words_target = config.min_words_target
    sa_solver.min_intersections_target = config.min_intersections_target
    
    # Adjust perturbation weights based on difficulty
    if config.name == "EASY":
        # Easy mode: Focus more on adding words, less on complex operations
        sa_solver.perturbation_weights = {
            PerturbationType.ADD_WORD: 0.6,
            PerturbationType.REMOVE_WORD: 0.15,
            PerturbationType.SWAP_WORD: 0.15,
            PerturbationType.RELOCATE_WORD: 0.1
        }
    elif config.name == "HARD":
        # Hard mode: More aggressive optimization
        sa_solver.perturbation_weights = {
            PerturbationType.ADD_WORD: 0.4,
            PerturbationType.REMOVE_WORD: 0.25,
            PerturbationType.SWAP_WORD: 0.25,
            PerturbationType.RELOCATE_WORD: 0.1
        }
    
    # Run the solver
    start_time = time.time() if 'time' in globals() else None
    
    success = sa_solver.solve(
        creator=creator,
        max_iterations=config.max_iterations,
        target_fill=config.target_fill,
        random_seed=random_seed
    )
    
    end_time = time.time() if 'time' in globals() else None
    
    # Get results
    puzzle_stats = creator.get_puzzle_statistics()
    sa_stats = sa_solver.get_statistics()
    
    # Display results
    print("\n" + "="*70)
    print(f"{config.name} CROSSWORD COMPLETE!")
    print("="*70)
    
    print(f"\nPuzzle Statistics:")
    print(f"  Grid Size: {config.grid_size}x{config.grid_size}")
    print(f"  Fill Percentage: {puzzle_stats['fill_percentage']:.1f}%")
    print(f"  Words Placed: {puzzle_stats['word_count']}")
    print(f"  Intersections: {puzzle_stats['intersection_count']}")
    print(f"  Connected: {puzzle_stats['is_connected']}")
    print(f"  Blocked Cells: {puzzle_stats['blocked_cells']}")
    
    # Calculate density metrics
    total_cells = config.grid_size * config.grid_size
    word_density = puzzle_stats['word_count'] / total_cells * 100
    intersection_density = puzzle_stats['intersection_count'] / puzzle_stats['word_count'] if puzzle_stats['word_count'] > 0 else 0
    
    print(f"  Word Density: {word_density:.1f} words per 100 cells")
    print(f"  Intersection Density: {intersection_density:.1f} intersections per word")
    
    print(f"\nSolver Performance:")
    print(f"  Iterations Used: {sa_stats['total_moves']}")
    print(f"  Accepted Moves: {sa_stats['accepted_moves']}")
    print(f"  Acceptance Rate: {sa_stats['acceptance_rate']:.1f}%")
    print(f"  Final Fitness: {sa_stats['current_fitness']:.1f}")
    if start_time and end_time:
        print(f"  Generation Time: {end_time - start_time:.1f} seconds")
    
    # Show difficulty achievement
    words_achieved = puzzle_stats['word_count'] >= config.min_words_target
    intersections_achieved = puzzle_stats['intersection_count'] >= config.min_intersections_target
    fill_achieved = puzzle_stats['fill_percentage'] >= config.target_fill
    
    print(f"\nDifficulty Targets:")
    print(f"  Words Target ({config.min_words_target}+): {'✓ ACHIEVED' if words_achieved else '✗ NOT MET'}")
    print(f"  Intersections Target ({config.min_intersections_target}+): {'✓ ACHIEVED' if intersections_achieved else '✗ NOT MET'}")
    print(f"  Fill Target ({config.target_fill}%+): {'✓ ACHIEVED' if fill_achieved else '✗ NOT MET'}")
    
    difficulty_score = sum([words_achieved, intersections_achieved, fill_achieved])
    if difficulty_score == 3:
        print(f"  🎉 PERFECT {config.name} PUZZLE! All targets achieved!")
    elif difficulty_score >= 2:
        print(f"  👍 GOOD {config.name} PUZZLE! Most targets achieved!")
    else:
        print(f"  ⚠️  {config.name} puzzle partially complete.")
    
    # Display the crossword
    print(f"\n{config.name} Crossword Grid:")
    print(creator.grid)
    
    # Show word list with clues
    print(f"\nWords in {config.name} Puzzle ({len(creator.word_placements)} total):")
    for i, wp in enumerate(creator.word_placements, 1):
        direction = "Across" if wp.direction == Direction.ACROSS else "Down"
        clue = wp.clue if wp.clue else f"Clue for {wp.word}"
        # Truncate long clues for readability
        if len(clue) > 60:
            clue = clue[:57] + "..."
        print(f"  {i:2d}. {wp.word:12s} ({direction:6s}) at ({wp.row:2d},{wp.col:2d}) - {clue}")
    
    return creator, sa_solver, config

def run_demo():
    """Run the multi-difficulty crossword generation demo."""
    print("="*70)
    print("MULTI-DIFFICULTY SIMULATED ANNEALING CROSSWORD GENERATOR")
    print("="*70)
    
    # Find the CSV file - look in current directory and parent directory
    csv_file = None
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations for the CSV file
    possible_paths = [
        "clues_bigdave.csv",  # Current directory
        os.path.join(current_dir, "clues_bigdave.csv"),  # Same directory as script
        os.path.join(os.path.dirname(current_dir), "clues_bigdave.csv"),  # Parent directory
        os.path.join(current_dir, "..", "clues_bigdave.csv"),  # Parent directory (alternative)
    ]
    
    print(f"Looking for CSV file in these locations:")
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        print(f"  {abs_path} - {'Found' if exists else 'Not found'}")
        if exists and csv_file is None:
            csv_file = abs_path
    
    if csv_file is None:
        print("Error: Could not find clues_bigdave.csv in any expected location!")
        return None, None
    
    print(f"\nUsing CSV file: {csv_file}")
    
    # Initialize word data manager with your CSV
    print("Loading word data...")
    word_data_manager = WordDataManagerWrapper(csv_file)
    
    # Check if data loaded successfully
    if not word_data_manager.load_data():
        print("Failed to load word data! Please ensure clues_bigdave.csv exists in the current directory.")
        return None, None
    
    # Show word data statistics
    stats = word_data_manager.get_statistics()
    print(f"Loaded {stats['unique_words']} unique words ({stats['total_entries']} total entries)")
    print(f"Word lengths: {stats['min_word_length']}-{stats['max_word_length']} characters")
    print(f"Average word length: {stats['avg_word_length']:.1f}")
    
    # Get difficulty configurations
    configs = get_difficulty_configs()
    
    # Ask user which difficulty to run or run all
    print(f"\nSelect difficulty level:")
    print("1. Easy (9x9, ~6 words, light complexity)")
    print("2. Medium (13x13, ~12 words, moderate complexity)")  
    print("3. Hard (17x17, ~20 words, high complexity)")
    print("4. All difficulties (run Easy → Medium → Hard)")
    
    choice = input("\nEnter your choice (1-4) [default: 4]: ").strip()
    if not choice:
        choice = "4"
    
    results = []
    
    if choice == "1":
        creator, solver, config = generate_crossword_by_difficulty(word_data_manager, configs['easy'], random_seed=42)
        results.append((creator, solver, config))
    elif choice == "2":
        creator, solver, config = generate_crossword_by_difficulty(word_data_manager, configs['medium'], random_seed=42)
        results.append((creator, solver, config))
    elif choice == "3":
        creator, solver, config = generate_crossword_by_difficulty(word_data_manager, configs['hard'], random_seed=42)
        results.append((creator, solver, config))
    else:  # choice == "4" or invalid
        print("\n🎯 Running all difficulty levels...\n")
        for difficulty in ['easy', 'medium', 'hard']:
            creator, solver, config = generate_crossword_by_difficulty(
                word_data_manager, configs[difficulty], random_seed=42
            )
            results.append((creator, solver, config))
            if difficulty != 'hard':  # Don't print separator after last one
                print("\n" + "⬇️ " * 35)
                input("Press Enter to continue to next difficulty level...")
                print()
    
    # Final summary if multiple difficulties were run
    if len(results) > 1:
        print("\n" + "="*70)
        print("DIFFICULTY PROGRESSION SUMMARY")
        print("="*70)
        
        for creator, solver, config in results:
            stats = creator.get_puzzle_statistics()
            sa_stats = solver.get_statistics()
            
            print(f"\n{config.name}:")
            print(f"  Grid: {config.grid_size}x{config.grid_size}")
            print(f"  Words: {stats['word_count']} (target: {config.min_words_target}+)")
            print(f"  Intersections: {stats['intersection_count']} (target: {config.min_intersections_target}+)")
            print(f"  Fill: {stats['fill_percentage']:.1f}% (target: {config.target_fill}%+)")
            print(f"  Fitness: {sa_stats['best_fitness']:.1f}")
            
        print(f"\n🎉 Difficulty progression complete! Each level increases complexity:")
        print("   • Grid size grows (9x9 → 13x13 → 17x17)")
        print("   • Word count increases (~6 → ~12 → ~20+)")
        print("   • Intersection density improves")
        print("   • Fill percentage targets rise (40% → 55% → 70%)")
    
    print(f"\nDemo completed!")
    return results

if __name__ == "__main__":
    # Run the multi-difficulty demo
    results = run_demo()
    
    if results:
        print("\n" + "="*70)
        print("How to use this multi-difficulty crossword generator:")
        print("="*70)
        print("1. Save this code as 'sa_crossword_demo.py'")
        print("2. Ensure 'word_data.py' and 'clues_bigdave.csv' are in the same directory")
        print("3. Run: python sa_crossword_demo.py")
        print("4. Choose your difficulty level or run all levels")
        print()
        print("Difficulty Level Characteristics:")
        print("📊 EASY:   9x9 grid, ~6 words, 40% fill, gentle complexity")
        print("📊 MEDIUM: 13x13 grid, ~12 words, 55% fill, moderate complexity")  
        print("📊 HARD:   17x17 grid, ~20+ words, 70% fill, high complexity")
        print()
        print("Each level automatically adjusts:")
        print("• Grid size and target fill percentage")
        print("• Number of iterations and temperature settings")
        print("• Preferred word lengths and intersection targets")
        print("• Perturbation strategies for optimization")
        print("="*70)
    else:
        print("\nDemo failed to complete. Please check file requirements.")
        print("Required files:")
        print("- word_data.py (your word data manager)")
        print("- clues_bigdave.csv (your word/clue database)")
        print("- Both files should be in the same directory as this script")