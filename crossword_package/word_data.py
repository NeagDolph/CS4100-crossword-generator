"""
Word Data Module

Handles loading and managing word-clue pairs from CSV files.
Provides functionality to filter and select words for crossword creation.
"""

import csv
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os

@dataclass
class WordClue:
    """
    Represents a word-clue pair.
    
    Attributes:
        word: The crossword word (uppercase)
        clue: The associated clue text
        date: Optional date from source (for NYT crosswords)
    """
    word: str
    clue: str
    date: Optional[str] = None
    
    def __post_init__(self):
        """Ensure word is uppercase for consistency."""
        self.word = self.word.upper()

class WordDataManager:
    """
    Manages word-clue data loading and filtering.
    """
    
    def __init__(self, csv_file_path: str = "clues_bigdave.csv"):
        """
        Initialize the word data manager.
        
        Args:
            csv_file_path: Path to the CSV file containing word-clue data
        """
        self.csv_file_path = csv_file_path
        self.word_clues: List[WordClue] = []
        self.word_to_clues: Dict[str, List[WordClue]] = {}
        self._loaded = False
    
    def load_data(self) -> bool:
        """
        Load word-clue data from CSV file.
        
        Returns:
            True if data loaded successfully, False otherwise
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
                        
                        for row in reader:
                            # Skip rows with missing data
                            if not row.get('answer') or not row.get('clue'):
                                continue
                                
                            word = row['answer'].strip()
                            # Skip words containing dashes or spaces
                            if '-' in word or ' ' in word:
                                continue
                            
                            word_clue = WordClue(
                                word=word,
                                clue=row['clue'].strip(),
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
    
    def ensure_loaded(self):
        """Ensure data is loaded, load if not already loaded."""
        if not self._loaded:
            self.load_data()
    
    def get_all_words(self) -> List[str]:
        """
        Get all unique words available.
        
        Returns:
            List of unique words (uppercase)
        """
        self.ensure_loaded()
        return list(self.word_to_clues.keys())
    
    def get_words_by_length(self, min_length: int = 3, max_length: int = 15) -> List[str]:
        """
        Get words filtered by length.
        
        Args:
            min_length: Minimum word length
            max_length: Maximum word length
            
        Returns:
            List of words within the specified length range
        """
        self.ensure_loaded()
        return [word for word in self.word_to_clues.keys() 
                if min_length <= len(word) <= max_length]
    
    def get_random_words(self, count: int = 50, min_length: int = 3, max_length: int = 15) -> List[str]:
        """
        Get a random selection of words.
        
        Args:
            count: Number of words to return
            min_length: Minimum word length
            max_length: Maximum word length
            
        Returns:
            List of randomly selected words
        """
        available_words = self.get_words_by_length(min_length, max_length)
        return random.sample(available_words, min(count, len(available_words)))
    
    def get_clue_for_word(self, word: str) -> Optional[str]:
        """
        Get a clue for a specific word.
        
        Args:
            word: The word to get a clue for
            
        Returns:
            A clue for the word, or None if word not found
        """
        self.ensure_loaded()
        word = word.upper()
        
        if word in self.word_to_clues:
            # Return a random clue if multiple exist
            clue_entry = random.choice(self.word_to_clues[word])
            return clue_entry.clue
        
        return None
    
    def get_all_clues_for_word(self, word: str) -> List[str]:
        """
        Get all available clues for a specific word.
        
        Args:
            word: The word to get clues for
            
        Returns:
            List of all clues for the word
        """
        self.ensure_loaded()
        word = word.upper()
        
        if word in self.word_to_clues:
            return [wc.clue for wc in self.word_to_clues[word]]
        
        return []
    
    def search_words_by_pattern(self, pattern: str) -> List[str]:
        """
        Search for words containing a pattern.
        
        Args:
            pattern: Pattern to search for
        """
        
    
    def get_word_clue_pairs(self, words: List[str]) -> List[Tuple[str, str]]:
        """
        Get word-clue pairs for a list of words.
        
        Args:
            words: List of words to get clues for
            
        Returns:
            List of (word, clue) tuples
        """
        self.ensure_loaded()
        pairs = []
        
        for word in words:
            clue = self.get_clue_for_word(word)
            if clue:
                pairs.append((word.upper(), clue))
            else:
                # Fallback to default clue if no clue found
                pairs.append((word.upper(), f"Word: {word}"))
        
        return pairs
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about the loaded word data.
        
        Returns:
            Dictionary with statistics
        """
        self.ensure_loaded()
        
        word_lengths = [len(word) for word in self.word_to_clues.keys()]
        
        return {
            'total_entries': len(self.word_clues),
            'unique_words': len(self.word_to_clues),
            'min_word_length': min(word_lengths) if word_lengths else 0,
            'max_word_length': max(word_lengths) if word_lengths else 0,
            'avg_word_length': sum(word_lengths) / len(word_lengths) if word_lengths else 0
        }

word_data_manager = WordDataManager() 