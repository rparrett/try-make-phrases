"""
Phrase validator to check if phrases can be constructed from available tiles.
"""

from typing import Dict, Tuple, List, Optional
from collections import Counter
import logging

from storage.models import TileInventory
from config.scrabble_values import is_free_character


class ValidationError(Exception):
    """Exception raised for validation errors."""

    pass


class PhraseValidator:
    """Validates that phrases can be constructed from available tiles."""

    def __init__(self):
        """Initialize the validator."""
        self.logger = logging.getLogger(__name__)

    def validate_phrase(
        self, phrase: str, tiles: TileInventory
    ) -> Tuple[bool, Dict[str, int], str]:
        """
        Validate that a phrase can be constructed from available tiles.

        Args:
            phrase: The phrase to validate
            tiles: Available tiles

        Returns:
            Tuple of (is_valid, tiles_used, error_message)
            - is_valid: True if phrase can be constructed
            - tiles_used: Dictionary mapping letters to quantities used
            - error_message: Description of validation error (empty if valid)
        """
        if not phrase or not phrase.strip():
            return False, {}, "Empty phrase"

        try:
            # Normalize phrase
            phrase = phrase.strip().upper()

            # Count required letters (excluding free characters)
            required_letters = Counter()
            for char in phrase:
                if is_free_character(char):
                    continue  # Skip spaces, punctuation, etc.
                required_letters[char] += 1

            # Check if we can construct the phrase
            can_construct, tiles_used = self._can_construct_with_blanks(
                required_letters, tiles
            )

            if can_construct:
                return True, tiles_used, ""
            else:
                missing = self._find_missing_letters(required_letters, tiles)
                return False, {}, f"Missing letters: {missing}"

        except Exception as e:
            self.logger.error(f"Validation error for phrase '{phrase}': {e}")
            return False, {}, f"Validation error: {e}"

    def _can_construct_with_blanks(
        self, required_letters: Counter, tiles: TileInventory
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if phrase can be constructed, optimally using blank tiles.

        This function tries to use regular tiles first, then blank tiles for remaining letters.

        Args:
            required_letters: Counter of letters needed
            tiles: Available tiles

        Returns:
            Tuple of (can_construct, tiles_used)
        """
        tiles_used = {}
        remaining_tiles = tiles.tiles.copy()

        # First pass: use regular tiles
        blanks_needed = 0
        for letter, count in required_letters.items():
            available = remaining_tiles.get(letter, 0)
            used_regular = min(count, available)

            if used_regular > 0:
                tiles_used[letter] = used_regular
                remaining_tiles[letter] = available - used_regular

            # Track how many blanks we need for this letter
            still_needed = count - used_regular
            if still_needed > 0:
                blanks_needed += still_needed

        # Check if we have enough blank tiles
        available_blanks = remaining_tiles.get("_", 0)
        if blanks_needed > available_blanks:
            return False, {}

        # Second pass: allocate blank tiles for remaining letters
        if blanks_needed > 0:
            tiles_used["_"] = blanks_needed

        return True, tiles_used

    def _find_missing_letters(
        self, required_letters: Counter, tiles: TileInventory
    ) -> str:
        """Find which letters are missing for constructing the phrase."""
        missing = []
        available_blanks = tiles.tiles.get("_", 0)
        total_shortage = 0

        for letter, needed in required_letters.items():
            available = tiles.tiles.get(letter, 0)
            shortage = max(0, needed - available)
            if shortage > 0:
                missing.append(f"{letter}×{shortage}")
                total_shortage += shortage

        # Account for blank tiles that could fill the gap
        if total_shortage > available_blanks:
            actual_shortage = total_shortage - available_blanks
            missing.append(f"(need {actual_shortage} more tiles/blanks)")

        return ", ".join(missing) if missing else "None"

    def batch_validate_phrases(
        self, phrases: List[str], tiles: TileInventory
    ) -> List[Tuple[str, bool, Dict[str, int], str]]:
        """
        Validate multiple phrases at once.

        Args:
            phrases: List of phrases to validate
            tiles: Available tiles

        Returns:
            List of tuples: (phrase, is_valid, tiles_used, error_message)
        """
        results = []

        for phrase in phrases:
            is_valid, tiles_used, error_msg = self.validate_phrase(phrase, tiles)
            results.append((phrase, is_valid, tiles_used, error_msg))

        return results

    def find_optimal_tile_usage(
        self, phrase: str, tiles: TileInventory
    ) -> Optional[Dict[str, int]]:
        """
        Find the optimal way to use tiles for a phrase (minimizing blank usage).

        Args:
            phrase: Phrase to construct
            tiles: Available tiles

        Returns:
            Dictionary of optimal tile usage, or None if impossible
        """
        is_valid, tiles_used, _ = self.validate_phrase(phrase, tiles)
        if is_valid:
            return tiles_used
        return None

    def get_phrase_difficulty(self, phrase: str, tiles: TileInventory) -> float:
        """
        Calculate a difficulty score for constructing a phrase (0.0 = easy, 1.0 = impossible).

        Args:
            phrase: Phrase to evaluate
            tiles: Available tiles

        Returns:
            Difficulty score (0.0 to 1.0)
        """
        try:
            # Normalize phrase
            phrase = phrase.strip().upper()

            # Count required letters
            required_letters = Counter()
            for char in phrase:
                if is_free_character(char):
                    continue
                required_letters[char] += 1

            total_letters_needed = sum(required_letters.values())
            if total_letters_needed == 0:
                return 0.0  # No letters needed

            # Calculate how many letters we need to use blanks for
            blanks_needed = 0
            for letter, count in required_letters.items():
                available = tiles.tiles.get(letter, 0)
                blanks_needed += max(0, count - available)

            # Difficulty increases with blank usage
            if blanks_needed > tiles.tiles.get("_", 0):
                return 1.0  # Impossible

            # Score based on proportion of blanks needed
            difficulty = blanks_needed / total_letters_needed
            return min(1.0, difficulty)

        except Exception:
            return 1.0  # Error = maximum difficulty


# Testing and examples
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from src.phrase_generator.tile_parser import parse_tile_string

    # Test with sample tiles
    test_tiles = parse_tile_string("9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh")
    print(f"Available tiles: {test_tiles.tiles}")

    validator = PhraseValidator()

    # Test phrases
    test_phrases = [
        "WINTER MORNING",
        "SNOW DAY",
        "HOLIDAY CHEER",
        "IMPOSSIBLE PHRASE WITH MISSING LETTERS",
        "COLD NIGHT",
        "WINTER WONDERLAND",  # This should be challenging
    ]

    print("\nTesting phrases:")
    for phrase in test_phrases:
        is_valid, tiles_used, error = validator.validate_phrase(phrase, test_tiles)
        difficulty = validator.get_phrase_difficulty(phrase, test_tiles)

        print(f"  '{phrase}':")
        print(f"    Valid: {is_valid}")
        print(f"    Tiles used: {tiles_used}")
        print(f"    Difficulty: {difficulty:.2f}")
        if not is_valid:
            print(f"    Error: {error}")
        print()
