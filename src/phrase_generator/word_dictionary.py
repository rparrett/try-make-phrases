"""
Word dictionary system for providing high-value word inspiration.
Uses google-10000-english.txt to find scorable words for phrase generation.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import random
from loguru import logger

from storage.models import TileInventory
from src.phrase_generator.scorer import ScrabbleScorer
from src.phrase_generator.phrase_validator import PhraseValidator


class WordDictionary:
    """Manages a scored word dictionary for phrase inspiration."""

    def __init__(self, dictionary_path: str = "google-10000-english.txt"):
        """
        Initialize the word dictionary.

        Args:
            dictionary_path: Path to the word list file
        """
        self.dictionary_path = Path(dictionary_path)
        self.scorer = ScrabbleScorer()
        self.validator = PhraseValidator()

        # Cache for scored words
        self.scored_words: List[Tuple[str, int]] = []
        self.words_by_score: Dict[int, List[str]] = {}

        self._load_and_score_dictionary()

    def _load_and_score_dictionary(self):
        """Load dictionary file and score all words."""
        try:
            if not self.dictionary_path.exists():
                logger.error(f"Dictionary file not found: {self.dictionary_path}")
                return

            logger.info(f"Loading dictionary from {self.dictionary_path}")

            with open(self.dictionary_path, "r", encoding="utf-8") as f:
                words = [line.strip().upper() for line in f if line.strip()]

            logger.info(f"Loaded {len(words)} words, scoring them...")

            # Score each word
            for word in words:
                if word and len(word) >= 2:  # Skip single letters
                    # Create a dummy tile dict for scoring (we have all letters)
                    dummy_tiles = {letter: 10 for letter in word}
                    score = self.scorer.score_phrase_simple(word, dummy_tiles)

                    if score > 0:  # Only keep scoreable words
                        self.scored_words.append((word, score))

            # Sort by score (highest first)
            self.scored_words.sort(key=lambda x: x[1], reverse=True)

            # Group by score for quick lookup
            for word, score in self.scored_words:
                if score not in self.words_by_score:
                    self.words_by_score[score] = []
                self.words_by_score[score].append(word)

            logger.info(
                f"Scored {len(self.scored_words)} words. "
                f"Top score: {self.scored_words[0][1] if self.scored_words else 0}"
            )

            # Debug: Show some examples
            if self.scored_words:
                logger.debug(f"Top 10 words by score: {self.scored_words[:10]}")
                logger.debug(
                    f"Score distribution sample: scores 20+: {len([w for w, s in self.scored_words if s >= 20])}, "
                    f"15+: {len([w for w, s in self.scored_words if s >= 15])}, "
                    f"10+: {len([w for w, s in self.scored_words if s >= 10])}, "
                    f"5+: {len([w for w, s in self.scored_words if s >= 5])}"
                )

        except Exception as e:
            logger.error(f"Failed to load dictionary: {e}")
            self.scored_words = []

    def get_buildable_words(
        self, tiles: TileInventory, min_score: int = 0, max_words: int = 100
    ) -> List[Tuple[str, int]]:
        """
        Get words that can be built with available tiles.

        Args:
            tiles: Available tiles
            min_score: Minimum word score to include
            max_words: Maximum number of words to return

        Returns:
            List of (word, score) tuples, sorted by score descending
        """
        buildable_words = []
        checked_count = 0
        skipped_low_score = 0

        for word, score in self.scored_words:
            checked_count += 1

            if score < min_score:
                skipped_low_score += 1
                continue

            # Check if word can be built with available tiles
            if tiles.can_construct_phrase(word):
                buildable_words.append((word, score))

                if len(buildable_words) >= max_words:
                    break

        logger.debug(
            f"Buildable words search: checked {checked_count} words, "
            f"skipped {skipped_low_score} low-scoring, "
            f"found {len(buildable_words)} buildable"
        )

        return buildable_words

    def get_inspiration_words(
        self, tiles: TileInventory, count: int = 5, min_score: int = 5
    ) -> List[str]:
        """
        Get random high-scoring words for LLM inspiration.

        Args:
            tiles: Available tiles
            count: Number of inspiration words to return
            min_score: Minimum score for inspiration words

        Returns:
            List of high-scoring words that can be built
        """
        logger.debug(f"Getting {count} inspiration words with min_score={min_score}")
        logger.debug(f"Available tiles for inspiration: {tiles.tiles}")

        buildable = self.get_buildable_words(tiles, min_score=min_score, max_words=4000)

        logger.debug(f"Found {len(buildable)} buildable words (min_score={min_score})")

        if not buildable:
            logger.debug("No buildable inspiration words found")
            return []

        if buildable:
            logger.debug(f"Buildable words sample: {buildable[:20]}")  # Show first 20

        # Prefer higher-scoring words but add some randomness
        # Take top 50% and randomly sample from those
        top_half_count = max(1, len(buildable) // 2)
        top_words = buildable[:top_half_count]

        logger.debug(
            f"Taking top {top_half_count} words from {len(buildable)} buildable words"
        )

        # Randomly sample from the top words
        sample_count = min(count, len(top_words))

        if len(top_words) < sample_count:
            logger.debug(
                f"Warning: Only {len(top_words)} top words available, requested {sample_count}"
            )

        sampled = random.sample(top_words, sample_count)

        words = [word for word, score in sampled]
        logger.debug(f"Selected inspiration words: {words}")

        return words

    def get_leftover_inspiration_words(
        self, base_phrase: str, tiles: TileInventory, count: int = 3, min_score: int = 4
    ) -> List[str]:
        """
        Get inspiration words that can be built from leftover tiles after making base phrase.

        Args:
            base_phrase: The phrase being improved
            tiles: Available tiles
            count: Number of inspiration words to return
            min_score: Minimum score for inspiration words

        Returns:
            List of words buildable from leftover tiles
        """
        logger.debug(f"Getting leftover inspiration for phrase: '{base_phrase}'")
        logger.debug(f"Original tiles: {tiles.tiles}")

        # Calculate leftover tiles after making the base phrase
        is_valid, tiles_used, error = self.validator.validate_phrase(base_phrase, tiles)

        if not is_valid:
            logger.debug(f"Base phrase '{base_phrase}' is not valid: {error}")
            return []

        logger.debug(f"Base phrase tiles used: {tiles_used}")

        # Create leftover tile inventory
        leftover_tiles_dict = tiles.tiles.copy()
        for letter, used_count in tiles_used.items():
            if letter in leftover_tiles_dict:
                original_count = leftover_tiles_dict[letter]
                leftover_tiles_dict[letter] = max(0, original_count - used_count)
                logger.debug(
                    f"Letter {letter}: {original_count} - {used_count} = {leftover_tiles_dict[letter]}"
                )

        # Remove tiles with 0 count
        leftover_tiles_dict = {k: v for k, v in leftover_tiles_dict.items() if v > 0}

        logger.debug(f"Leftover tiles after base phrase: {leftover_tiles_dict}")
        logger.debug(f"Total leftover tiles: {sum(leftover_tiles_dict.values())}")

        if not leftover_tiles_dict:
            logger.debug("No leftover tiles for inspiration words")
            return []

        # Create temporary tile inventory for leftover tiles
        leftover_tiles = TileInventory(tiles=leftover_tiles_dict)

        # Get buildable words from leftovers (with detailed logging)
        buildable_words = self.get_buildable_words(
            leftover_tiles, min_score=min_score, max_words=4000
        )
        logger.debug(
            f"Found {len(buildable_words)} buildable words from leftovers (min_score={min_score})"
        )

        if buildable_words:
            logger.debug(f"Top 10 buildable leftover words: {buildable_words[:10]}")

        # Get inspiration words from leftovers
        inspiration = self.get_inspiration_words(
            leftover_tiles, count=count, min_score=min_score
        )

        logger.debug(f"Final leftover inspiration words: {inspiration}")

        return inspiration

    def get_stats(self) -> Dict[str, any]:
        """Get dictionary statistics."""
        if not self.scored_words:
            return {"total_words": 0, "error": "Dictionary not loaded"}

        scores = [score for _, score in self.scored_words]

        return {
            "total_words": len(self.scored_words),
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "top_10_words": self.scored_words[:10],
        }


# Singleton instance for easy access
_word_dictionary = None


def get_word_dictionary() -> WordDictionary:
    """Get the global word dictionary instance."""
    global _word_dictionary
    if _word_dictionary is None:
        _word_dictionary = WordDictionary()
    return _word_dictionary


# Testing
if __name__ == "__main__":
    from src.phrase_generator.tile_parser import parse_tile_string

    # Test the dictionary system
    dictionary = WordDictionary()
    stats = dictionary.get_stats()
    print(f"Dictionary stats: {stats}")

    # Test with sample tiles
    test_tiles = parse_tile_string("9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh")
    print(f"Available tiles: {test_tiles.tiles}")

    # Get inspiration words
    inspiration = dictionary.get_inspiration_words(test_tiles, count=8)
    print(f"Inspiration words: {inspiration}")

    # Test leftover inspiration
    base_phrase = "WINTER MORNING"
    leftover_inspiration = dictionary.get_leftover_inspiration_words(
        base_phrase, test_tiles, count=5
    )
    print(f"Leftover inspiration for '{base_phrase}': {leftover_inspiration}")
