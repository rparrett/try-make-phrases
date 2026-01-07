"""
Phrase ranking and optimization system with SQLite persistence.
"""

from loguru import logger
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from storage.models import GeneratedPhrase, TileInventory, OptimizationConfig
from storage.database import PhraseDatabase, DatabaseError
from src.phrase_generator.scorer import ScrabbleScorer
from src.phrase_generator.phrase_validator import PhraseValidator


class RankingError(Exception):
    """Exception raised for ranking operations."""
    pass


class PhraseRanker:
    """Manages phrase ranking and optimization with database persistence."""

    def __init__(self, db_path: str = "data/phrases.db", config: Optional[OptimizationConfig] = None):
        """
        Initialize the phrase ranker.

        Args:
            db_path: Path to SQLite database
            config: Optimization configuration
        """
        self.db = PhraseDatabase(db_path)
        self.scorer = ScrabbleScorer()
        self.validator = PhraseValidator()
        self.config = config or OptimizationConfig()
        self.logger = logger

    def add_phrase_candidates(self, phrase_candidates: List[str], tiles: TileInventory,
                            model_used: str = "llama2:7b", prompt_context: Optional[str] = None) -> List[GeneratedPhrase]:
        """
        Validate, score, and add phrase candidates to the ranking.

        Args:
            phrase_candidates: List of phrase strings from LLM
            tiles: Available tiles for validation
            model_used: Model that generated the phrases
            prompt_context: Context used in generation

        Returns:
            List of successfully added GeneratedPhrase objects
        """
        added_phrases = []

        try:
            # Validate and score all candidates
            valid_phrases = []
            for phrase in phrase_candidates:
                is_valid, tiles_used, error = self.validator.validate_phrase(phrase, tiles)

                if is_valid:
                    # Check minimum score threshold
                    score = self.scorer.score_phrase_simple(phrase, tiles_used)
                    if score >= self.config.min_score_threshold:
                        generated_phrase = self.scorer.create_scored_phrase(
                            phrase, tiles, tiles_used, model_used, prompt_context
                        )
                        valid_phrases.append(generated_phrase)
                    else:
                        self.logger.debug(f"Phrase '{phrase}' scored {score}, below threshold {self.config.min_score_threshold}")
                else:
                    self.logger.debug(f"Invalid phrase '{phrase}': {error}")

            if not valid_phrases:
                self.logger.debug("No valid phrases from candidates")
                return []

            # Sort by score before adding to database
            valid_phrases.sort(key=lambda p: p.score, reverse=True)

            # Add to database
            phrase_ids = self.db.add_phrases_batch(valid_phrases)
            self.logger.info(f"Added {len(phrase_ids)} new phrases to database")

            # Update phrase objects with IDs
            for phrase, phrase_id in zip(valid_phrases, phrase_ids):
                if phrase_id:  # Some might be None due to duplicates
                    phrase.id = phrase_id
                    added_phrases.append(phrase)

            # Check if database cleanup is needed
            if self._should_cleanup():
                self._cleanup_low_scoring_phrases()

            return added_phrases

        except DatabaseError as e:
            self.logger.error(f"Database error adding phrases: {e}")
            raise RankingError(f"Failed to add phrases: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error adding phrases: {e}")
            raise RankingError(f"Unexpected error: {e}")

    def get_top_phrases(self, limit: int = 10) -> List[GeneratedPhrase]:
        """Get the top-scoring phrases."""
        try:
            return self.db.get_top_phrases(limit)
        except DatabaseError as e:
            raise RankingError(f"Failed to get top phrases: {e}")

    def get_context_phrases(self, count: Optional[int] = None) -> List[str]:
        """
        Get phrases to use as context for LLM generation.

        Args:
            count: Number of phrases to return (uses config default if None)

        Returns:
            List of phrase strings for context
        """
        if count is None:
            count = self.config.context_phrases_count

        try:
            top_phrases = self.db.get_top_phrases(count)
            return [phrase.phrase for phrase in top_phrases]
        except DatabaseError as e:
            self.logger.warning(f"Could not get context phrases: {e}")
            return []

    def get_ranking_stats(self) -> Dict[str, any]:
        """Get current ranking statistics."""
        try:
            db_stats = self.db.get_database_stats()
            top_phrases = self.db.get_top_phrases(5)

            stats = {
                'total_phrases': db_stats['total_phrases'],
                'max_score': db_stats['max_score'],
                'avg_score': db_stats['avg_score'],
                'top_5_phrases': [f"{p.phrase} ({p.score})" for p in top_phrases],
                'last_cleanup': getattr(self, '_last_cleanup', 'Never'),
                'config': {
                    'min_score_threshold': self.config.min_score_threshold,
                    'max_phrases_stored': self.config.max_phrases_stored,
                    'cleanup_threshold': self.config.cleanup_threshold
                }
            }

            return stats

        except DatabaseError as e:
            raise RankingError(f"Failed to get ranking stats: {e}")

    def optimize_phrase_selection(self, candidates: List[str], tiles: TileInventory) -> List[str]:
        """
        Select the most promising phrases from candidates based on various criteria.

        Args:
            candidates: List of phrase candidates
            tiles: Available tiles

        Returns:
            Filtered list of most promising phrases
        """
        if not candidates:
            return []

        # Score and validate all candidates
        phrase_scores = []
        for phrase in candidates:
            is_valid, tiles_used, _ = self.validator.validate_phrase(phrase, tiles)
            if is_valid:
                score = self.scorer.score_phrase_simple(phrase, tiles_used)
                difficulty = self.validator.get_phrase_difficulty(phrase, tiles)
                efficiency = self.scorer.get_score_efficiency(score, tiles)

                # Combined optimization score
                optimization_score = score * (1.0 - difficulty * 0.3) * (1.0 + efficiency * 0.2)

                phrase_scores.append((phrase, score, optimization_score))

        # Sort by optimization score
        phrase_scores.sort(key=lambda x: x[2], reverse=True)

        # Return top candidates (limit to reasonable number)
        limit = min(len(phrase_scores), self.config.generation_batch_size)
        return [phrase for phrase, _, _ in phrase_scores[:limit]]

    def get_improvable_phrases(self, limit: int = 10, max_failed_attempts: int = 5, max_children_created: int = 5) -> List[GeneratedPhrase]:
        """Get phrases that can still be improved (haven't reached retirement thresholds)."""
        try:
            return self.db.get_improvable_phrases(limit, max_failed_attempts, max_children_created)
        except DatabaseError as e:
            raise RankingError(f"Failed to get improvable phrases: {e}")

    def mark_improvement_success(self, phrase_id: int, children_count: int):
        """Mark that an improvement attempt was successful (reset failure counter and add children count)."""
        try:
            self.db.reset_failed_improvements(phrase_id)
            self.db.add_children_created(phrase_id, children_count)
            self.logger.debug(f"Reset failure counter and added {children_count} children for phrase {phrase_id}")
        except DatabaseError as e:
            raise RankingError(f"Failed to mark improvement success: {e}")

    def mark_improvement_failure(self, phrase_id: int):
        """Mark that an improvement attempt failed (increment failure counter)."""
        try:
            self.db.increment_failed_improvement(phrase_id)
            self.logger.debug(f"Incremented improvement failure counter for phrase {phrase_id}")
        except DatabaseError as e:
            raise RankingError(f"Failed to mark improvement failure: {e}")

    def _should_cleanup(self) -> bool:
        """Check if database cleanup is needed."""
        try:
            phrase_count = self.db.get_phrase_count()
            return phrase_count >= self.config.cleanup_threshold
        except DatabaseError:
            return False

    def _cleanup_low_scoring_phrases(self):
        """Remove low-scoring phrases to maintain performance."""
        try:
            deleted_count = self.db.cleanup_low_scoring_phrases(self.config.max_phrases_stored)
            self._last_cleanup = datetime.now().isoformat()
            self.logger.info(f"Cleaned up {deleted_count} low-scoring phrases")
        except DatabaseError as e:
            self.logger.error(f"Cleanup failed: {e}")

    def force_cleanup(self) -> int:
        """Force cleanup and return number of phrases removed."""
        try:
            deleted_count = self.db.cleanup_low_scoring_phrases(self.config.max_phrases_stored)
            self._last_cleanup = datetime.now().isoformat()
            return deleted_count
        except DatabaseError as e:
            raise RankingError(f"Force cleanup failed: {e}")

    def update_config(self, new_config: OptimizationConfig):
        """Update optimization configuration."""
        self.config = new_config
        self.logger.info(f"Updated optimization config: {new_config}")

    def get_phrase_history(self, limit: int = 50) -> List[GeneratedPhrase]:
        """Get recent phrase history."""
        try:
            return self.db.get_top_phrases(limit)  # Already sorted by score and date
        except DatabaseError as e:
            raise RankingError(f"Failed to get phrase history: {e}")

    def get_recent_phrases(self, limit: int = 10) -> List[GeneratedPhrase]:
        """Get the most recently generated phrases by date."""
        try:
            return self.db.get_recent_phrases(limit)
        except DatabaseError as e:
            raise RankingError(f"Failed to get recent phrases: {e}")

    def find_similar_phrases(self, target_phrase: str, limit: int = 5) -> List[GeneratedPhrase]:
        """
        Find phrases similar to the target phrase.
        Simple implementation based on shared words.
        """
        try:
            all_phrases = self.db.get_top_phrases(200)  # Get reasonable sample
            target_words = set(target_phrase.upper().split())

            similar_phrases = []
            for phrase in all_phrases:
                phrase_words = set(phrase.phrase.split())
                overlap = len(target_words.intersection(phrase_words))

                if overlap > 0 and phrase.phrase != target_phrase.upper():
                    similar_phrases.append((phrase, overlap))

            # Sort by overlap and score
            similar_phrases.sort(key=lambda x: (x[1], x[0].score), reverse=True)

            return [phrase for phrase, _ in similar_phrases[:limit]]

        except DatabaseError as e:
            self.logger.warning(f"Could not find similar phrases: {e}")
            return []


# Testing and examples
if __name__ == "__main__":

    from src.phrase_generator.tile_parser import parse_tile_string

    # Test with sample tiles
    test_tiles = parse_tile_string("9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh")
    print(f"Available tiles: {test_tiles.tiles}")

    # Initialize ranker with test database
    config = OptimizationConfig(
        max_phrases_stored=20,
        min_score_threshold=10,
        cleanup_threshold=50
    )

    ranker = PhraseRanker("test_ranking.db", config)

    # Test adding phrases
    test_candidates = [
        "WINTER MORNING",
        "SNOW DAY",
        "HOLIDAY CHEER",
        "COLD NIGHT",
        "WINTER WONDERLAND",
        "ICE AGE",  # Low score
        "FREEZING RAIN",
        "WINTER STORM"
    ]

    print(f"\nTesting phrase ranking with {len(test_candidates)} candidates...")

    added_phrases = ranker.add_phrase_candidates(
        test_candidates, test_tiles, "test-model", "test context"
    )

    print(f"Added {len(added_phrases)} valid phrases")

    # Get top phrases
    top_phrases = ranker.get_top_phrases(5)
    print(f"\nTop 5 phrases:")
    for i, phrase in enumerate(top_phrases, 1):
        print(f"  {i}. {phrase}")

    # Get stats
    stats = ranker.get_ranking_stats()
    print(f"\nRanking stats: {stats}")

    # Test context phrases
    context = ranker.get_context_phrases(3)
    print(f"\nContext phrases: {context}")

    # Cleanup test database
    import os
    if os.path.exists("test_ranking.db"):
        os.remove("test_ranking.db")
    print("\nTest completed and cleaned up")