"""
Pydantic data models for the phrase generator.
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class TileInventory(BaseModel):
    """Represents available tiles with quantities."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tiles: Dict[str, int] = Field(..., description="Letter to quantity mapping")

    def get_tile_count(self, letter: str) -> int:
        """Get the count of a specific tile."""
        return self.tiles.get(letter.upper(), 0)

    def use_tile(self, letter: str) -> bool:
        """Use a tile if available. Returns True if successful."""
        letter = letter.upper()
        if letter in self.tiles and self.tiles[letter] > 0:
            self.tiles[letter] -= 1
            return True
        return False

    def can_construct_phrase(self, phrase: str) -> bool:
        """Check if phrase can be constructed from available tiles."""
        from config.scrabble_values import is_free_character

        temp_tiles = self.tiles.copy()

        for char in phrase.upper():
            if is_free_character(char):
                continue  # Free characters don't consume tiles

            # Check if we have the letter
            if char in temp_tiles and temp_tiles[char] > 0:
                temp_tiles[char] -= 1
            # Check if we can use a blank tile
            elif '_' in temp_tiles and temp_tiles['_'] > 0:
                temp_tiles['_'] -= 1
            else:
                return False

        return True


class GeneratedPhrase(BaseModel):
    """Represents a generated phrase with metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[int] = None
    phrase: str = Field(..., description="The actual phrase text")
    score: int = Field(..., description="Scrabble score for the phrase")
    tiles_used: Dict[str, int] = Field(..., description="Tiles consumed to make this phrase")
    generated_at: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(default="llama2:7b", description="LLM model that generated this phrase")
    prompt_context: Optional[str] = None
    consecutive_failed_improvements: int = Field(default=0, description="Number of consecutive failed improvement attempts")
    children_created: int = Field(default=0, description="Total number of child phrases created from this phrase")

    def __str__(self) -> str:
        return f"{self.phrase} (Score: {self.score})"

    def __repr__(self) -> str:
        return f"GeneratedPhrase(phrase='{self.phrase}', score={self.score})"


class GenerationSession(BaseModel):
    """Statistics for a generation session."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: Optional[int] = None
    session_start: datetime = Field(default_factory=datetime.now)
    phrases_generated: int = 0
    valid_phrases: int = 0
    top_score: int = 0
    avg_score: float = 0.0
    tiles_input: str = Field(..., description="Original tile input string")

    # Fresh generation stats
    fresh_attempts: int = 0
    fresh_successes: int = 0
    fresh_avg_score: float = 0.0

    # Improvement stats
    improvement_attempts: int = 0
    improvement_successes: int = 0
    improvement_avg_score: float = 0.0

    # Improvement effectiveness
    improvements_better_than_original: int = 0
    improvements_worse_than_original: int = 0

    def update_stats(self, new_phrases: List[GeneratedPhrase]):
        """Update session statistics with new phrases."""
        if not new_phrases:
            return

        self.phrases_generated += len(new_phrases)

        valid_count = 0
        total_score = 0
        max_score = 0

        for phrase in new_phrases:
            if phrase.score > 0:  # Consider non-zero scores as valid
                valid_count += 1
                total_score += phrase.score
                max_score = max(max_score, phrase.score)

        self.valid_phrases += valid_count
        self.top_score = max(self.top_score, max_score)

        if self.valid_phrases > 0:
            # Recalculate average across all valid phrases
            self.avg_score = total_score / self.valid_phrases

    def update_fresh_stats(self, attempts: int, successes: int, scores: List[int]):
        """Update statistics for fresh phrase generation."""
        self.fresh_attempts += attempts
        self.fresh_successes += successes

        if scores and successes > 0:
            # Calculate running average for fresh phrases
            if self.fresh_avg_score == 0.0:
                self.fresh_avg_score = sum(scores) / len(scores)
            else:
                # Weighted average incorporating new scores
                total_fresh_score = self.fresh_avg_score * (self.fresh_successes - successes)
                total_fresh_score += sum(scores)
                self.fresh_avg_score = total_fresh_score / self.fresh_successes

    def update_improvement_stats(self, attempts: int, successes: int, scores: List[int],
                               original_scores: List[int], improved_scores: List[int]):
        """Update statistics for phrase improvements."""
        self.improvement_attempts += attempts
        self.improvement_successes += successes

        if scores and successes > 0:
            # Calculate running average for improvements
            if self.improvement_avg_score == 0.0:
                self.improvement_avg_score = sum(scores) / len(scores)
            else:
                # Weighted average incorporating new scores
                total_improvement_score = self.improvement_avg_score * (self.improvement_successes - successes)
                total_improvement_score += sum(scores)
                self.improvement_avg_score = total_improvement_score / self.improvement_successes

        # Track improvement effectiveness
        for orig_score, improved_score in zip(original_scores, improved_scores):
            if improved_score > orig_score:
                self.improvements_better_than_original += 1
            elif improved_score < orig_score:
                self.improvements_worse_than_original += 1
            # Equal scores don't count in either category

    def get_fresh_success_rate(self) -> float:
        """Get success rate for fresh generation."""
        if self.fresh_attempts == 0:
            return 0.0
        return (self.fresh_successes / self.fresh_attempts) * 100

    def get_improvement_success_rate(self) -> float:
        """Get success rate for improvements."""
        if self.improvement_attempts == 0:
            return 0.0
        return (self.improvement_successes / self.improvement_attempts) * 100

    def get_improvement_effectiveness(self) -> float:
        """Get percentage of improvements that were better than original."""
        total_comparisons = self.improvements_better_than_original + self.improvements_worse_than_original
        if total_comparisons == 0:
            return 0.0
        return (self.improvements_better_than_original / total_comparisons) * 100


class OptimizationConfig(BaseModel):
    """Configuration for the optimization process."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_phrases_stored: int = Field(default=1000, description="Maximum phrases to keep in database")
    generation_batch_size: int = Field(default=15, description="Number of phrases to generate per LLM call")
    iteration_delay_seconds: int = Field(default=30, description="Seconds between generation cycles")
    context_phrases_count: int = Field(default=5, description="Number of top phrases to use as context")
    cleanup_threshold: int = Field(default=10000, description="Trigger cleanup when exceeding this many phrases")
    min_score_threshold: int = Field(default=10, description="Minimum score to consider keeping a phrase")

    # MacBook optimization settings
    max_cpu_usage: float = Field(default=80.0, description="Max CPU usage before throttling")
    max_memory_mb: int = Field(default=2048, description="Max memory usage in MB")
    thermal_monitoring: bool = Field(default=True, description="Enable thermal monitoring")


class SystemHealth(BaseModel):
    """Current system health metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    temperature: Optional[float] = None
    phrases_per_hour: float = 0.0
    valid_phrase_rate: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)

    def is_healthy(self, config: OptimizationConfig) -> bool:
        """Check if system is healthy for continued operation."""
        return (
            self.cpu_usage < config.max_cpu_usage and
            self.memory_usage_mb < config.max_memory_mb and
            (self.temperature is None or self.temperature < 85.0)  # CPU temp threshold
        )