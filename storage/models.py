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