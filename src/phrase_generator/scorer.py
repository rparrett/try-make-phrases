"""
Scrabble scoring system for phrases.
"""

from typing import Dict, Tuple, Optional
import logging

from storage.models import TileInventory, GeneratedPhrase
from config.scrabble_values import get_letter_value, is_free_character


class ScoringError(Exception):
    """Exception raised for scoring errors."""
    pass


class ScrabbleScorer:
    """Calculates Scrabble scores for phrases."""

    def __init__(self):
        """Initialize the scorer."""
        self.logger = logging.getLogger(__name__)

    def score_phrase(self, phrase: str, tiles_used: Dict[str, int]) -> Tuple[int, Dict[str, Dict[str, int]]]:
        """
        Calculate the Scrabble score for a phrase.

        Args:
            phrase: The phrase to score
            tiles_used: Dictionary of tiles used (from validator)

        Returns:
            Tuple of (total_score, score_breakdown)
            - total_score: Total points for the phrase
            - score_breakdown: Detailed breakdown by letter
        """
        if not phrase or not phrase.strip():
            return 0, {}

        try:
            phrase = phrase.strip().upper()
            total_score = 0
            score_breakdown = {}

            # Count how many of each letter we used from tiles
            letter_usage = {}
            blanks_used = tiles_used.get('_', 0)
            blank_assignments = {}

            # First, assign regular tiles
            for letter in phrase:
                if is_free_character(letter):
                    continue  # Free characters don't contribute to score

                if letter in tiles_used and letter != '_':
                    # Use regular tile
                    if letter not in letter_usage:
                        letter_usage[letter] = 0

                    if letter_usage[letter] < tiles_used[letter]:
                        letter_usage[letter] += 1
                        letter_score = get_letter_value(letter)
                        total_score += letter_score

                        if letter not in score_breakdown:
                            score_breakdown[letter] = {'count': 0, 'points_per': letter_score, 'total': 0}
                        score_breakdown[letter]['count'] += 1
                        score_breakdown[letter]['total'] += letter_score
                    else:
                        # Must use a blank tile for this letter
                        if blanks_used > 0:
                            if letter not in blank_assignments:
                                blank_assignments[letter] = 0
                            blank_assignments[letter] += 1
                            blanks_used -= 1

                            # Blank tiles score 0 points
                            if '_' not in score_breakdown:
                                score_breakdown['_'] = {'count': 0, 'points_per': 0, 'total': 0, 'representing': {}}
                            score_breakdown['_']['count'] += 1
                            if letter not in score_breakdown['_']['representing']:
                                score_breakdown['_']['representing'][letter] = 0
                            score_breakdown['_']['representing'][letter] += 1

            return total_score, score_breakdown

        except Exception as e:
            self.logger.error(f"Scoring error for phrase '{phrase}': {e}")
            raise ScoringError(f"Failed to score phrase: {e}")

    def score_phrase_simple(self, phrase: str, tiles_used: Dict[str, int]) -> int:
        """
        Simple scoring that just returns the total score.

        Args:
            phrase: The phrase to score
            tiles_used: Dictionary of tiles used

        Returns:
            Total score
        """
        score, _ = self.score_phrase(phrase, tiles_used)
        return score

    def create_scored_phrase(self, phrase: str, tiles: TileInventory, tiles_used: Dict[str, int],
                           model_used: str = "llama2:7b", prompt_context: Optional[str] = None) -> GeneratedPhrase:
        """
        Create a GeneratedPhrase object with calculated score.

        Args:
            phrase: The phrase text
            tiles: Available tiles (for validation)
            tiles_used: Tiles consumed for this phrase
            model_used: LLM model that generated the phrase
            prompt_context: Context used in the prompt

        Returns:
            GeneratedPhrase object with score
        """
        score = self.score_phrase_simple(phrase, tiles_used)

        return GeneratedPhrase(
            phrase=phrase.strip().upper(),
            score=score,
            tiles_used=tiles_used,
            model_used=model_used,
            prompt_context=prompt_context
        )

    def get_max_possible_score(self, tiles: TileInventory) -> int:
        """
        Calculate the maximum possible score with the given tiles.
        This is a rough estimate using highest-value letters.

        Args:
            tiles: Available tiles

        Returns:
            Maximum possible score (approximate)
        """
        total_score = 0

        for letter, count in tiles.tiles.items():
            if letter == '_':
                continue  # Blanks are worth 0
            letter_value = get_letter_value(letter)
            total_score += letter_value * count

        return total_score

    def get_score_efficiency(self, phrase_score: int, tiles: TileInventory) -> float:
        """
        Calculate score efficiency as a percentage of maximum possible.

        Args:
            phrase_score: Actual phrase score
            tiles: Available tiles

        Returns:
            Efficiency ratio (0.0 to 1.0)
        """
        max_score = self.get_max_possible_score(tiles)
        if max_score == 0:
            return 0.0

        return min(1.0, phrase_score / max_score)

    def format_score_breakdown(self, score_breakdown: Dict[str, Dict[str, int]]) -> str:
        """
        Format a score breakdown for display.

        Args:
            score_breakdown: Detailed score breakdown

        Returns:
            Human-readable score breakdown
        """
        if not score_breakdown:
            return "No score breakdown available"

        parts = []

        # Regular letters
        for letter, details in sorted(score_breakdown.items()):
            if letter == '_':
                continue  # Handle blanks separately

            count = details['count']
            points_per = details['points_per']
            total = details['total']

            if count == 1:
                parts.append(f"{letter}({points_per})")
            else:
                parts.append(f"{letter}×{count}({points_per}ea={total})")

        # Blank tiles
        if '_' in score_breakdown:
            blank_details = score_breakdown['_']
            blank_count = blank_details['count']
            representing = blank_details.get('representing', {})

            blank_desc = f"Blanks×{blank_count}(0pts"
            if representing:
                repr_parts = [f"{letter}×{count}" for letter, count in representing.items()]
                blank_desc += f" as {', '.join(repr_parts)}"
            blank_desc += ")"
            parts.append(blank_desc)

        return " + ".join(parts)


# Testing and examples
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    from src.phrase_generator.tile_parser import parse_tile_string
    from src.phrase_generator.phrase_validator import PhraseValidator

    # Test with sample tiles
    test_tiles = parse_tile_string("9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh")
    print(f"Available tiles: {test_tiles.tiles}")

    validator = PhraseValidator()
    scorer = ScrabbleScorer()

    # Test phrases
    test_phrases = [
        "WINTER MORNING",
        "SNOW DAY",
        "HOLIDAY CHEER",
        "COLD NIGHT",
        "WINTER WONDERLAND"
    ]

    print(f"\nMaximum possible score: {scorer.get_max_possible_score(test_tiles)}")
    print("\nTesting phrase scoring:")

    for phrase in test_phrases:
        is_valid, tiles_used, error = validator.validate_phrase(phrase, test_tiles)

        if is_valid:
            score, breakdown = scorer.score_phrase(phrase, tiles_used)
            efficiency = scorer.get_score_efficiency(score, test_tiles)
            breakdown_str = scorer.format_score_breakdown(breakdown)

            print(f"\n  '{phrase}':")
            print(f"    Score: {score} points")
            print(f"    Efficiency: {efficiency:.1%}")
            print(f"    Breakdown: {breakdown_str}")

            # Test creating GeneratedPhrase
            generated_phrase = scorer.create_scored_phrase(phrase, test_tiles, tiles_used)
            print(f"    Generated: {generated_phrase}")
        else:
            print(f"\n  '{phrase}': INVALID - {error}")

    print()