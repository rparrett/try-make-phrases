#!/usr/bin/env python3
"""
Standalone script to recalculate phrase scores for a specific tile set.
Useful if you want to update scores without running the full generator.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.phrase_generator.tile_parser import parse_tile_string
from src.phrase_generator.phrase_validator import PhraseValidator
from src.phrase_generator.scorer import ScrabbleScorer
from storage.database import PhraseDatabase


def recalculate_scores(tiles_input: str, db_path: str = "data/phrases.db"):
    """Recalculate all phrase scores for the given tile set."""
    try:
        print("🎯 Recalculating Phrase Scores")
        print(f"Tiles: {tiles_input}")
        print(f"Database: {db_path}")
        print("-" * 50)

        # Parse tiles
        tiles = parse_tile_string(tiles_input)
        print(f"📋 Parsed tiles: {tiles.tiles}")
        print(f"📊 Total tiles: {sum(tiles.tiles.values())}")

        # Initialize components
        scorer = ScrabbleScorer()
        validator = PhraseValidator()
        db = PhraseDatabase(db_path)

        # Get initial stats
        initial_count = db.get_phrase_count()
        print(f"🗄️  Initial phrases in database: {initial_count}")

        if initial_count == 0:
            print("✨ No phrases to recalculate!")
            return

        print("\n🔄 Recalculating...")

        # Recalculate scores
        recalc_stats = db.recalculate_scores_for_tileset(tiles, scorer, validator)

        print("\n✅ Recalculation Complete!")
        print(f"📈 Phrases updated: {recalc_stats['updated_count']}")
        print(f"🗑️  Phrases removed: {recalc_stats['removed_count']} (unbuildable)")

        if recalc_stats["removed_count"] > 0:
            final_count = db.get_phrase_count()
            print(f"📊 Final phrase count: {final_count}")

        # Show some score changes if verbose
        if len(sys.argv) > 2 and sys.argv[2] == "--verbose":
            print("\n📋 Score Changes (showing first 10):")
            for phrase, score in recalc_stats["score_changes"][:10]:
                print(f"   {phrase}: {score}")

        print(f"\n🎯 Database ready for tile set: {tiles_input}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("🎯 Phrase Score Recalculator")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python recalculate_scores.py 'tile_string' [--verbose]")
        print(
            "  python recalculate_scores.py '9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh'"
        )
        print()
        print("This will:")
        print("  - Recalculate all phrase scores for the new tile set")
        print("  - Remove phrases that cannot be built with the tiles")
        print("  - Update the database with new scores")
        sys.exit(1)

    tiles_input = sys.argv[1]
    recalculate_scores(tiles_input)


if __name__ == "__main__":
    main()
