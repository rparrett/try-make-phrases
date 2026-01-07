#!/usr/bin/env python3
"""
Manual phrase seeding script for the Scrabble phrase generator.
Allows you to add manually discovered high-scoring phrases to the database.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.phrase_generator.tile_parser import parse_tile_string
from src.phrase_generator.phrase_validator import PhraseValidator
from src.phrase_generator.scorer import ScrabbleScorer
from storage.database import PhraseDatabase
from storage.models import GeneratedPhrase


def add_manual_phrase(phrase: str, tiles_input: str = "9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh") -> bool:
    """
    Add a manually discovered phrase to the database.

    Args:
        phrase: The phrase to add
        tiles_input: Tile string (defaults to your current set)

    Returns:
        True if successfully added, False otherwise
    """
    try:
        # Parse your tiles
        print(f"Parsing tiles: {tiles_input}")
        tiles = parse_tile_string(tiles_input)
        print(f"Available tiles: {tiles.tiles}")

        # Validate phrase can be built
        validator = PhraseValidator()
        is_valid, tiles_used, error = validator.validate_phrase(phrase.upper(), tiles)

        if not is_valid:
            print(f"❌ ERROR: '{phrase}' cannot be built: {error}")
            return False

        # Calculate score
        scorer = ScrabbleScorer()
        score = scorer.score_phrase_simple(phrase.upper(), tiles_used)

        print(f"✅ Phrase validated: '{phrase.upper()}'")
        print(f"   Score: {score}")
        print(f"   Tiles used: {tiles_used}")

        # Create phrase object
        generated_phrase = GeneratedPhrase(
            phrase=phrase.upper(),
            score=score,
            tiles_used=tiles_used,
            model_used="manual-entry",
            prompt_context="Manually discovered phrase"
        )

        # Add to database
        db = PhraseDatabase("data/phrases.db")
        phrase_id = db.add_phrase(generated_phrase)

        print(f"🎯 Successfully added '{phrase.upper()}' with score {score} (ID: {phrase_id})")
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to add phrase: {e}")
        return False


def add_multiple_phrases(phrases: list, tiles_input: str = "9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh"):
    """Add multiple phrases at once."""
    print(f"Adding {len(phrases)} phrases...")
    print("-" * 50)

    successful = 0
    failed = 0

    for i, phrase in enumerate(phrases, 1):
        print(f"\n[{i}/{len(phrases)}] Processing: '{phrase}'")
        if add_manual_phrase(phrase, tiles_input):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"✅ Successfully added: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total processed: {len(phrases)}")


def main():
    """Main script entry point."""
    if len(sys.argv) < 2:
        print("🎯 Manual Phrase Seeding Script")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python manual_phrase.py 'YOUR PHRASE HERE'")
        print("  python manual_phrase.py 'PHRASE' 'custom_tile_string'")
        print("  python manual_phrase.py --batch")
        print()
        print("Examples:")
        print("  python manual_phrase.py 'WINTER SLEDDING CHAMPIONSHIP'")
        print("  python manual_phrase.py 'BRONZE MEDAL CEREMONY' '9i13e2mk...'")
        print()
        print("For batch mode, edit the 'phrases' list in this script.")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        # Edit this list to add multiple phrases at once
        phrases = [
            "WINTER OLYMPICS BRONZE MEDAL CEREMONY",
            "FROZEN POND ICE SKATING CHAMPIONSHIP",
            "COZY FIREPLACE EVENING WITH JAZZ MUSIC",
            "SLEDDING ADVENTURE IN THE SNOWY MOUNTAINS"
            # Add your phrases here...
        ]

        tiles = sys.argv[2] if len(sys.argv) > 2 else "9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh"
        add_multiple_phrases(phrases, tiles)
    else:
        # Single phrase mode
        phrase = sys.argv[1]
        tiles = sys.argv[2] if len(sys.argv) > 2 else "9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh"

        print("🎯 Adding Manual Phrase")
        print("-" * 30)
        add_manual_phrase(phrase, tiles)


if __name__ == "__main__":
    main()