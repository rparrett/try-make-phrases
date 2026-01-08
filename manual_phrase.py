#!/usr/bin/env python3
"""
Manual phrase seeding script for the Scrabble phrase generator.
Allows you to add manually discovered high-scoring phrases to the database.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.phrase_generator.tile_parser import parse_tile_string
from src.phrase_generator.phrase_validator import PhraseValidator
from src.phrase_generator.scorer import ScrabbleScorer
from storage.database import PhraseDatabase
from storage.models import GeneratedPhrase


def add_manual_phrase(phrase: str, tiles_input: str = None) -> bool:
    """
    Add a manually discovered phrase to the database.

    Args:
        phrase: The phrase to add
        tiles_input: Optional tile string for validation. If not provided, phrase is added without tile validation.

    Returns:
        True if successfully added, False otherwise
    """
    try:
        if tiles_input:
            # Parse and validate against provided tiles
            print(f"Parsing tiles: {tiles_input}")
            tiles = parse_tile_string(tiles_input)
            print(f"Available tiles: {tiles.tiles}")

            # Validate phrase can be built
            validator = PhraseValidator()
            is_valid, tiles_used, error = validator.validate_phrase(
                phrase.upper(), tiles
            )

            if not is_valid:
                print(f"❌ ERROR: '{phrase}' cannot be built: {error}")
                return False

            # Calculate score
            scorer = ScrabbleScorer()
            score = scorer.score_phrase_simple(phrase.upper(), tiles_used)

            print(f"✅ Phrase validated: '{phrase.upper()}'")
            print(f"   Score: {score}")
            print(f"   Tiles used: {tiles_used}")

            # Create phrase object with tile validation
            generated_phrase = GeneratedPhrase(
                phrase=phrase.upper(),
                score=score,
                tiles_used=tiles_used,
                model_used="manual-entry",
                prompt_context="Manually discovered phrase",
            )
        else:
            # No tiles provided - add phrase without validation
            # Calculate basic score using letter values
            from config.scrabble_values import get_letter_value, is_free_character

            score = 0
            for char in phrase.upper():
                if not is_free_character(char):
                    score += get_letter_value(char)

            print(f"✅ Adding phrase without tile validation: '{phrase.upper()}'")
            print(f"   Basic score: {score}")

            # Create phrase object without tile validation
            generated_phrase = GeneratedPhrase(
                phrase=phrase.upper(),
                score=score,
                tiles_used={},  # Empty dict since no tiles validated
                model_used="manual-entry",
                prompt_context="Manually added phrase (no tile validation)",
            )

        # Add to database
        db = PhraseDatabase("data/phrases.db")
        phrase_id = db.add_phrase(generated_phrase)

        print(
            f"🎯 Successfully added '{phrase.upper()}' with score {score} (ID: {phrase_id})"
        )
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to add phrase: {e}")
        return False


def add_multiple_phrases(phrases: list, tiles_input: str = None):
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


def clear_all_phrases():
    """Delete all phrases from the database."""
    try:
        db = PhraseDatabase("data/phrases.db")
        count = db.get_phrase_count()

        if count == 0:
            print("✅ Database is already empty")
            return

        # Clear the database
        db.clear_all_phrases()
        print(f"🗑️  Successfully deleted all {count} phrases")

    except Exception as e:
        print(f"❌ ERROR: Failed to clear phrases: {e}")


def main():
    """Main script entry point."""
    if len(sys.argv) < 2:
        print("🎯 Manual Phrase Seeding Script")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python manual_phrase.py 'YOUR PHRASE HERE'")
        print("  python manual_phrase.py 'PHRASE' 'tile_string_for_validation'")
        print("  python manual_phrase.py --clear 'PHRASE' [tile_string]")
        print("  python manual_phrase.py --batch [optional_tile_string]")
        print("  python manual_phrase.py --clear --batch [optional_tile_string]")
        print()
        print("Examples:")
        print(
            "  python manual_phrase.py 'WINTER SLEDDING CHAMPIONSHIP'  # No tile validation"
        )
        print(
            "  python manual_phrase.py 'BRONZE MEDAL CEREMONY' '9i13e2mk...'  # With tile validation"
        )
        print("  python manual_phrase.py --clear 'WINTER WONDERLAND'  # Clear all, add phrase")
        print("  python manual_phrase.py --clear --batch  # Clear all, add batch")
        print()
        print(
            "If no tiles provided, phrase is added with basic scoring (no tile validation)."
        )
        print("For batch mode, edit the 'phrases' list in this script.")
        sys.exit(1)

    # Check for clear flag
    should_clear = "--clear" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--clear"]

    if should_clear:
        clear_all_phrases()
        print()  # Add spacing after clear

    if not args:
        # Just --clear with no other args
        return
    elif args[0] == "--batch":
        # Batch mode
        phrases = [
            "WINTER OLYMPICS BRONZE MEDAL CEREMONY",
            "FROZEN POND ICE SKATING CHAMPIONSHIP",
            "COZY FIREPLACE EVENING WITH JAZZ MUSIC",
            "SLEDDING ADVENTURE IN THE SNOWY MOUNTAINS",
            # Add your phrases here...
        ]

        tiles = args[1] if len(args) > 1 else None
        add_multiple_phrases(phrases, tiles)
    else:
        # Single phrase mode
        phrase = args[0]
        tiles = args[1] if len(args) > 1 else None

        print("🎯 Adding Manual Phrase")
        print("-" * 30)
        add_manual_phrase(phrase, tiles)


if __name__ == "__main__":
    main()
