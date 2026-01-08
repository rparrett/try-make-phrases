"""
Parser for scrabble tile input format.

Handles format like: 9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh
Where:
- Numbers before letters indicate quantity (default is 1)
- Blank tiles represented by underscore '_'
"""

import re
from typing import Dict
from storage.models import TileInventory


class TileParseError(Exception):
    """Exception raised when tile parsing fails."""

    pass


def parse_tile_string(tile_string: str) -> TileInventory:
    """
    Parse a tile string into a TileInventory.

    Args:
        tile_string: Format like '9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh'

    Returns:
        TileInventory with parsed tiles

    Raises:
        TileParseError: If the format is invalid
    """
    if not tile_string:
        raise TileParseError("Empty tile string")

    tiles: Dict[str, int] = {}

    # Pattern to match optional number followed by letter/underscore
    # (\d+)? captures optional number
    # ([a-zA-Z_]) captures the letter or underscore
    pattern = r"(\d+)?([a-zA-Z_])"

    matches = re.findall(pattern, tile_string)

    if not matches:
        raise TileParseError(f"No valid tiles found in: {tile_string}")

    # Track position for error reporting
    processed_chars = 0

    for count_str, letter in matches:
        # Convert count to integer (default 1)
        count = int(count_str) if count_str else 1

        if count <= 0:
            raise TileParseError(f"Invalid tile count: {count} for letter '{letter}'")

        if count > 50:  # Reasonable upper limit
            raise TileParseError(f"Tile count too large: {count} for letter '{letter}'")

        # Convert to uppercase for consistency, except for blank tile
        if letter == "_":
            tile_key = "_"
        else:
            tile_key = letter.upper()

        # Add to existing count if letter appears multiple times
        if tile_key in tiles:
            tiles[tile_key] += count
        else:
            tiles[tile_key] = count

        # Track processed characters for validation
        if count_str:
            processed_chars += len(count_str) + 1
        else:
            processed_chars += 1

    # Validate that we processed the entire string
    if processed_chars != len(tile_string):
        raise TileParseError(f"Invalid characters in tile string: {tile_string}")

    return TileInventory(tiles=tiles)


def validate_tiles(tiles: TileInventory) -> bool:
    """
    Validate that the tile inventory is reasonable.

    Args:
        tiles: TileInventory to validate

    Returns:
        True if valid, False otherwise
    """
    from config.scrabble_values import TILE_DISTRIBUTION

    total_tiles = sum(tiles.tiles.values())

    # Basic sanity checks
    if total_tiles == 0:
        return False

    if total_tiles > 200:  # Unreasonably large
        return False

    # Check individual letter counts aren't too extreme
    for letter, count in tiles.tiles.items():
        if letter == "_":
            # Blanks: allow up to reasonable number
            if count > 10:
                return False
        else:
            # Regular letters: check against standard distribution with some flexibility
            standard_count = TILE_DISTRIBUTION.get(letter, 0)
            if count > standard_count * 5:  # Allow up to 5x standard distribution
                return False

    return True


def format_tile_summary(tiles: TileInventory) -> str:
    """
    Format tiles for display.

    Args:
        tiles: TileInventory to format

    Returns:
        Human-readable summary
    """
    if not tiles.tiles:
        return "No tiles"

    # Sort for consistent output
    sorted_tiles = sorted(tiles.tiles.items())

    # Format as letter(count) for counts > 1, just letter for count = 1
    formatted = []
    for letter, count in sorted_tiles:
        if count == 1:
            formatted.append(letter)
        else:
            formatted.append(f"{letter}({count})")

    total_count = sum(tiles.tiles.values())
    return f"{' '.join(formatted)} (Total: {total_count} tiles)"


def tiles_to_string(tiles: TileInventory) -> str:
    """
    Convert TileInventory back to the original string format.

    Args:
        tiles: TileInventory to convert

    Returns:
        String in original format
    """
    if not tiles.tiles:
        return ""

    # Sort for consistent output
    sorted_tiles = sorted(tiles.tiles.items())

    parts = []
    for letter, count in sorted_tiles:
        if count == 1:
            parts.append(letter.lower() if letter != "_" else letter)
        else:
            parts.append(f"{count}{letter.lower() if letter != '_' else letter}")

    return "".join(parts)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_strings = [
        "9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh",
        "abc",  # Simple case
        "2a3b_",  # With blank
        "10z",  # High count
        "",  # Empty case
        "invalid123",  # Invalid format
    ]

    for test_str in test_strings:
        try:
            tiles = parse_tile_string(test_str)
            print(f"'{test_str}' -> {format_tile_summary(tiles)}")
            print(f"  Valid: {validate_tiles(tiles)}")
            print(f"  Back to string: '{tiles_to_string(tiles)}'")
        except TileParseError as e:
            print(f"'{test_str}' -> ERROR: {e}")
        print()
