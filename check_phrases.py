#!/usr/bin/env python3
"""
Quick database checker for manual phrases.
Shows your manually added phrases and top scorers.
"""

import sys
from pathlib import Path
import sqlite3

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def check_database(db_path: str = "data/phrases.db"):
    """Check the database contents."""
    try:
        # Direct SQLite connection for flexible queries
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        print("🎯 Phrase Database Summary")
        print("=" * 50)

        # Overall stats
        total_phrases = conn.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
        manual_phrases = conn.execute(
            "SELECT COUNT(*) FROM phrases WHERE model_used = 'manual-entry'"
        ).fetchone()[0]
        max_score = conn.execute("SELECT MAX(score) FROM phrases").fetchone()[0] or 0

        print(f"📊 Total phrases: {total_phrases}")
        print(f"✋ Manual entries: {manual_phrases}")
        print(f"🏆 Highest score: {max_score}")

        if manual_phrases > 0:
            print(f"\n✋ Your Manual Phrases (Top {min(10, manual_phrases)}):")
            print("-" * 50)
            manual = conn.execute("""
                SELECT phrase, score, generated_at
                FROM phrases
                WHERE model_used = 'manual-entry'
                ORDER BY score DESC
                LIMIT 10
            """).fetchall()

            for i, row in enumerate(manual, 1):
                print(f"{i:2}. {row['phrase']} ({row['score']})")

        print("\n🏆 Top Scoring Phrases (All Sources):")
        print("-" * 50)
        top_phrases = conn.execute("""
            SELECT phrase, score, model_used
            FROM phrases
            ORDER BY score DESC
            LIMIT 10
        """).fetchall()

        for i, row in enumerate(top_phrases, 1):
            source = "✋" if row["model_used"] == "manual-entry" else "🤖"
            print(f"{i:2}. {source} {row['phrase']} ({row['score']})")

        print("\n📈 Recent Activity (Last 10):")
        print("-" * 50)
        recent = conn.execute("""
            SELECT phrase, score, model_used, generated_at
            FROM phrases
            ORDER BY generated_at DESC
            LIMIT 10
        """).fetchall()

        for i, row in enumerate(recent, 1):
            source = "✋" if row["model_used"] == "manual-entry" else "🤖"
            timestamp = row["generated_at"][:16]  # YYYY-MM-DD HH:MM
            print(f"{i:2}. {source} {row['phrase']} ({row['score']}) - {timestamp}")

        conn.close()

    except Exception as e:
        print(f"❌ Error checking database: {e}")


def main():
    """Main entry point."""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/phrases.db"
    check_database(db_path)


if __name__ == "__main__":
    main()
