"""
SQLite database operations for phrase storage and ranking.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from storage.models import GeneratedPhrase, GenerationSession, OptimizationConfig


class DatabaseError(Exception):
    """Exception raised for database operations."""
    pass


class PhraseDatabase:
    """Handles SQLite operations for phrase storage."""

    def __init__(self, db_path: str = "data/phrases.db"):
        """Initialize database connection and schema."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")  # Better performance

            # Phrases table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS phrases (
                    id INTEGER PRIMARY KEY,
                    phrase TEXT UNIQUE NOT NULL,
                    score INTEGER NOT NULL,
                    tiles_used TEXT NOT NULL,  -- JSON encoded
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_used TEXT DEFAULT 'llama2:7b',
                    prompt_context TEXT
                )
            """)

            # Generation sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_sessions (
                    id INTEGER PRIMARY KEY,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    phrases_generated INTEGER DEFAULT 0,
                    valid_phrases INTEGER DEFAULT 0,
                    top_score INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0.0,
                    tiles_input TEXT NOT NULL
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_phrase_score ON phrases(score DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_phrase_generated ON phrases(generated_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session_start ON generation_sessions(session_start DESC)")

            conn.commit()

    def add_phrase(self, phrase: GeneratedPhrase) -> int:
        """
        Add a generated phrase to the database.

        Args:
            phrase: GeneratedPhrase to add

        Returns:
            The ID of the inserted phrase

        Raises:
            DatabaseError: If phrase already exists or other DB error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO phrases (phrase, score, tiles_used, generated_at, model_used, prompt_context)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    phrase.phrase,
                    phrase.score,
                    json.dumps(phrase.tiles_used),
                    phrase.generated_at,
                    phrase.model_used,
                    phrase.prompt_context
                ))
                phrase_id = cursor.lastrowid
                conn.commit()
                return phrase_id

        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                raise DatabaseError(f"Phrase already exists: {phrase.phrase}")
            raise DatabaseError(f"Database integrity error: {e}")
        except Exception as e:
            raise DatabaseError(f"Failed to add phrase: {e}")

    def add_phrases_batch(self, phrases: List[GeneratedPhrase]) -> List[int]:
        """Add multiple phrases in a single transaction."""
        phrase_ids = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                for phrase in phrases:
                    try:
                        cursor = conn.execute("""
                            INSERT INTO phrases (phrase, score, tiles_used, generated_at, model_used, prompt_context)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            phrase.phrase,
                            phrase.score,
                            json.dumps(phrase.tiles_used),
                            phrase.generated_at,
                            phrase.model_used,
                            phrase.prompt_context
                        ))
                        phrase_ids.append(cursor.lastrowid)
                    except sqlite3.IntegrityError:
                        # Skip duplicates but continue with batch
                        continue

                conn.commit()
                return phrase_ids

        except Exception as e:
            raise DatabaseError(f"Failed to add phrase batch: {e}")

    def get_top_phrases(self, limit: int = 10) -> List[GeneratedPhrase]:
        """Get the top scoring phrases."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM phrases
                    ORDER BY score DESC, generated_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()

                phrases = []
                for row in rows:
                    phrases.append(GeneratedPhrase(
                        id=row['id'],
                        phrase=row['phrase'],
                        score=row['score'],
                        tiles_used=json.loads(row['tiles_used']),
                        generated_at=datetime.fromisoformat(row['generated_at']),
                        model_used=row['model_used'],
                        prompt_context=row['prompt_context']
                    ))

                return phrases

        except Exception as e:
            raise DatabaseError(f"Failed to get top phrases: {e}")

    def get_recent_phrases(self, limit: int = 10) -> List[GeneratedPhrase]:
        """Get the most recently generated phrases by date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM phrases
                    ORDER BY generated_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()

                phrases = []
                for row in rows:
                    phrases.append(GeneratedPhrase(
                        id=row['id'],
                        phrase=row['phrase'],
                        score=row['score'],
                        tiles_used=json.loads(row['tiles_used']),
                        generated_at=datetime.fromisoformat(row['generated_at']),
                        model_used=row['model_used'],
                        prompt_context=row['prompt_context']
                    ))

                return phrases

        except Exception as e:
            raise DatabaseError(f"Failed to get recent phrases: {e}")

    def get_phrase_count(self) -> int:
        """Get total number of phrases in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM phrases").fetchone()[0]
                return count
        except Exception as e:
            raise DatabaseError(f"Failed to get phrase count: {e}")

    def cleanup_low_scoring_phrases(self, keep_count: int = 1000) -> int:
        """Remove low-scoring phrases, keeping only the top N."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get the score threshold for the Nth best phrase
                threshold_score = conn.execute("""
                    SELECT score FROM phrases
                    ORDER BY score DESC
                    LIMIT 1 OFFSET ?
                """, (keep_count - 1,)).fetchone()

                if threshold_score is None:
                    return 0  # Not enough phrases to clean up

                threshold = threshold_score[0]

                # Delete phrases below threshold
                cursor = conn.execute("""
                    DELETE FROM phrases
                    WHERE score < ?
                """, (threshold,))

                deleted_count = cursor.rowcount
                conn.commit()

                # Also vacuum to reclaim space
                conn.execute("VACUUM")

                return deleted_count

        except Exception as e:
            raise DatabaseError(f"Failed to cleanup phrases: {e}")

    def start_generation_session(self, tiles_input: str) -> int:
        """Start a new generation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO generation_sessions (tiles_input)
                    VALUES (?)
                """, (tiles_input,))
                session_id = cursor.lastrowid
                conn.commit()
                return session_id

        except Exception as e:
            raise DatabaseError(f"Failed to start session: {e}")

    def update_session_stats(self, session_id: int, phrases_generated: int,
                           valid_phrases: int, top_score: int, avg_score: float):
        """Update generation session statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE generation_sessions
                    SET phrases_generated = phrases_generated + ?,
                        valid_phrases = valid_phrases + ?,
                        top_score = MAX(top_score, ?),
                        avg_score = ?
                    WHERE id = ?
                """, (phrases_generated, valid_phrases, top_score, avg_score, session_id))
                conn.commit()

        except Exception as e:
            raise DatabaseError(f"Failed to update session stats: {e}")

    def get_session_stats(self, session_id: int) -> Optional[GenerationSession]:
        """Get statistics for a generation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute("""
                    SELECT * FROM generation_sessions WHERE id = ?
                """, (session_id,)).fetchone()

                if row is None:
                    return None

                return GenerationSession(
                    id=row['id'],
                    session_start=datetime.fromisoformat(row['session_start']),
                    phrases_generated=row['phrases_generated'],
                    valid_phrases=row['valid_phrases'],
                    top_score=row['top_score'],
                    avg_score=row['avg_score'],
                    tiles_input=row['tiles_input']
                )

        except Exception as e:
            raise DatabaseError(f"Failed to get session stats: {e}")

    def get_database_stats(self) -> Dict[str, any]:
        """Get overall database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Phrase statistics
                phrase_stats = conn.execute("""
                    SELECT
                        COUNT(*) as total_phrases,
                        MAX(score) as max_score,
                        AVG(score) as avg_score,
                        MIN(generated_at) as first_phrase,
                        MAX(generated_at) as last_phrase
                    FROM phrases
                """).fetchone()

                # Session statistics
                session_stats = conn.execute("""
                    SELECT
                        COUNT(*) as total_sessions,
                        SUM(phrases_generated) as total_generated,
                        SUM(valid_phrases) as total_valid
                    FROM generation_sessions
                """).fetchone()

                return {
                    'total_phrases': phrase_stats[0] if phrase_stats else 0,
                    'max_score': phrase_stats[1] if phrase_stats else 0,
                    'avg_score': round(phrase_stats[2], 2) if phrase_stats and phrase_stats[2] else 0,
                    'first_phrase': phrase_stats[3] if phrase_stats else None,
                    'last_phrase': phrase_stats[4] if phrase_stats else None,
                    'total_sessions': session_stats[0] if session_stats else 0,
                    'total_generated': session_stats[1] if session_stats else 0,
                    'total_valid': session_stats[2] if session_stats else 0
                }

        except Exception as e:
            raise DatabaseError(f"Failed to get database stats: {e}")

    def close(self):
        """Close database connection (if needed for cleanup)."""
        # With context managers, connections are automatically closed
        pass


# Example usage and testing
if __name__ == "__main__":
    # Test the database
    db = PhraseDatabase("test_phrases.db")

    # Test adding a phrase
    test_phrase = GeneratedPhrase(
        phrase="WINTER WONDERLAND",
        score=85,
        tiles_used={"W": 2, "I": 1, "N": 4, "T": 1, "E": 2, "R": 3, "O": 1, "D": 2, "L": 2, "A": 1},
        model_used="test"
    )

    try:
        phrase_id = db.add_phrase(test_phrase)
        print(f"Added phrase with ID: {phrase_id}")

        # Test getting top phrases
        top_phrases = db.get_top_phrases(5)
        print(f"Top phrases: {[str(p) for p in top_phrases]}")

        # Test stats
        stats = db.get_database_stats()
        print(f"Database stats: {stats}")

    except DatabaseError as e:
        print(f"Database error: {e}")

    # Cleanup test database
    import os
    if os.path.exists("test_phrases.db"):
        os.remove("test_phrases.db")