"""
SQLite database operations for phrase storage and ranking.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

from storage.models import GeneratedPhrase, GenerationSession


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
                    prompt_context TEXT,
                    consecutive_failed_improvements INTEGER DEFAULT 0,
                    children_created INTEGER DEFAULT 0
                )
            """)

            # Add the new columns to existing tables (migration)
            try:
                conn.execute(
                    "ALTER TABLE phrases ADD COLUMN consecutive_failed_improvements INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass

            try:
                conn.execute(
                    "ALTER TABLE phrases ADD COLUMN children_created INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                # Column already exists, ignore
                pass

            # Migrate data from old column name if it exists
            try:
                # Check if old column exists
                cursor = conn.execute("PRAGMA table_info(phrases)")
                columns = [row[1] for row in cursor.fetchall()]
                if (
                    "total_successful_improvements" in columns
                    and "children_created" in columns
                ):
                    # Copy data from old column to new column
                    conn.execute(
                        "UPDATE phrases SET children_created = total_successful_improvements WHERE children_created = 0"
                    )
                    conn.commit()
            except sqlite3.OperationalError:
                # Migration failed, ignore
                pass

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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_phrase_score ON phrases(score DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_phrase_generated ON phrases(generated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_start ON generation_sessions(session_start DESC)"
            )

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
                cursor = conn.execute(
                    """
                    INSERT INTO phrases (phrase, score, tiles_used, generated_at, model_used, prompt_context, consecutive_failed_improvements, children_created)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        phrase.phrase,
                        phrase.score,
                        json.dumps(phrase.tiles_used),
                        phrase.generated_at,
                        phrase.model_used,
                        phrase.prompt_context,
                        phrase.consecutive_failed_improvements,
                        phrase.children_created,
                    ),
                )
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
                        cursor = conn.execute(
                            """
                            INSERT INTO phrases (phrase, score, tiles_used, generated_at, model_used, prompt_context, consecutive_failed_improvements, children_created)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                            (
                                phrase.phrase,
                                phrase.score,
                                json.dumps(phrase.tiles_used),
                                phrase.generated_at,
                                phrase.model_used,
                                phrase.prompt_context,
                                phrase.consecutive_failed_improvements,
                                phrase.children_created,
                            ),
                        )
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
                rows = conn.execute(
                    """
                    SELECT * FROM phrases
                    ORDER BY score DESC, generated_at DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()

                phrases = []
                for row in rows:
                    # Handle backward compatibility for the new columns
                    try:
                        consecutive_failed_improvements = (
                            row["consecutive_failed_improvements"]
                            if "consecutive_failed_improvements" in row.keys()
                            else 0
                        )
                    except (KeyError, IndexError):
                        consecutive_failed_improvements = 0

                    try:
                        children_created = (
                            row["children_created"]
                            if "children_created" in row.keys()
                            else 0
                        )
                        # Fallback to old column name for backward compatibility
                        if (
                            children_created == 0
                            and "total_successful_improvements" in row.keys()
                        ):
                            children_created = row["total_successful_improvements"]
                    except (KeyError, IndexError):
                        children_created = 0

                    phrases.append(
                        GeneratedPhrase(
                            id=row["id"],
                            phrase=row["phrase"],
                            score=row["score"],
                            tiles_used=json.loads(row["tiles_used"]),
                            generated_at=datetime.fromisoformat(row["generated_at"]),
                            model_used=row["model_used"],
                            prompt_context=row["prompt_context"],
                            consecutive_failed_improvements=consecutive_failed_improvements,
                            children_created=children_created,
                        )
                    )

                return phrases

        except Exception as e:
            raise DatabaseError(f"Failed to get top phrases: {e}")

    def get_recent_phrases(self, limit: int = 10) -> List[GeneratedPhrase]:
        """Get the most recently generated phrases by date."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT * FROM phrases
                    ORDER BY generated_at DESC
                    LIMIT ?
                """,
                    (limit,),
                ).fetchall()

                phrases = []
                for row in rows:
                    # Handle backward compatibility for the new columns
                    try:
                        consecutive_failed_improvements = (
                            row["consecutive_failed_improvements"]
                            if "consecutive_failed_improvements" in row.keys()
                            else 0
                        )
                    except (KeyError, IndexError):
                        consecutive_failed_improvements = 0

                    try:
                        children_created = (
                            row["children_created"]
                            if "children_created" in row.keys()
                            else 0
                        )
                        # Fallback to old column name for backward compatibility
                        if (
                            children_created == 0
                            and "total_successful_improvements" in row.keys()
                        ):
                            children_created = row["total_successful_improvements"]
                    except (KeyError, IndexError):
                        children_created = 0

                    phrases.append(
                        GeneratedPhrase(
                            id=row["id"],
                            phrase=row["phrase"],
                            score=row["score"],
                            tiles_used=json.loads(row["tiles_used"]),
                            generated_at=datetime.fromisoformat(row["generated_at"]),
                            model_used=row["model_used"],
                            prompt_context=row["prompt_context"],
                            consecutive_failed_improvements=consecutive_failed_improvements,
                            children_created=children_created,
                        )
                    )

                return phrases

        except Exception as e:
            raise DatabaseError(f"Failed to get recent phrases: {e}")

    def recalculate_scores_for_tileset(
        self, tiles, scorer, validator
    ) -> Dict[str, int]:
        """
        Recalculate all phrase scores for a new tile set.
        Remove phrases that cannot be constructed.

        Args:
            tiles: TileInventory for the current session
            scorer: ScrabbleScorer instance
            validator: PhraseValidator instance

        Returns:
            Dict with recalculation stats
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Get all phrases
                rows = conn.execute("SELECT id, phrase FROM phrases").fetchall()

                updated_count = 0
                removed_count = 0
                score_changes = []

                for row in rows:
                    phrase_id = row["id"]
                    phrase_text = row["phrase"]

                    # Check if phrase can be constructed with current tiles
                    is_valid, tiles_used, error = validator.validate_phrase(
                        phrase_text, tiles
                    )

                    if not is_valid:
                        # Remove phrases that can't be constructed
                        conn.execute("DELETE FROM phrases WHERE id = ?", (phrase_id,))
                        removed_count += 1
                        continue

                    # Calculate new score
                    new_score = scorer.score_phrase_simple(phrase_text, tiles_used)

                    # Update phrase with new score and tiles_used
                    conn.execute(
                        """
                        UPDATE phrases
                        SET score = ?, tiles_used = ?
                        WHERE id = ?
                    """,
                        (new_score, json.dumps(tiles_used), phrase_id),
                    )

                    updated_count += 1
                    score_changes.append((phrase_text, new_score))

                conn.commit()

                return {
                    "updated_count": updated_count,
                    "removed_count": removed_count,
                    "score_changes": score_changes,
                }

        except Exception as e:
            raise DatabaseError(f"Failed to recalculate scores: {e}")

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
                threshold_score = conn.execute(
                    """
                    SELECT score FROM phrases
                    ORDER BY score DESC
                    LIMIT 1 OFFSET ?
                """,
                    (keep_count - 1,),
                ).fetchone()

                if threshold_score is None:
                    return 0  # Not enough phrases to clean up

                threshold = threshold_score[0]

                # Delete phrases below threshold
                cursor = conn.execute(
                    """
                    DELETE FROM phrases
                    WHERE score < ?
                """,
                    (threshold,),
                )

                deleted_count = cursor.rowcount
                conn.commit()

                # Also vacuum to reclaim space
                conn.execute("VACUUM")

                return deleted_count

        except Exception as e:
            raise DatabaseError(f"Failed to cleanup phrases: {e}")

    def remove_phrases_containing_word(
        self, word: str, case_sensitive: bool = False
    ) -> int:
        """Remove all phrases that contain a specific word.

        Args:
            word: The word to search for
            case_sensitive: Whether the search should be case sensitive

        Returns:
            Number of phrases removed
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if case_sensitive:
                    # Case sensitive search
                    cursor = conn.execute(
                        """
                        DELETE FROM phrases
                        WHERE phrase LIKE ?
                    """,
                        (f"%{word}%",),
                    )
                else:
                    # Case insensitive search
                    cursor = conn.execute(
                        """
                        DELETE FROM phrases
                        WHERE UPPER(phrase) LIKE UPPER(?)
                    """,
                        (f"%{word}%",),
                    )

                deleted_count = cursor.rowcount
                conn.commit()

                # Vacuum to reclaim space if significant deletions
                if deleted_count > 10:
                    conn.execute("VACUUM")

                return deleted_count

        except Exception as e:
            raise DatabaseError(f"Failed to remove phrases containing '{word}': {e}")

    def clear_all_phrases(self) -> None:
        """Delete all phrases from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM phrases")
                conn.commit()
                # Vacuum to reclaim space
                conn.execute("VACUUM")

        except Exception as e:
            raise DatabaseError(f"Failed to clear all phrases: {e}")

    def get_improvable_phrases(
        self,
        limit: int = 10,
        max_failed_attempts: int = 5,
        max_children_created: int = 5,
    ) -> List[GeneratedPhrase]:
        """Get phrases that can still be improved (haven't reached retirement thresholds)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT * FROM phrases
                    WHERE consecutive_failed_improvements < ?
                    AND children_created < ?
                    ORDER BY score DESC, generated_at DESC
                    LIMIT ?
                """,
                    (max_failed_attempts, max_children_created, limit),
                ).fetchall()

                phrases = []
                for row in rows:
                    # Handle backward compatibility for the new columns
                    try:
                        consecutive_failed_improvements = (
                            row["consecutive_failed_improvements"]
                            if "consecutive_failed_improvements" in row.keys()
                            else 0
                        )
                    except (KeyError, IndexError):
                        consecutive_failed_improvements = 0

                    try:
                        children_created = (
                            row["children_created"]
                            if "children_created" in row.keys()
                            else 0
                        )
                        # Fallback to old column name for backward compatibility
                        if (
                            children_created == 0
                            and "total_successful_improvements" in row.keys()
                        ):
                            children_created = row["total_successful_improvements"]
                    except (KeyError, IndexError):
                        children_created = 0

                    phrases.append(
                        GeneratedPhrase(
                            id=row["id"],
                            phrase=row["phrase"],
                            score=row["score"],
                            tiles_used=json.loads(row["tiles_used"]),
                            generated_at=datetime.fromisoformat(row["generated_at"]),
                            model_used=row["model_used"],
                            prompt_context=row["prompt_context"],
                            consecutive_failed_improvements=consecutive_failed_improvements,
                            children_created=children_created,
                        )
                    )

                return phrases

        except Exception as e:
            raise DatabaseError(f"Failed to get improvable phrases: {e}")

    def increment_failed_improvement(self, phrase_id: int):
        """Increment the consecutive failed improvements counter for a phrase."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE phrases
                    SET consecutive_failed_improvements = consecutive_failed_improvements + 1
                    WHERE id = ?
                """,
                    (phrase_id,),
                )
                conn.commit()

        except Exception as e:
            raise DatabaseError(f"Failed to increment failed improvement counter: {e}")

    def reset_failed_improvements(self, phrase_id: int):
        """Reset the consecutive failed improvements counter to 0 for a phrase."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE phrases
                    SET consecutive_failed_improvements = 0
                    WHERE id = ?
                """,
                    (phrase_id,),
                )
                conn.commit()

        except Exception as e:
            raise DatabaseError(f"Failed to reset failed improvement counter: {e}")

    def add_children_created(self, phrase_id: int, count: int = 1):
        """Add to the children created counter for a phrase."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE phrases
                    SET children_created = children_created + ?
                    WHERE id = ?
                """,
                    (count, phrase_id),
                )
                conn.commit()

        except Exception as e:
            raise DatabaseError(f"Failed to increment children created counter: {e}")

    def start_generation_session(self, tiles_input: str) -> int:
        """Start a new generation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO generation_sessions (tiles_input)
                    VALUES (?)
                """,
                    (tiles_input,),
                )
                session_id = cursor.lastrowid
                conn.commit()
                return session_id

        except Exception as e:
            raise DatabaseError(f"Failed to start session: {e}")

    def update_session_stats(
        self,
        session_id: int,
        phrases_generated: int,
        valid_phrases: int,
        top_score: int,
        avg_score: float,
    ):
        """Update generation session statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE generation_sessions
                    SET phrases_generated = phrases_generated + ?,
                        valid_phrases = valid_phrases + ?,
                        top_score = MAX(top_score, ?),
                        avg_score = ?
                    WHERE id = ?
                """,
                    (
                        phrases_generated,
                        valid_phrases,
                        top_score,
                        avg_score,
                        session_id,
                    ),
                )
                conn.commit()

        except Exception as e:
            raise DatabaseError(f"Failed to update session stats: {e}")

    def get_session_stats(self, session_id: int) -> Optional[GenerationSession]:
        """Get statistics for a generation session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    """
                    SELECT * FROM generation_sessions WHERE id = ?
                """,
                    (session_id,),
                ).fetchone()

                if row is None:
                    return None

                return GenerationSession(
                    id=row["id"],
                    session_start=datetime.fromisoformat(row["session_start"]),
                    phrases_generated=row["phrases_generated"],
                    valid_phrases=row["valid_phrases"],
                    top_score=row["top_score"],
                    avg_score=row["avg_score"],
                    tiles_input=row["tiles_input"],
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
                    "total_phrases": phrase_stats[0] if phrase_stats else 0,
                    "max_score": phrase_stats[1] if phrase_stats else 0,
                    "avg_score": round(phrase_stats[2], 2)
                    if phrase_stats and phrase_stats[2]
                    else 0,
                    "first_phrase": phrase_stats[3] if phrase_stats else None,
                    "last_phrase": phrase_stats[4] if phrase_stats else None,
                    "total_sessions": session_stats[0] if session_stats else 0,
                    "total_generated": session_stats[1] if session_stats else 0,
                    "total_valid": session_stats[2] if session_stats else 0,
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
        tiles_used={
            "W": 2,
            "I": 1,
            "N": 4,
            "T": 1,
            "E": 2,
            "R": 3,
            "O": 1,
            "D": 2,
            "L": 2,
            "A": 1,
        },
        model_used="test",
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
