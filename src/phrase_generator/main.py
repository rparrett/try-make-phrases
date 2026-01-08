"""
Main orchestration loop for the Scrabble phrase generator.
"""

import asyncio
import time

# psutil import removed (system health monitoring removed)
import signal
import sys
import sqlite3
from datetime import datetime
from typing import Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from loguru import logger

from storage.models import TileInventory, OptimizationConfig, GenerationSession
from storage.database import PhraseDatabase
from src.phrase_generator.tile_parser import parse_tile_string, TileParseError
from src.phrase_generator.llm_client import OllamaClient, LLMError
from src.phrase_generator.phrase_ranker import PhraseRanker, RankingError

# Global state for graceful shutdown
running = True
generation_session: Optional[GenerationSession] = None

# Track recent phrase generation attempts for TUI display
recent_phrase_attempts = []

# Strategy constants
MAX_FAILED_IMPROVEMENT_ATTEMPTS = 5
MAX_CHILDREN_CREATED = 5


def add_phrase_attempt(phrase: str, score: int, accepted: bool, reason: str = ""):
    """Track a phrase generation attempt for TUI display."""
    global recent_phrase_attempts
    attempt = {
        "phrase": phrase,
        "score": score,
        "accepted": accepted,
        "reason": reason,
        "timestamp": datetime.now(),
    }
    recent_phrase_attempts.append(attempt)

    # Keep only last 15 attempts
    if len(recent_phrase_attempts) > 15:
        recent_phrase_attempts.pop(0)


def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT."""
    global running
    logger.info("Received shutdown signal, stopping gracefully...")
    running = False


# System health monitoring removed per user request


# Throttling removed per user request


def create_status_display(
    ranker: PhraseRanker,
    tiles: TileInventory,
    session_stats: Optional[GenerationSession] = None,
) -> Layout:
    """Create a rich layout for status display."""
    layout = Layout()

    # Top Phrases Panel
    try:
        stats = ranker.get_ranking_stats()
        top_phrases = ranker.get_top_phrases(12)  # Show up to 12 top phrases

        top_text = "Top Phrases:\n"
        for i, phrase in enumerate(top_phrases, 1):
            top_text += f"{i}. {phrase.phrase} ({phrase.score})\n"

    except Exception as e:
        top_text = f"Top phrases unavailable: {e}"

    # Recent Phrase Attempts Panel
    global recent_phrase_attempts
    try:
        if recent_phrase_attempts:
            recent_text = "Recent Attempts (✓ accepted, ✗ rejected):\n"
        else:
            recent_text = "No recent attempts yet"

        # Show last 12 attempts in reverse order (newest first)
        recent_attempts = recent_phrase_attempts[-12:][::-1]
        for i, attempt in enumerate(recent_attempts, 1):
            # Show time ago
            time_ago = datetime.now() - attempt["timestamp"]
            if time_ago.total_seconds() < 3600:  # Less than an hour
                time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() // 3600)}h ago"

            # Color coding: green for accepted, red for rejected
            if attempt["accepted"]:
                status_icon = "[green]✓[/green]"
                phrase_color = "[white]"
            else:
                status_icon = "[red]✗[/red]"
                phrase_color = "[dim white]"  # Dimmer for rejected

            reason = f" ({attempt['reason']})" if attempt["reason"] else ""
            recent_text += f"{i}. {status_icon} {phrase_color}{attempt['phrase']}[/] ({attempt['score']}) - {time_str}{reason}\n"

    except Exception as e:
        recent_text = f"Recent attempts unavailable: {e}"

    # Session Stats Panel
    if session_stats:
        runtime = datetime.now() - session_stats.session_start
        overall_attempts = (
            session_stats.fresh_attempts + session_stats.improvement_attempts
        )
        overall_successes = (
            session_stats.fresh_successes + session_stats.improvement_successes
        )

        session_text = f"""Runtime: {runtime}
Generated: {overall_attempts} attempts, {overall_successes} valid

Fresh Generation:
  Attempts: {session_stats.fresh_attempts}
  Successes: {session_stats.fresh_successes}
  Rate: {session_stats.get_fresh_success_rate():.1f}%
  Avg Score: {session_stats.fresh_avg_score:.1f}

Improvements:
  Attempts: {session_stats.improvement_attempts}
  Successes: {session_stats.improvement_successes}
  Rate: {session_stats.get_improvement_success_rate():.1f}%
  Avg Score: {session_stats.improvement_avg_score:.1f}
  Better than Original: {session_stats.get_improvement_effectiveness():.1f}%
"""
    else:
        session_text = "Session not started"

    # Overall Panel (combines stats and tiles)
    try:
        sorted_tiles = sorted(tiles.tiles.items())
        total_tiles = sum(tiles.tiles.values())
        unique_tiles = len(sorted_tiles)

        overall_text = f"""Total Phrases: {stats["total_phrases"]} | Max Score: {stats["max_score"]}

{total_tiles} tiles ({unique_tiles} unique)
"""
        overall_text += ", ".join(
            [
                f"[cyan]{count}[/cyan][green]{letter}[/green]"
                if count > 1
                else f"[green]{letter}[/green]"
                for letter, count in sorted_tiles
            ]
        )
    except Exception as e:
        overall_text = f"Overall stats unavailable: {e}"

    layout.split_column(
        Layout(Panel(top_text, title="Top Phrases"), ratio=4),
        Layout(Panel(recent_text, title="Recent Phrases"), ratio=4),
        Layout(Panel(session_text, title="Current Session"), ratio=4),
        Layout(Panel(overall_text, title="Overall"), ratio=2),
    )

    return layout


async def generation_cycle(
    tiles: TileInventory,
    llm_client: OllamaClient,
    ranker: PhraseRanker,
    config: OptimizationConfig,
    console: Console,
    iteration: int,
    generation_session: GenerationSession,
) -> int:
    """
    Run a single generation cycle (either fresh generation or improvement).

    Args:
        iteration: Current iteration number (used to decide between fresh/improvement)

    Returns:
        Number of valid phrases generated
    """
    try:
        # New strategy: Check for improvable phrases first
        improvable_phrases = ranker.get_improvable_phrases(
            999, MAX_FAILED_IMPROVEMENT_ATTEMPTS, MAX_CHILDREN_CREATED
        )

        # Only do fresh generation if no phrases are improvable
        should_improve = len(improvable_phrases) > 0

        if should_improve:
            strategy_names = [
                "top-score",
                "weighted-random",
                "top-5-random",
                "pure-random",
            ]
            next_strategy = strategy_names[iteration % 4]
            logger.debug(
                f"Found {len(improvable_phrases)} improvable phrases. "
                f"Strategy: improvement [{next_strategy}]"
            )
        else:
            logger.debug(
                f"Found {len(improvable_phrases)} improvable phrases. "
                f"Strategy: fresh generation"
            )

        if should_improve:
            # Improvement cycle: Select phrase with weighted randomness
            import random

            if not improvable_phrases:
                logger.warning("No improvable phrase available")
                return 0

            # Show top candidates for visibility
            top_3 = improvable_phrases[: min(3, len(improvable_phrases))]
            candidate_info = [f"'{p.phrase}' ({p.score})" for p in top_3]
            logger.debug(f"Top improvable candidates: {', '.join(candidate_info)}")

            # Weighted random selection - higher scores get better odds, but not guaranteed
            # Use strategy based on iteration to add variety
            strategy = iteration % 3

            if strategy == 0:
                # 33% of time: Pure top score (deterministic)
                base_phrase_object = improvable_phrases[0]
                selection_method = "top-score"
            elif strategy == 1:
                # 33% of time: Weighted random (score-based probabilities)
                scores = [phrase.score for phrase in improvable_phrases]
                min_score = min(scores)
                # Shift scores to be positive and add 1 to avoid zero weights
                weights = [score - min_score + 1 for score in scores]
                base_phrase_object = random.choices(
                    improvable_phrases, weights=weights
                )[0]
                selection_method = "weighted-random"
            else:
                # 33% of time: Pure random
                base_phrase_object = random.choice(improvable_phrases)
                selection_method = "pure-random"

            # Extract phrase and score for improvement tracking
            base_phrase = base_phrase_object.phrase
            original_score = base_phrase_object.score
            base_phrase_id = base_phrase_object.id

            logger.debug(
                f"Improving phrase: '{base_phrase}' (score: {original_score}, "
                f"failed: {base_phrase_object.consecutive_failed_improvements}, "
                f"children: {base_phrase_object.children_created}) "
                f"[{selection_method}]"
            )

            # Generate multiple attempts to improve this single phrase
            phrase_candidates = llm_client.improve_single_phrase(
                base_phrase=base_phrase,
                tiles=tiles,
                num_attempts=config.generation_batch_size,
            )

            # For statistics tracking, we need the original score repeated for each attempt
            original_scores = (
                [original_score] * len(phrase_candidates)
                if phrase_candidates
                else [original_score]
            )

            cycle_type = "improvement"
        else:
            # Fresh generation cycle: create new phrases
            context_phrases = ranker.get_context_phrases()
            logger.debug(f"Generating {config.generation_batch_size} fresh phrases...")

            phrase_candidates = llm_client.generate_phrases(
                tiles=tiles,
                context_phrases=context_phrases,
                batch_size=config.generation_batch_size,
            )

            cycle_type = "fresh"
            original_scores = []  # No originals for fresh generation

        if not phrase_candidates:
            logger.warning(f"No phrases generated from LLM ({cycle_type} cycle)")

            # Track failed attempts
            if should_improve:
                generation_session.update_improvement_stats(
                    attempts=config.generation_batch_size,
                    successes=0,
                    scores=[],
                    original_scores=original_scores,
                    improved_scores=[],
                )
            else:
                generation_session.update_fresh_stats(
                    attempts=config.generation_batch_size, successes=0, scores=[]
                )
            return 0

        # Add to ranker (validates, scores, and stores)
        if should_improve:
            # For improvement cycles, pre-filter candidates to only include those better than parent
            logger.debug(
                f"Pre-filtering {len(phrase_candidates)} improvement candidates against parent score {original_score}"
            )

            # First, validate and score the candidates without adding them
            temp_phrases = []
            for phrase in phrase_candidates:
                is_valid, tiles_used, error = ranker.validator.validate_phrase(
                    phrase, tiles
                )
                if is_valid:
                    score = ranker.scorer.score_phrase_simple(phrase, tiles_used)

                    if score >= ranker.config.min_score_threshold:
                        if score > original_score:
                            # Phrase beats parent - accept it
                            add_phrase_attempt(phrase, score, True, "beats parent")
                            generated_phrase = ranker.scorer.create_scored_phrase(
                                phrase,
                                tiles,
                                tiles_used,
                                llm_client.model_name,
                                f"{cycle_type.title()} generation",
                            )
                            temp_phrases.append(generated_phrase)
                        else:
                            # Phrase doesn't beat parent - reject it
                            add_phrase_attempt(
                                phrase, score, False, f"≤ parent ({original_score})"
                            )
                    else:
                        # Below minimum threshold - reject it
                        add_phrase_attempt(
                            phrase,
                            score,
                            False,
                            f"< min ({ranker.config.min_score_threshold})",
                        )
                else:
                    # Invalid phrase - track it too
                    add_phrase_attempt(phrase, 0, False, "invalid")

            # Add only the filtered (better) phrases to database
            if temp_phrases:
                phrase_ids = ranker.db.add_phrases_batch(temp_phrases)
                added_phrases = []
                for phrase, phrase_id in zip(temp_phrases, phrase_ids):
                    if phrase_id:  # Some might be None due to duplicates
                        phrase.id = phrase_id
                        added_phrases.append(phrase)
                logger.info(
                    f"Added {len(added_phrases)} improved phrases to database (filtered from {len(phrase_candidates)} candidates)"
                )
            else:
                added_phrases = []
                logger.debug("No improvement candidates beat the parent score")
        else:
            # For fresh generation, track all attempts before adding to database
            temp_phrases = []
            for phrase in phrase_candidates:
                is_valid, tiles_used, error = ranker.validator.validate_phrase(
                    phrase, tiles
                )
                if is_valid:
                    score = ranker.scorer.score_phrase_simple(phrase, tiles_used)
                    if score >= ranker.config.min_score_threshold:
                        # Fresh phrase meets threshold - accept it
                        add_phrase_attempt(phrase, score, True, "fresh gen")
                        generated_phrase = ranker.scorer.create_scored_phrase(
                            phrase,
                            tiles,
                            tiles_used,
                            llm_client.model_name,
                            f"{cycle_type.title()} generation",
                        )
                        temp_phrases.append(generated_phrase)
                    else:
                        # Below minimum threshold - reject it
                        add_phrase_attempt(
                            phrase,
                            score,
                            False,
                            f"< min ({ranker.config.min_score_threshold})",
                        )
                else:
                    # Invalid phrase - track it
                    add_phrase_attempt(phrase, 0, False, "invalid")

            # Add the valid phrases to database
            added_phrases = []
            if temp_phrases:
                phrase_ids = ranker.db.add_phrases_batch(temp_phrases)
                for phrase, phrase_id in zip(temp_phrases, phrase_ids):
                    if phrase_id:  # Some might be None due to duplicates
                        phrase.id = phrase_id
                        added_phrases.append(phrase)

        # Track statistics
        successful_scores = [phrase.score for phrase in added_phrases]

        if should_improve:
            # Track improvement statistics
            improved_scores = (
                successful_scores if successful_scores else [0] * len(original_scores)
            )
            generation_session.update_improvement_stats(
                attempts=config.generation_batch_size,
                successes=len(added_phrases),
                scores=successful_scores,
                original_scores=original_scores,
                improved_scores=improved_scores,
            )

            # Track improvement success/failure for consecutive attempts strategy
            if added_phrases:
                # Since we pre-filtered, all added_phrases are guaranteed to be better than parent
                children_count = len(added_phrases)
                ranker.mark_improvement_success(base_phrase_id, children_count)
                logger.debug(
                    f"Improvement success! Created {children_count} better children from '{base_phrase}' "
                    f"(scores: {[p.score for p in added_phrases]} vs original {original_score})"
                )
            else:
                # No better phrases generated, mark as failure
                ranker.mark_improvement_failure(base_phrase_id)
                logger.debug(
                    f"Improvement failed - no phrases generated from '{base_phrase}' beat original score {original_score}"
                )
        else:
            # Track fresh generation statistics
            generation_session.update_fresh_stats(
                attempts=config.generation_batch_size,
                successes=len(added_phrases),
                scores=successful_scores,
            )

        logger.info(
            f"Generated {len(phrase_candidates)} {cycle_type} candidates, "
            f"added {len(added_phrases)} valid phrases"
        )

        return len(added_phrases)

    except LLMError as e:
        logger.error(f"LLM generation failed: {e}")
        return 0
    except RankingError as e:
        logger.error(f"Ranking failed: {e}")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error in generation cycle: {e}")
        return 0


async def main_generation_loop(
    tiles_input: str,
    config: OptimizationConfig,
    console: Console,
    db_path: str = "data/phrases.db",
    model_name: str = "llama2:7b",
    skip_recalc: bool = False,
):
    """
    Main continuous generation loop with MacBook optimizations.
    """
    global running, generation_session

    # Reset running state
    running = True

    try:
        # Parse tiles
        tiles = parse_tile_string(tiles_input)
        logger.info(f"Parsed {sum(tiles.tiles.values())} tiles: {tiles.tiles}")

        # Initialize components
        llm_client = OllamaClient(model_name)
        ranker = PhraseRanker(db_path, config)
        db = PhraseDatabase(db_path)

        # Recalculate existing phrase scores for current tile set
        if not skip_recalc:
            logger.info("Recalculating phrase scores for current tile set...")
            try:
                from src.phrase_generator.scorer import ScrabbleScorer
                from src.phrase_generator.phrase_validator import PhraseValidator

                scorer = ScrabbleScorer()
                validator = PhraseValidator()

                recalc_stats = db.recalculate_scores_for_tileset(
                    tiles, scorer, validator
                )
                logger.info(
                    f"Score recalculation completed: "
                    f"{recalc_stats['updated_count']} phrases updated, "
                    f"{recalc_stats['removed_count']} phrases removed (unbuildable)"
                )

                if recalc_stats["updated_count"] > 0:
                    console.print(
                        f"[green]Recalculated scores for {recalc_stats['updated_count']} phrases[/green]"
                    )
                if recalc_stats["removed_count"] > 0:
                    console.print(
                        f"[yellow]Removed {recalc_stats['removed_count']} unbuildable phrases[/yellow]"
                    )

            except Exception as e:
                logger.warning(f"Failed to recalculate scores: {e}")
                console.print(
                    f"[yellow]Warning: Could not recalculate phrase scores: {e}[/yellow]"
                )
        else:
            logger.info("Skipping phrase score recalculation (--skip-recalc flag set)")
            console.print("[blue]Skipping phrase score recalculation[/blue]")

        # Start session tracking
        session_id = db.start_generation_session(tiles_input)
        generation_session = GenerationSession(id=session_id, tiles_input=tiles_input)

        logger.info(f"Started generation session {session_id}")
        logger.info(f"Using model: {llm_client.model_name}")

        # Main generation loop
        iteration = 0
        last_status_update = time.time()

        with Live(
            create_status_display(ranker, tiles, generation_session),
            console=console,
            refresh_per_second=0.5,
        ) as live:
            while running:
                iteration += 1
                cycle_start = time.time()

                # System health checking removed per user request

                # Update display every 10 seconds
                if time.time() - last_status_update > 10:
                    live.update(
                        create_status_display(ranker, tiles, generation_session)
                    )
                    last_status_update = time.time()

                # Throttling removed per user request

                # Run generation cycle (fresh or improvement)
                valid_phrases = await generation_cycle(
                    tiles,
                    llm_client,
                    ranker,
                    config,
                    console,
                    iteration,
                    generation_session,
                )

                if valid_phrases > 0:
                    top_phrases = ranker.get_top_phrases(1)
                    if top_phrases:
                        generation_session.top_score = max(
                            generation_session.top_score, top_phrases[0].score
                        )

                # Calculate timing for next iteration
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, config.iteration_delay_seconds - cycle_time)

                logger.debug(
                    f"Iteration {iteration} completed in {cycle_time:.1f}s, "
                    f"sleeping for {sleep_time:.1f}s"
                )

                # Sleep until next iteration
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Periodic session stats update (every 10 iterations)
                if iteration % 10 == 0:
                    try:
                        db.update_session_stats(
                            session_id,
                            generation_session.phrases_generated,
                            generation_session.valid_phrases,
                            generation_session.top_score,
                            generation_session.avg_score,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update session stats: {e}")

    except TileParseError as e:
        console.print(f"[red]Invalid tile format: {e}[/red]")
        return False
    except LLMError as e:
        console.print(f"[red]LLM connection failed: {e}[/red]")
        return False
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        console.print(f"[red]Error: {e}[/red]")
        return False
    finally:
        # Final session update
        if generation_session:
            try:
                db.update_session_stats(
                    session_id,
                    generation_session.phrases_generated,
                    generation_session.valid_phrases,
                    generation_session.top_score,
                    generation_session.avg_score,
                )
                logger.info(
                    f"Session completed: {generation_session.valid_phrases} valid phrases "
                    f"from {generation_session.phrases_generated} generated"
                )
            except Exception as e:
                logger.warning(f"Failed final session update: {e}")

    return True


# CLI Application
app = typer.Typer(
    help="Scrabble Phrase Generator - Generate wintery phrases from tiles"
)


@app.command()
def generate(
    tiles: str = typer.Argument(
        ..., help="Tiles in format like '9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh'"
    ),
    db_path: str = typer.Option("data/phrases.db", "--db", "-d", help="Database path"),
    batch_size: int = typer.Option(
        15, "--batch", "-b", help="Phrases per generation cycle"
    ),
    delay: int = typer.Option(30, "--delay", help="Seconds between cycles"),
    min_score: int = typer.Option(10, "--min-score", help="Minimum score threshold"),
    max_phrases: int = typer.Option(
        1000, "--max-phrases", help="Maximum phrases to keep"
    ),
    model: str = typer.Option("llama2:7b", "--model", "-m", help="Ollama model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    skip_recalc: bool = typer.Option(
        False, "--skip-recalc", help="Skip phrase score recalculation"
    ),
):
    """Start continuous phrase generation."""
    console = Console()

    # Configure logging
    logger.remove()  # Remove default handler

    # Only show logs on screen if in verbose mode, otherwise just log to file
    if verbose:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time}</green> | <level>{level}</level> | {message}",
        )
    else:
        # In non-verbose mode, only show critical errors on screen
        logger.add(sys.stderr, level="ERROR", format="ERROR: {message}")

    # Always log detailed info to file
    logger.add(
        "logs/generator.log", rotation="1 day", retention="7 days", level="DEBUG"
    )

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure separate logger for raw LLM interactions (prompts + responses)
    from loguru import logger as llm_logger

    llm_logger.add(
        "logs/llm_interactions.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        filter=lambda record: any(
            phrase in record["message"]
            for phrase in [
                "Raw LLM response",
                "Fresh generation prompt",
                "Single phrase improvement prompt",
            ]
        ),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    # Configure standard Python logging to not interfere with TUI
    import logging

    if not verbose:
        logging.getLogger().setLevel(logging.ERROR)  # Suppress most standard logging

    config = OptimizationConfig(
        generation_batch_size=batch_size,
        iteration_delay_seconds=delay,
        min_score_threshold=min_score,
        max_phrases_stored=max_phrases,
    )

    console.print(
        Panel.fit(
            f"Starting Scrabble Phrase Generator\n"
            f"Tiles: {tiles}\n"
            f"Model: {model}\n"
            f"Batch size: {batch_size}\n"
            f"Delay: {delay}s\n"
            f"Min score: {min_score}\n"
            f"Database: {db_path}",
            title="Configuration",
        )
    )

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Run the generation loop
    try:
        result = asyncio.run(
            main_generation_loop(tiles, config, console, db_path, model, skip_recalc)
        )
        if result:
            console.print("[green]Generation completed successfully[/green]")
        else:
            console.print("[red]Generation failed[/red]")
            raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped by user[/yellow]")
        global running
        running = False
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    db_path: str = typer.Option("data/phrases.db", "--db", "-d", help="Database path"),
    top: int = typer.Option(10, "--top", "-t", help="Number of top phrases to show"),
):
    """Show current status and top phrases."""
    console = Console()

    try:
        ranker = PhraseRanker(db_path)
        stats = ranker.get_ranking_stats()

        # Overall stats
        console.print(
            Panel.fit(
                f"Total phrases: {stats['total_phrases']}\n"
                f"Max score: {stats['max_score']}\n"
                f"Average score: {stats['avg_score']:.1f}",
                title="Database Statistics",
            )
        )

        # Top phrases
        top_phrases = ranker.get_top_phrases(top)
        if top_phrases:
            table = Table(title=f"Top {len(top_phrases)} Phrases")
            table.add_column("Rank", style="cyan")
            table.add_column("Phrase", style="green")
            table.add_column("Score", style="yellow")
            table.add_column("Failed", style="red")
            table.add_column("Children", style="bright_green")
            table.add_column("Status", style="magenta")
            table.add_column("Generated", style="blue")

            for i, phrase in enumerate(top_phrases, 1):
                # Color-code failed attempts: red if high, yellow if medium, green if low
                failed_attempts = phrase.consecutive_failed_improvements
                children_created = phrase.children_created

                if failed_attempts >= 4:
                    failed_color = "[red]"
                elif failed_attempts >= 2:
                    failed_color = "[yellow]"
                else:
                    failed_color = "[green]"

                # Determine retirement status
                if failed_attempts >= MAX_FAILED_IMPROVEMENT_ATTEMPTS:
                    status = "[red]RETIRED-F[/red]"  # Retired due to failures
                elif children_created >= MAX_CHILDREN_CREATED:
                    status = (
                        "[yellow]RETIRED-C[/yellow]"  # Retired due to children created
                    )
                else:
                    status = "[green]ACTIVE[/green]"

                table.add_row(
                    str(i),
                    phrase.phrase,
                    str(phrase.score),
                    f"{failed_color}{failed_attempts}[/{failed_color[1:]}",
                    f"[bright_green]{children_created}[/bright_green]",
                    status,
                    phrase.generated_at.strftime("%m/%d %H:%M"),
                )

            console.print(table)
        else:
            console.print("[yellow]No phrases found in database[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def remove_word(
    word: str = typer.Argument(..., help="Word to search for and remove from phrases"),
    db_path: str = typer.Option("data/phrases.db", "--db", "-d", help="Database path"),
    case_sensitive: bool = typer.Option(
        False, "--case-sensitive", "-c", help="Case sensitive search"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be removed without actually deleting"
    ),
):
    """Remove all phrases containing a specific word."""
    console = Console()

    try:
        db = PhraseDatabase(db_path)

        if dry_run:
            # Show what would be removed
            with sqlite3.connect(db_path) as conn:
                if case_sensitive:
                    cursor = conn.execute(
                        "SELECT phrase FROM phrases WHERE phrase LIKE ?", (f"%{word}%",)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT phrase FROM phrases WHERE UPPER(phrase) LIKE UPPER(?)",
                        (f"%{word}%",),
                    )

                matching_phrases = cursor.fetchall()

            if matching_phrases:
                console.print(
                    f"[yellow]DRY RUN: Would remove {len(matching_phrases)} phrases containing '{word}':[/yellow]"
                )
                for phrase_row in matching_phrases[:10]:  # Show first 10
                    console.print(f"  - {phrase_row[0]}")
                if len(matching_phrases) > 10:
                    console.print(f"  ... and {len(matching_phrases) - 10} more")
                console.print(
                    "\n[yellow]Run without --dry-run to actually delete these phrases.[/yellow]"
                )
            else:
                console.print(f"[green]No phrases found containing '{word}'[/green]")
        else:
            # Actually remove the phrases
            deleted_count = db.remove_phrases_containing_word(word, case_sensitive)

            if deleted_count > 0:
                console.print(
                    f"[green]Successfully removed {deleted_count} phrases containing '{word}'[/green]"
                )
            else:
                console.print(f"[yellow]No phrases found containing '{word}'[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
