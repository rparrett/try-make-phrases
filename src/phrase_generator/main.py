"""
Main orchestration loop for the Scrabble phrase generator.
"""

import asyncio
import time
# psutil import removed (system health monitoring removed)
import signal
import sys
from datetime import datetime, timedelta
from typing import Optional, List
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


def signal_handler(sig, frame):
    """Handle graceful shutdown on SIGINT."""
    global running
    logger.info("Received shutdown signal, stopping gracefully...")
    running = False


# System health monitoring removed per user request


# Throttling removed per user request


def create_status_display(ranker: PhraseRanker, tiles: TileInventory,
                         session_stats: Optional[GenerationSession] = None) -> Layout:
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

    # Recent Phrases Panel
    try:
        recent_phrases = ranker.get_recent_phrases(12)  # Get up to 12 most recent by date

        recent_text = "Recent Phrases:\n"
        # Already newest first from database, no need to reverse
        for i, phrase in enumerate(recent_phrases, 1):
            # Show time ago for recent phrases
            time_ago = datetime.now() - phrase.generated_at
            if time_ago.total_seconds() < 3600:  # Less than an hour
                time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
            else:
                time_str = f"{int(time_ago.total_seconds() // 3600)}h ago"
            recent_text += f"{i}. {phrase.phrase} ({phrase.score}) - {time_str}\n"

    except Exception as e:
        recent_text = f"Recent phrases unavailable: {e}"

    # Session Stats Panel
    if session_stats:
        runtime = datetime.now() - session_stats.session_start
        overall_attempts = session_stats.fresh_attempts + session_stats.improvement_attempts
        overall_successes = session_stats.fresh_successes + session_stats.improvement_successes

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

        overall_text = f"""Total Phrases: {stats['total_phrases']} | Max Score: {stats['max_score']}

{total_tiles} tiles ({unique_tiles} unique)
"""
        overall_text += ", ".join([f"[cyan]{count}[/cyan][green]{letter}[/green]" if count > 1 else f"[green]{letter}[/green]"
                                 for letter, count in sorted_tiles])
    except Exception as e:
        overall_text = f"Overall stats unavailable: {e}"

    layout.split_column(
        Layout(Panel(top_text, title="Top Phrases"), ratio=4),
        Layout(Panel(recent_text, title="Recent Phrases"), ratio=4),
        Layout(Panel(session_text, title="Current Session"), ratio=4),
        Layout(Panel(overall_text, title="Overall"), ratio=2)
    )

    return layout


async def generation_cycle(tiles: TileInventory, llm_client: OllamaClient,
                          ranker: PhraseRanker, config: OptimizationConfig,
                          console: Console, iteration: int, generation_session: GenerationSession) -> int:
    """
    Run a single generation cycle (either fresh generation or improvement).

    Args:
        iteration: Current iteration number (used to decide between fresh/improvement)

    Returns:
        Number of valid phrases generated
    """
    try:
        # Get top phrases for potential improvement
        top_phrases = ranker.get_top_phrases(5)

        # Decide whether to do fresh generation or improvement
        # After we have some phrases (3+), focus more on improvement since we now improve one phrase at a time
        should_do_fresh = (
            len(top_phrases) < 3 or    # Need more phrases to improve
            iteration <= 2 or          # Bootstrap with fresh generation first
            iteration % 3 == 0         # Every 3rd iteration, do fresh generation
        )

        should_improve = not should_do_fresh

        if should_improve:
            # Improvement cycle: focus on single phrase with multiple attempts
            improvement_strategy = iteration % 9  # Cycle through different strategies

            if improvement_strategy < 3:
                # Strategy 1: Improve top-scoring phrases (33% of time)
                base_phrase_object = top_phrases[0] if top_phrases else None
                strategy_name = "top-scoring"
            elif improvement_strategy < 6:
                # Strategy 2: Improve recent phrases (33% of time)
                recent_phrases = ranker.get_recent_phrases(3)  # Get truly recent phrases by date
                base_phrase_object = recent_phrases[0] if recent_phrases else None
                strategy_name = "recent"
            else:
                # Strategy 3: Improve mixed/diverse phrases (33% of time)
                import random

                # Get a mix and randomly select one
                top_phrases_for_mix = ranker.get_top_phrases(5)
                recent_phrases_for_mix = ranker.get_recent_phrases(5)
                all_phrases_for_mix = ranker.get_phrase_history(20)  # Still score-based for broader pool

                candidates = []
                candidates.extend(top_phrases_for_mix[:2])  # Top 2 scorers
                candidates.extend(recent_phrases_for_mix[:2])  # 2 recent
                candidates.extend(all_phrases_for_mix[5:10])  # Some mid-range phrases

                # Remove duplicates by ID
                seen_ids = set()
                unique_candidates = []
                for phrase in candidates:
                    if phrase.id not in seen_ids:
                        unique_candidates.append(phrase)
                        seen_ids.add(phrase.id)

                base_phrase_object = random.choice(unique_candidates) if unique_candidates else None
                strategy_name = "mixed"

            if not base_phrase_object:
                logger.warning("No phrase available for improvement")
                return 0

            # Extract phrase and score for improvement tracking
            base_phrase = base_phrase_object.phrase
            original_score = base_phrase_object.score

            logger.debug(f"Improving {strategy_name} phrase: '{base_phrase}' (score: {original_score})")

            # Generate multiple attempts to improve this single phrase
            phrase_candidates = llm_client.improve_single_phrase(
                base_phrase=base_phrase,
                tiles=tiles,
                num_attempts=config.generation_batch_size
            )

            # For statistics tracking, we need the original score repeated for each attempt
            original_scores = [original_score] * len(phrase_candidates) if phrase_candidates else [original_score]

            cycle_type = f"improvement-{strategy_name}"
        else:
            # Fresh generation cycle: create new phrases
            context_phrases = ranker.get_context_phrases()
            logger.debug(f"Generating {config.generation_batch_size} fresh phrases...")

            phrase_candidates = llm_client.generate_phrases(
                tiles=tiles,
                context_phrases=context_phrases,
                batch_size=config.generation_batch_size
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
                    improved_scores=[]
                )
            else:
                generation_session.update_fresh_stats(
                    attempts=config.generation_batch_size,
                    successes=0,
                    scores=[]
                )
            return 0

        # Add to ranker (validates, scores, and stores)
        added_phrases = ranker.add_phrase_candidates(
            phrase_candidates,
            tiles,
            llm_client.model_name,
            f"{cycle_type.title()} generation"
        )

        # Track statistics
        successful_scores = [phrase.score for phrase in added_phrases]

        if should_improve:
            # Track improvement statistics
            improved_scores = successful_scores if successful_scores else [0] * len(original_scores)
            generation_session.update_improvement_stats(
                attempts=config.generation_batch_size,
                successes=len(added_phrases),
                scores=successful_scores,
                original_scores=original_scores,
                improved_scores=improved_scores
            )
        else:
            # Track fresh generation statistics
            generation_session.update_fresh_stats(
                attempts=config.generation_batch_size,
                successes=len(added_phrases),
                scores=successful_scores
            )

        logger.info(f"Generated {len(phrase_candidates)} {cycle_type} candidates, "
                   f"added {len(added_phrases)} valid phrases")

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


async def main_generation_loop(tiles_input: str, config: OptimizationConfig,
                             console: Console, db_path: str = "data/phrases.db",
                             model_name: str = "llama2:7b", skip_recalc: bool = False):
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

                recalc_stats = db.recalculate_scores_for_tileset(tiles, scorer, validator)
                logger.info(f"Score recalculation completed: "
                           f"{recalc_stats['updated_count']} phrases updated, "
                           f"{recalc_stats['removed_count']} phrases removed (unbuildable)")

                if recalc_stats['updated_count'] > 0:
                    console.print(f"[green]Recalculated scores for {recalc_stats['updated_count']} phrases[/green]")
                if recalc_stats['removed_count'] > 0:
                    console.print(f"[yellow]Removed {recalc_stats['removed_count']} unbuildable phrases[/yellow]")

            except Exception as e:
                logger.warning(f"Failed to recalculate scores: {e}")
                console.print(f"[yellow]Warning: Could not recalculate phrase scores: {e}[/yellow]")
        else:
            logger.info("Skipping phrase score recalculation (--skip-recalc flag set)")
            console.print("[blue]Skipping phrase score recalculation[/blue]")

        # Start session tracking
        session_id = db.start_generation_session(tiles_input)
        generation_session = GenerationSession(
            id=session_id,
            tiles_input=tiles_input
        )

        logger.info(f"Started generation session {session_id}")
        logger.info(f"Using model: {llm_client.model_name}")

        # Main generation loop
        iteration = 0
        last_status_update = time.time()

        with Live(create_status_display(ranker, tiles, generation_session),
                  console=console, refresh_per_second=0.5) as live:

            while running:
                iteration += 1
                cycle_start = time.time()

# System health checking removed per user request

                # Update display every 10 seconds
                if time.time() - last_status_update > 10:
                    live.update(create_status_display(ranker, tiles, generation_session))
                    last_status_update = time.time()

# Throttling removed per user request

                # Run generation cycle (fresh or improvement)
                valid_phrases = await generation_cycle(tiles, llm_client, ranker, config, console, iteration, generation_session)

                if valid_phrases > 0:
                    top_phrases = ranker.get_top_phrases(1)
                    if top_phrases:
                        generation_session.top_score = max(generation_session.top_score,
                                                         top_phrases[0].score)

                # Calculate timing for next iteration
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, config.iteration_delay_seconds - cycle_time)

                logger.debug(f"Iteration {iteration} completed in {cycle_time:.1f}s, "
                           f"sleeping for {sleep_time:.1f}s")

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
                            generation_session.avg_score
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
                    generation_session.avg_score
                )
                logger.info(f"Session completed: {generation_session.valid_phrases} valid phrases "
                           f"from {generation_session.phrases_generated} generated")
            except Exception as e:
                logger.warning(f"Failed final session update: {e}")

    return True


# CLI Application
app = typer.Typer(help="Scrabble Phrase Generator - Generate wintery phrases from tiles")


@app.command()
def generate(
    tiles: str = typer.Argument(..., help="Tiles in format like '9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh'"),
    db_path: str = typer.Option("data/phrases.db", "--db", "-d", help="Database path"),
    batch_size: int = typer.Option(15, "--batch", "-b", help="Phrases per generation cycle"),
    delay: int = typer.Option(30, "--delay", help="Seconds between cycles"),
    min_score: int = typer.Option(10, "--min-score", help="Minimum score threshold"),
    max_phrases: int = typer.Option(1000, "--max-phrases", help="Maximum phrases to keep"),
    model: str = typer.Option("llama2:7b", "--model", "-m", help="Ollama model to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    skip_recalc: bool = typer.Option(False, "--skip-recalc", help="Skip phrase score recalculation")
):
    """Start continuous phrase generation."""
    console = Console()

    # Configure logging
    logger.remove()  # Remove default handler
    log_level = "DEBUG" if verbose else "ERROR"  # Only show errors on screen unless verbose

    # Only show logs on screen if in verbose mode, otherwise just log to file
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="<green>{time}</green> | <level>{level}</level> | {message}")
    else:
        # In non-verbose mode, only show critical errors on screen
        logger.add(sys.stderr, level="ERROR", format="ERROR: {message}")

    # Always log detailed info to file
    logger.add("logs/generator.log", rotation="1 day", retention="7 days", level="DEBUG")

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure separate logger for raw LLM interactions (prompts + responses)
    from loguru import logger as llm_logger
    llm_logger.add("logs/llm_interactions.log",
                   rotation="1 day",
                   retention="7 days",
                   level="DEBUG",
                   filter=lambda record: any(phrase in record["message"] for phrase in [
                       "Raw LLM response",
                       "Fresh generation prompt",
                       "Single phrase improvement prompt"
                   ]),
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    # Configure standard Python logging to not interfere with TUI
    import logging
    if not verbose:
        logging.getLogger().setLevel(logging.ERROR)  # Suppress most standard logging

    config = OptimizationConfig(
        generation_batch_size=batch_size,
        iteration_delay_seconds=delay,
        min_score_threshold=min_score,
        max_phrases_stored=max_phrases
    )

    console.print(Panel.fit(
        f"Starting Scrabble Phrase Generator\n"
        f"Tiles: {tiles}\n"
        f"Model: {model}\n"
        f"Batch size: {batch_size}\n"
        f"Delay: {delay}s\n"
        f"Min score: {min_score}\n"
        f"Database: {db_path}",
        title="Configuration"
    ))

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Run the generation loop
    try:
        result = asyncio.run(main_generation_loop(tiles, config, console, db_path, model, skip_recalc))
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
    top: int = typer.Option(10, "--top", "-t", help="Number of top phrases to show")
):
    """Show current status and top phrases."""
    console = Console()

    try:
        ranker = PhraseRanker(db_path)
        stats = ranker.get_ranking_stats()

        # Overall stats
        console.print(Panel.fit(
            f"Total phrases: {stats['total_phrases']}\n"
            f"Max score: {stats['max_score']}\n"
            f"Average score: {stats['avg_score']:.1f}",
            title="Database Statistics"
        ))

        # Top phrases
        top_phrases = ranker.get_top_phrases(top)
        if top_phrases:
            table = Table(title=f"Top {len(top_phrases)} Phrases")
            table.add_column("Rank", style="cyan")
            table.add_column("Phrase", style="green")
            table.add_column("Score", style="yellow")
            table.add_column("Generated", style="blue")

            for i, phrase in enumerate(top_phrases, 1):
                table.add_row(
                    str(i),
                    phrase.phrase,
                    str(phrase.score),
                    phrase.generated_at.strftime("%m/%d %H:%M")
                )

            console.print(table)
        else:
            console.print("[yellow]No phrases found in database[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()