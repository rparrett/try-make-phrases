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

    # Phrase Stats Panel
    try:
        stats = ranker.get_ranking_stats()
        top_phrases = ranker.get_top_phrases(3)

        stats_text = f"""
Total Phrases: {stats['total_phrases']}
Max Score: {stats['max_score']}
Avg Score: {stats['avg_score']}

Top Phrases:
"""
        for i, phrase in enumerate(top_phrases, 1):
            stats_text += f"{i}. {phrase.phrase} ({phrase.score})\n"

    except Exception as e:
        stats_text = f"Stats unavailable: {e}"

    # Session Stats Panel
    if session_stats:
        runtime = datetime.now() - session_stats.session_start
        session_text = f"""
Runtime: {runtime}
Generated: {session_stats.phrases_generated}
Valid: {session_stats.valid_phrases}
Success Rate: {session_stats.valid_phrases / max(1, session_stats.phrases_generated) * 100:.1f}%
"""
    else:
        session_text = "Session not started"

    # Tiles Panel
    tile_text = f"Total: {sum(tiles.tiles.values())} tiles\n"
    sorted_tiles = sorted(tiles.tiles.items())
    for i in range(0, len(sorted_tiles), 6):  # 6 tiles per line
        line_tiles = sorted_tiles[i:i+6]
        tile_text += " ".join([f"{letter}×{count}" if count > 1 else letter
                              for letter, count in line_tiles]) + "\n"

    layout.split_column(
        Panel(stats_text, title="Phrase Statistics"),
        Panel(session_text, title="Current Session"),
        Panel(tile_text, title="Available Tiles")
    )

    return layout


async def generation_cycle(tiles: TileInventory, llm_client: OllamaClient,
                          ranker: PhraseRanker, config: OptimizationConfig,
                          console: Console) -> int:
    """
    Run a single generation cycle.

    Returns:
        Number of valid phrases generated
    """
    try:
        # Get context phrases from previous successes
        context_phrases = ranker.get_context_phrases()

        # Generate new phrases
        logger.debug(f"Generating {config.generation_batch_size} phrases...")
        phrase_candidates = llm_client.generate_phrases(
            tiles=tiles,
            context_phrases=context_phrases,
            batch_size=config.generation_batch_size
        )

        if not phrase_candidates:
            logger.warning("No phrases generated from LLM")
            return 0

        # Add to ranker (validates, scores, and stores)
        added_phrases = ranker.add_phrase_candidates(
            phrase_candidates,
            tiles,
            llm_client.model_name,
            f"Context: {len(context_phrases)} phrases"
        )

        logger.info(f"Generated {len(phrase_candidates)} candidates, "
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
                             console: Console, db_path: str = "data/phrases.db"):
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
        llm_client = OllamaClient()
        ranker = PhraseRanker(db_path, config)
        db = PhraseDatabase(db_path)

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

                # Run generation cycle
                valid_phrases = await generation_cycle(tiles, llm_client, ranker, config, console)

                # Update session statistics
                generation_session.phrases_generated += config.generation_batch_size
                generation_session.valid_phrases += valid_phrases

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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")
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
        result = asyncio.run(main_generation_loop(tiles, config, console, db_path))
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