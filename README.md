# Scrabble Phrase Generator

A local LLM-powered system that generates wintery phrases from Scrabble tiles, optimizing for the highest scores. Designed for hands-off operation on MacBook with thermal and performance monitoring.

## Features

- **Local LLM Integration**: Uses Ollama for completely local phrase generation (no external APIs)
- **Complex Tile Parsing**: Handles complex tile formats like `9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh`
- **Intelligent Scoring**: Proper Scrabble scoring with blank tile optimization
- **Continuous Operation**: Runs continuously, building a database of top-scoring phrases
- **MacBook Optimized**: Thermal monitoring and CPU/memory management for safe long-running operation
- **Rich CLI Interface**: Beautiful terminal interface with real-time status updates

## Installation

1. **Prerequisites**:
   - Python 3.11+
   - UV package manager
   - Ollama installed and running

2. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama2:7b  # Or another preferred model
   ```

3. **Clone and setup the project**:
   ```bash
   cd try-make-phrases
   uv sync  # Install dependencies and create virtual environment
   ```

## Usage

### Basic Generation
```bash
# Generate phrases from tiles
uv run python -m src.phrase_generator.main generate "9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh"
```

### Tile Format
The tile format supports:
- Numbers before letters indicate quantity: `9i` = 9 'i' tiles
- Letters without numbers default to 1: `k` = 1 'k' tile
- Underscore represents blank tiles: `2_` = 2 blank tiles
- Example: `9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh`

### Advanced Options
```bash
# Custom configuration
uv run python -m src.phrase_generator.main generate \
  "your_tiles_here" \
  --batch 20 \              # Generate 20 phrases per cycle
  --delay 60 \              # Wait 60 seconds between cycles
  --min-score 15 \          # Only keep phrases scoring 15+
  --max-phrases 500 \       # Keep top 500 phrases
  --verbose                 # Enable debug logging

# Check current status and top phrases
uv run python -m src.phrase_generator.main status --top 20

# Use custom database location
uv run python -m src.phrase_generator.main generate "tiles" --db "custom_path.db"
```

### Example Session
```bash
$ uv run python -m src.phrase_generator.main generate "2winter5snow3cold"

Starting Scrabble Phrase Generator
Configuration:
├ Tiles: 2winter5snow3cold
├ Batch size: 15
├ Delay: 30s
├ Min score: 10
└ Database: data/phrases.db

[Live display showing system health, top phrases, and generation progress]
```

## Architecture

### Core Components

- **TileParser**: Parses complex tile format into usable inventory
- **OllamaClient**: Interfaces with local LLM for phrase generation
- **PhraseValidator**: Validates phrases can be constructed from available tiles
- **ScrabbleScorer**: Calculates accurate Scrabble scores
- **PhraseRanker**: Manages SQLite database of top-scoring phrases
- **Main Loop**: Orchestrates continuous generation with MacBook optimizations

### MacBook Optimizations

- **Thermal Monitoring**: Automatically throttles when CPU temperature rises
- **CPU Usage Control**: Reduces activity when CPU usage exceeds threshold
- **Memory Management**: Monitors memory usage and triggers cleanup
- **Efficient Intervals**: 30-second delays between generations to prevent overheating
- **Graceful Shutdown**: Handles Ctrl+C to save state before stopping

### Data Storage

- **SQLite Database**: Efficient storage with indexes for fast ranking queries
- **Automatic Cleanup**: Removes low-scoring phrases when database grows large
- **Session Tracking**: Records generation statistics for each run
- **WAL Mode**: Better concurrent access for improved performance

## Generated Phrase Examples

With tiles `9i13e2mk10a3r5dj2t4s6o2bx5n5pc2_4glzvwyh`:

```
Top Scoring Phrases:
1. FREEZING RAIN (21 points)
2. WINTER WONDERLAND (20 points)
3. HOLIDAY CHEER (20 points)
4. WINTER MORNING (19 points)
5. WINTER STORM (16 points)
```

## Monitoring

The system provides real-time monitoring of:
- System health (CPU, memory, temperature)
- Generation statistics (phrases generated, success rate)
- Top scoring phrases
- Database status

## Configuration

Default settings optimized for MacBook:
- Generate 15 phrases per cycle
- 30-second intervals between generations
- Keep top 1000 phrases in database
- Minimum score threshold of 10 points
- CPU usage limit of 80%
- Memory limit of 2GB

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Pull a model if needed
ollama pull llama2:7b
```

### High CPU Usage
The system automatically throttles when CPU usage is high, but you can:
- Increase `--delay` for longer intervals
- Use a smaller model like `phi:2.7b`
- Reduce `--batch` size

### Database Issues
```bash
# Check database status
uv run python -m src.phrase_generator.main status

# Force cleanup if database is too large
# (This happens automatically but can be triggered manually)
```

## Development

### Setup Development Environment
```bash
# Install development dependencies (includes ruff)
uv sync
```

### Code Formatting and Linting
This project uses Ruff for formatting and linting:

```bash
# Format code
uv run ruff format .

# Check/lint code
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .
```

### Project Structure
```
try-make-phrases/
├── src/phrase_generator/     # Core generation logic
├── storage/                  # Database and data models
├── config/                   # Configuration and settings
├── data/                     # SQLite database
├── logs/                     # Application logs
└── tests/                    # Test suite (when added)
```

### Testing Individual Components
```bash
# Test tile parser
uv run python -m src.phrase_generator.tile_parser

# Test phrase validator
uv run python -m src.phrase_generator.phrase_validator

# Test scoring system
uv run python -m src.phrase_generator.scorer

# Test ranking system
uv run python -m src.phrase_generator.phrase_ranker
```

## License

This project is for educational and personal use.