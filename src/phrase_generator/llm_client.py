"""
Ollama LLM client for generating wintery phrases.
"""

import ollama
from typing import List, Optional, Dict
import re
import time
from loguru import logger
from storage.models import TileInventory, GeneratedPhrase
from src.phrase_generator.word_dictionary import get_word_dictionary


class LLMError(Exception):
    """Exception raised for LLM operations."""

    pass


class OllamaClient:
    """Client for interacting with Ollama to generate wintery phrases."""

    def __init__(self, model_name: str = "llama2:7b", max_retries: int = 3):
        """
        Initialize Ollama client.

        Args:
            model_name: Name of the Ollama model to use
            max_retries: Maximum retry attempts for failed requests
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.logger = logger

        # Test connection and model availability
        self._verify_model()

    def _verify_model(self):
        """Verify that Ollama is running and the model is available."""
        try:
            # Try to list models to check Ollama connection
            models_response = ollama.list()

            # Handle the new typed response from ollama
            model_names = []
            if hasattr(models_response, "models"):
                # New ollama client returns ListResponse object
                for model in models_response.models:
                    if hasattr(model, "model"):
                        model_names.append(model.model)
                    elif hasattr(model, "name"):
                        model_names.append(model.name)
            elif isinstance(models_response, dict) and "models" in models_response:
                # Fallback for older API format
                for model in models_response["models"]:
                    if isinstance(model, dict):
                        name = model.get("name") or model.get("model")
                        if name:
                            model_names.append(name)

            self.logger.info(f"Available Ollama models: {model_names}")

            if self.model_name not in model_names:
                self.logger.warning(
                    f"Model {self.model_name} not found in available models: {model_names}"
                )

                # Try to pull the model if it's a standard one
                if self.model_name in [
                    "llama2:7b",
                    "mistral:7b",
                    "phi:2.7b",
                    "llama2",
                    "mistral",
                    "phi",
                ]:
                    self.logger.info(f"Attempting to pull model: {self.model_name}")
                    try:
                        ollama.pull(self.model_name)
                        self.logger.info(
                            f"Successfully pulled model: {self.model_name}"
                        )
                    except Exception as pull_error:
                        self.logger.error(f"Failed to pull model: {pull_error}")
                        if model_names:
                            self.model_name = model_names[0]
                            self.logger.warning(
                                f"Using first available model as fallback: {self.model_name}"
                            )
                        else:
                            raise LLMError(
                                f"No models available and failed to pull {self.model_name}"
                            )
                else:
                    # Use first available model as fallback
                    if model_names:
                        self.model_name = model_names[0]
                        self.logger.warning(f"Using fallback model: {self.model_name}")
                    else:
                        raise LLMError("No models available in Ollama")
            else:
                self.logger.info(f"Using model: {self.model_name}")

        except LLMError:
            # Re-raise LLMError as-is
            raise
        except Exception as e:
            # Try a simple ping to check if Ollama is running
            try:
                # Try a minimal generate call to test connection
                test_response = ollama.generate(
                    model=self.model_name, prompt="test", options={"num_predict": 1}
                )
                self.logger.info(
                    f"Ollama connection verified with model: {self.model_name}"
                )
            except Exception as test_error:
                raise LLMError(
                    f"Failed to connect to Ollama. Make sure Ollama is running and the model is available. Error: {e}"
                )

    def create_wintery_prompt(
        self,
        tiles: TileInventory,
        context_phrases: Optional[List[str]] = None,
        batch_size: int = 10,
    ) -> str:
        """
        Create a prompt for generating wintery phrases from available tiles.

        Args:
            tiles: Available scrabble tiles
            context_phrases: Previous successful phrases for inspiration
            batch_size: Number of phrases to request

        Returns:
            Formatted prompt string
        """
        # Convert tiles to a readable format
        available_letters = []
        for letter, count in sorted(tiles.tiles.items()):
            if count == 1:
                available_letters.append(letter)
            else:
                available_letters.append(f"{letter}×{count}")

        tiles_display = " ".join(available_letters)

        # Get high-value inspiration words from dictionary
        word_dict = get_word_dictionary()
        inspiration_words = word_dict.get_inspiration_words(
            tiles, count=15, min_score=6
        )

        inspiration_section = ""
        if inspiration_words:
            inspiration_section = f"""
FEEL FREE TO USE THESE OPTIONAL HIGH VALUE WORDS IN YOUR OUTPUT:
{" - ".join(inspiration_words)}
"""

        prompt = f"""Create {batch_size} winter-themed phrases.

RULES:
- Must not be full-blown sentences. Phrases only. Try for 4-10 words.
- Spaces and punctuation are okay
- Focus on winter, snow, holiday themes

{inspiration_section}

IMPORTANT: Output one phrase per line in plaintext format. ONLY THE PHRASE ITSELF.

EXAMPLE FORMAT:
HAVING A HOLLY JOLLY CHRISTMAS
WINTER MORNING SLEDDING ADVENTURE WITH FRIENDS
COZY FIREPLACE ON SNOWY EVENING
HOLIDAY CELEBRATION WITH FAMILY
ICE SKATING ON FROZEN POND IN NEBRASKA
MERRY XMAS YA FILTHY ANIMAL
BRONZE MEDALIST AT THE WINTER OLYMPICS
KEANU REEVES AND UMA THURMAN CAROLING TOGETHER

{batch_size} new unique phrases:
"""
        return prompt

    def generate_phrases(
        self,
        tiles: TileInventory,
        context_phrases: Optional[List[str]] = None,
        batch_size: int = 10,
    ) -> List[str]:
        """
        Generate wintery phrases using the LLM.

        Args:
            tiles: Available scrabble tiles
            context_phrases: Previous successful phrases for context
            batch_size: Number of phrases to generate

        Returns:
            List of generated phrase strings

        Raises:
            LLMError: If generation fails after retries
        """
        prompt = self.create_wintery_prompt(tiles, context_phrases, batch_size)

        # Log the complete prompt for debugging
        self.logger.debug(f"Fresh generation prompt:\n{prompt}")

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Generating phrases (attempt {attempt + 1})")

                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": 0.8,  # Some creativity but not too random
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_predict": 200,  # Limit response length
                    },
                )

                raw_response = response["response"]

                # Log raw LLM response for debugging
                self.logger.debug(
                    f"Raw LLM response for fresh generation:\n{raw_response}"
                )

                phrases = self._parse_phrase_response(raw_response)

                if phrases:
                    self.logger.debug(f"Generated {len(phrases)} phrases successfully")
                    return phrases
                else:
                    self.logger.warning(
                        f"No valid phrases in response: {raw_response[:100]}..."
                    )

            except Exception as e:
                self.logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise LLMError(
                        f"Failed to generate phrases after {self.max_retries} attempts: {e}"
                    )

                # Wait before retrying
                time.sleep(2**attempt)

        return []

    def _parse_phrase_response(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract phrase candidates.

        Args:
            response: Raw LLM response

        Returns:
            List of cleaned phrase strings
        """
        phrases = []

        # Split by lines and clean up
        lines = response.strip().split("\n")

        for line in lines:
            original_line = line  # Keep for debugging

            # Clean the line
            line = line.strip()

            # Skip empty lines, numbers, or lines that look like instructions
            if not line or line.isdigit() or len(line) < 3:
                continue

            # Remove common prefixes/bullets and numbering
            line_before_regex = line
            line = re.sub(r"^[\d\.\-\*\+\s]*", "", line)
            line = line.strip()

            # Log the parsing steps for debugging
            if line_before_regex != line:
                self.logger.debug(
                    f"Parsing: '{original_line}' → '{line_before_regex}' → '{line}'"
                )

            # Remove explanatory text after dashes or descriptions
            # Handle formats like: "PHRASE NAME" - This phrase adds...
            if " - " in line:
                line = line.split(" - ")[0].strip()

            # Remove word count annotations like "- 7 WORDS" or "(5 words)"
            line = re.sub(r"\s*-\s*\d+\s*WORDS?\s*$", "", line, flags=re.IGNORECASE)
            line = re.sub(r"\s*\(\d+\s*words?\)\s*$", "", line, flags=re.IGNORECASE)
            line = line.strip()

            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            if line.startswith("'") and line.endswith("'"):
                line = line[1:-1]

            line = line.strip()

            # Skip if it looks like a model response artifact
            if any(
                skip_phrase in line.lower()
                for skip_phrase in [
                    "your",
                    "phrases:",
                    "generate",
                    "available",
                    "tiles",
                    "format",
                    "response",
                    "here are",
                    "output",
                    "improved",
                ]
            ):
                continue

            # Basic validation - should look like a phrase
            if self._is_valid_phrase_format(line):
                phrases.append(line.upper().strip())
            else:
                self.logger.debug(f"Rejected phrase format: '{line}'")

        # Log final extracted phrases
        if phrases:
            self.logger.debug(
                f"Successfully extracted {len(phrases)} phrases: {phrases}"
            )
        else:
            self.logger.debug("No phrases extracted from response")

        return phrases[:15]  # Limit to reasonable number

    def improve_single_phrase(
        self, base_phrase: str, tiles: TileInventory, num_attempts: int = 5
    ) -> List[str]:
        """
        Improve a single phrase by making multiple improvement attempts.
        Uses leftover tiles to find high-value inspiration words.

        Args:
            base_phrase: Single phrase to improve
            tiles: Available tiles for validation
            num_attempts: Number of improvement attempts to generate

        Returns:
            List of improved phrase variants
        """
        if not base_phrase:
            return []

        # Convert tiles to a readable format
        available_letters = []
        for letter, count in sorted(tiles.tiles.items()):
            if count == 1:
                available_letters.append(letter)
            else:
                available_letters.append(f"{letter}×{count}")

        tiles_display = " ".join(available_letters)

        # Get leftover tile inspiration words
        word_dict = get_word_dictionary()
        leftover_inspiration = word_dict.get_leftover_inspiration_words(
            base_phrase, tiles, count=12, min_score=4
        )

        leftover_section = ""
        if leftover_inspiration:
            leftover_section = f"""
FEEL FREE TO USE THESE OPTIONAL HIGH VALUE WORDS FOR YOUR IMPROVEMENTS:
{" - ".join(leftover_inspiration)}
"""

        prompt = f"""Improve this winter phrase by making {num_attempts} different variations using tiles: {tiles_display}

{leftover_section}

IMPROVEMENT METHODS:
- Add adjectives: COLD BREEZE → COLD WINTER BREEZE
- Add adverbs: COLD BREEZE → REMARKABLY COLD BREEZE
- Add locations: SNOWY PARK → SNOWY PARK WITH TREES
- Add activities: WINTER MORNING → WINTER MORNING SLEDDING
- Swap words for higher-scoring ones: WINTER MORNING → WINTER EVENING
- Add details: HOLIDAY CHEER → HOLIDAY CHEER AND JOY

Generate {num_attempts} different improved versions. Improve the phrase for maximum Scrabble points. Try some different strategies, including just small improvements.

IMPORTANT: Output one phrase per line in plaintext format. ONLY THE PHRASE ITSELF.

ORIGINAL PHRASE TO IMPROVE:
{base_phrase}

{num_attempts} improved versions:
"""

        # Log the improvement prompt for debugging
        self.logger.debug(f"Single phrase improvement prompt:\n{prompt}")

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.9,  # More creative for improvements
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_predict": 300,  # Allow longer responses for multiple attempts
                },
            )

            raw_response = response["response"]

            # Log raw LLM response for debugging
            self.logger.debug(
                f"Raw LLM response for single phrase improvement:\n{raw_response}"
            )

            improved_phrases = self._parse_phrase_response(raw_response)

            if improved_phrases:
                self.logger.debug(
                    f"Generated {len(improved_phrases)} variations of '{base_phrase}'"
                )
                return improved_phrases
            else:
                self.logger.warning(
                    f"No valid improved phrases in response: {raw_response[:100]}..."
                )
                return []

        except Exception as e:
            self.logger.error(f"Single phrase improvement failed: {e}")
            return []

    def _is_valid_phrase_format(self, phrase: str) -> bool:
        """Check if a string looks like a valid phrase."""
        # Must contain letters
        if not re.search(r"[a-zA-Z]", phrase):
            self.logger.debug(f"Rejected '{phrase}': No letters found")
            return False

        # Reasonable length
        if len(phrase) < 3 or len(phrase) > 100:
            self.logger.debug(
                f"Rejected '{phrase}': Length {len(phrase)} not between 3-100"
            )
            return False

        # Should have at least one space (multi-word phrase) or be a compound word
        if " " not in phrase and len(phrase) < 8:
            self.logger.debug(f"Rejected '{phrase}': No spaces and length < 8")
            return False

        # Shouldn't have too many special characters
        special_chars = re.findall(r"[^a-zA-Z\s\'\-\.]", phrase)
        special_count = len(special_chars)
        if special_count > 2:
            self.logger.debug(
                f"Rejected '{phrase}': Too many special chars ({special_count}): {special_chars}"
            )
            return False

        self.logger.debug(f"Accepted '{phrase}': Passed all validation checks")
        return True

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        try:
            models_response = ollama.list()

            if hasattr(models_response, "models"):
                # New ollama client returns ListResponse object
                for model in models_response.models:
                    if hasattr(model, "model") and model.model == self.model_name:
                        return {
                            "name": model.model,
                            "size": getattr(model, "size", "unknown"),
                            "modified": str(getattr(model, "modified_at", "unknown")),
                            "family": getattr(
                                getattr(model, "details", None), "family", "unknown"
                            )
                            if hasattr(model, "details")
                            else "unknown",
                        }
            elif isinstance(models_response, dict) and "models" in models_response:
                # Fallback for older API format
                for model in models_response["models"]:
                    if isinstance(model, dict):
                        model_name = model.get("name") or model.get("model")
                        if model_name == self.model_name:
                            return {
                                "name": model_name,
                                "size": model.get("size", "unknown"),
                                "modified": model.get("modified_at", "unknown"),
                                "family": model.get("details", {}).get(
                                    "family", "unknown"
                                )
                                if "details" in model
                                else "unknown",
                            }

            return {"name": self.model_name, "status": "not_found"}
        except Exception as e:
            return {"name": self.model_name, "error": str(e)}


# Testing and examples
if __name__ == "__main__":
    # Test the LLM client

    try:
        client = OllamaClient()
        print(f"Using model: {client.model_name}")
        print(f"Model info: {client.get_model_info()}")

        # Test with sample tiles
        from src.phrase_generator.tile_parser import parse_tile_string

        test_tiles = parse_tile_string("2winter5snow3cold")
        print(f"Test tiles: {test_tiles.tiles}")

        phrases = client.generate_phrases(test_tiles, batch_size=5)
        print(f"Generated phrases: {phrases}")

    except LLMError as e:
        print(f"LLM Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
