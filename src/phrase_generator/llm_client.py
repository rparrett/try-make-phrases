"""
Ollama LLM client for generating wintery phrases.
"""

import ollama
from typing import List, Optional, Dict
import re
import time
import logging
from storage.models import TileInventory, GeneratedPhrase


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
        self.logger = logging.getLogger(__name__)

        # Test connection and model availability
        self._verify_model()

    def _verify_model(self):
        """Verify that Ollama is running and the model is available."""
        try:
            # Try to list models to check Ollama connection
            models_response = ollama.list()

            # Handle the new typed response from ollama
            model_names = []
            if hasattr(models_response, 'models'):
                # New ollama client returns ListResponse object
                for model in models_response.models:
                    if hasattr(model, 'model'):
                        model_names.append(model.model)
                    elif hasattr(model, 'name'):
                        model_names.append(model.name)
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Fallback for older API format
                for model in models_response['models']:
                    if isinstance(model, dict):
                        name = model.get('name') or model.get('model')
                        if name:
                            model_names.append(name)

            self.logger.info(f"Available Ollama models: {model_names}")

            if self.model_name not in model_names:
                self.logger.warning(f"Model {self.model_name} not found in available models: {model_names}")

                # Try to pull the model if it's a standard one
                if self.model_name in ['llama2:7b', 'mistral:7b', 'phi:2.7b', 'llama2', 'mistral', 'phi']:
                    self.logger.info(f"Attempting to pull model: {self.model_name}")
                    try:
                        ollama.pull(self.model_name)
                        self.logger.info(f"Successfully pulled model: {self.model_name}")
                    except Exception as pull_error:
                        self.logger.error(f"Failed to pull model: {pull_error}")
                        if model_names:
                            self.model_name = model_names[0]
                            self.logger.warning(f"Using first available model as fallback: {self.model_name}")
                        else:
                            raise LLMError(f"No models available and failed to pull {self.model_name}")
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
                test_response = ollama.generate(model=self.model_name, prompt="test", options={"num_predict": 1})
                self.logger.info(f"Ollama connection verified with model: {self.model_name}")
            except Exception as test_error:
                raise LLMError(f"Failed to connect to Ollama. Make sure Ollama is running and the model is available. Error: {e}")

    def create_wintery_prompt(self, tiles: TileInventory, context_phrases: Optional[List[str]] = None,
                            batch_size: int = 10) -> str:
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

        prompt = f"""You are creating winter and holiday-themed phrases using ONLY the following Scrabble tiles:

🎯 AVAILABLE TILES: {tiles_display}
(Note: _ represents blank tiles that can be any letter)

⚠️ CRITICAL RULES:
1. Use ONLY the letters shown above (each letter can be used up to the number of times shown)
2. BEFORE writing each phrase, check that ALL letters are available in the tiles
3. Blank tiles (_) can represent any letter but are worth 0 points
4. Spaces, punctuation, and apostrophes are free (don't consume tiles)
5. Create phrases that evoke winter, snow, holidays, or cold weather themes
6. Aim for LONGER phrases (4-8+ words) to maximize Scrabble points
7. Use as many tiles as possible - longer phrases score more points
8. Double-check: Can you spell this phrase with the available letters?

EXAMPLES of HIGH-SCORING winter phrases (using common letters):
- WINTER STORIES AND MEMORIES
- ICE SKATING ON THE POND
- COZY READING BY THE FIRE
- WINTER GAMES AND ACTIVITIES
- SLEDDING DOWN THE SNOWY SLOPE
- HOT CHOCOLATE AND COOKIES
- DECORATING THE TREE FOR CHRISTMAS

"""

        if context_phrases:
            prompt += f"\nINSPIRATION from previous good phrases:\n"
            for phrase in context_phrases[-3:]:  # Use last 3 for inspiration
                prompt += f"- {phrase}\n"

        prompt += f"""
🎯 CHALLENGE: Generate {batch_size} LONG winter-themed phrases using only the available tiles.
📏 TARGET: 4-8+ words per phrase to maximize Scrabble points!
✅ REQUIREMENT: Every letter must be available in the tiles shown above.

Format your response as a simple list, one phrase per line:

WINTER STORIES AND MEMORIES
ICE SKATING ON THE POND
COZY READING BY THE FIRE

Your {batch_size} long, high-scoring phrases:"""

        return prompt

    def generate_phrases(self, tiles: TileInventory, context_phrases: Optional[List[str]] = None,
                        batch_size: int = 10) -> List[str]:
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

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Generating phrases (attempt {attempt + 1})")

                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.8,  # Some creativity but not too random
                        'top_k': 40,
                        'top_p': 0.9,
                        'num_predict': 200,  # Limit response length
                    }
                )

                raw_response = response['response']
                phrases = self._parse_phrase_response(raw_response)

                if phrases:
                    self.logger.debug(f"Generated {len(phrases)} phrases successfully")
                    return phrases
                else:
                    self.logger.warning(f"No valid phrases in response: {raw_response[:100]}...")

            except Exception as e:
                self.logger.error(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise LLMError(f"Failed to generate phrases after {self.max_retries} attempts: {e}")

                # Wait before retrying
                time.sleep(2 ** attempt)

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
        lines = response.strip().split('\n')

        for line in lines:
            # Clean the line
            line = line.strip()

            # Skip empty lines, numbers, or lines that look like instructions
            if not line or line.isdigit() or len(line) < 3:
                continue

            # Remove common prefixes/bullets and numbering
            line = re.sub(r'^[\d\.\-\*\+\s]*', '', line)
            line = line.strip()

            # Remove word count annotations like "- 7 WORDS" or "(5 words)"
            line = re.sub(r'\s*-\s*\d+\s*WORDS?\s*$', '', line, flags=re.IGNORECASE)
            line = re.sub(r'\s*\(\d+\s*words?\)\s*$', '', line, flags=re.IGNORECASE)
            line = line.strip()

            # Remove quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            if line.startswith("'") and line.endswith("'"):
                line = line[1:-1]

            # Skip if it looks like a model response artifact
            if any(skip_phrase in line.lower() for skip_phrase in [
                'your', 'phrases', 'generate', 'winter', 'available', 'tiles',
                'example', 'format', 'response', 'here are', 'great!', 'using only',
                'long winter', 'high-scoring'
            ]):
                continue

            # Basic validation - should look like a phrase
            if self._is_valid_phrase_format(line):
                phrases.append(line.upper().strip())

        return phrases[:15]  # Limit to reasonable number

    def _is_valid_phrase_format(self, phrase: str) -> bool:
        """Check if a string looks like a valid phrase."""
        # Must contain letters
        if not re.search(r'[a-zA-Z]', phrase):
            return False

        # Reasonable length
        if len(phrase) < 3 or len(phrase) > 50:
            return False

        # Should have at least one space (multi-word phrase) or be a compound word
        if ' ' not in phrase and len(phrase) < 8:
            return False

        # Shouldn't have too many special characters
        special_count = len(re.findall(r'[^a-zA-Z\s\'\-\.]', phrase))
        if special_count > 2:
            return False

        return True

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        try:
            models_response = ollama.list()

            if hasattr(models_response, 'models'):
                # New ollama client returns ListResponse object
                for model in models_response.models:
                    if hasattr(model, 'model') and model.model == self.model_name:
                        return {
                            'name': model.model,
                            'size': getattr(model, 'size', 'unknown'),
                            'modified': str(getattr(model, 'modified_at', 'unknown')),
                            'family': getattr(getattr(model, 'details', None), 'family', 'unknown') if hasattr(model, 'details') else 'unknown',
                        }
            elif isinstance(models_response, dict) and 'models' in models_response:
                # Fallback for older API format
                for model in models_response['models']:
                    if isinstance(model, dict):
                        model_name = model.get('name') or model.get('model')
                        if model_name == self.model_name:
                            return {
                                'name': model_name,
                                'size': model.get('size', 'unknown'),
                                'modified': model.get('modified_at', 'unknown'),
                                'family': model.get('details', {}).get('family', 'unknown') if 'details' in model else 'unknown',
                            }

            return {'name': self.model_name, 'status': 'not_found'}
        except Exception as e:
            return {'name': self.model_name, 'error': str(e)}


# Testing and examples
if __name__ == "__main__":
    # Test the LLM client
    logging.basicConfig(level=logging.DEBUG)

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