from typing import Literal, Union

from src.core.config import settings
from src.llm.base import BaseLLM
from src.llm.providers.gemini_client import GeminiClient
from src.llm.providers.ollama_client import OllamaClient
from src.utils.logger import logger


class LLLMFactory:
    """Factory for creating pre-configured LLM clients by purpose.

    Selects the provider (Gemini or Ollama) from ``settings.llm.provider``
    and picks the appropriate model name + temperature for each purpose.

    Supported purposes
    ------------------
    - ``"summary"``    — creative generation, uses ``summary_model`` + configured temperature
    - ``"rag"``        — deterministic retrieval tasks, forces ``temperature=0.0``
    - ``"classifier"`` — deterministic classification, forces ``temperature=0.0``
    """

    @staticmethod
    def create_client(
        purpose: Literal["summary", "rag", "classifier"],
        provider: str = None,
    ) -> Union[GeminiClient, OllamaClient]:
        """Return an LLM client configured for the given purpose.

        Args:
            purpose: Intended use case — ``"summary"``, ``"rag"``, or ``"classifier"``.
            provider: Optional provider override (``"gemini"`` or ``"ollama"``).
                     If None, defaults to ``settings.llm.provider``.

        Returns:
            A :class:`GeminiClient` or :class:`OllamaClient` with the
            model and temperature appropriate for *purpose*.

        Raises:
            ValueError: If an unsupported purpose string is provided.
            ValueError: If the provider is not ``"gemini"`` or ``"ollama"``.
        """
        if provider is None:
            provider = settings.llm.provider.lower()
        else:
            provider = provider.lower()

        # Handle provider aliases
        if provider == "google":
            provider = "gemini"

        if purpose == "summary":
            temperature = settings.llm.temperature
            if provider == "gemini":
                model = settings.llm.summary_model
                logger.debug(
                    f"Creating Gemini LLM for SUMMARY (model: {model}, temp: {temperature})"
                )
                return GeminiClient(model_name=model, temperature=temperature)
            elif provider == "ollama":
                model = settings.ollama.summary_model
                logger.debug(
                    f"Creating Ollama LLM for SUMMARY (model: {model}, temp: {temperature})"
                )
                return OllamaClient(model_name=model, temperature=temperature)
            else:
                raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'gemini' or 'ollama'.")

        elif purpose in ("rag", "classifier"):
            # Both tasks require deterministic output — force temperature to 0
            temperature = 0.0
            if provider == "gemini":
                model = settings.llm.rag_model
                logger.debug(
                    f"Creating Gemini LLM for {purpose.upper()} (model: {model}, temp: {temperature})"
                )
                return GeminiClient(model_name=model, temperature=temperature)
            elif provider == "ollama":
                model = settings.ollama.rag_model
                logger.debug(
                    f"Creating Ollama LLM for {purpose.upper()} (model: {model}, temp: {temperature})"
                )
                return OllamaClient(model_name=model, temperature=temperature)
            else:
                raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'gemini' or 'ollama'.")

        else:
            logger.error(f"Unsupported LLM purpose: '{purpose}'")
            raise ValueError(f"LLM purpose not supported: {purpose}")


