"""
LLM Client Module

Flexible interface for integrating with different LLM providers (Mistral, OpenAI, Anthropic).
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    MISTRAL = "mistral"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMResponse:
    """Standardized LLM response."""

    def __init__(
        self,
        content: str,
        usage: dict[str, Any | None] = None,
        model: str | None = None,
    ):
        self.content = content
        self.usage = usage or {}
        self.model = model


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)

    @abstractmethod
    async def generate_response(
        self, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        pass

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class MistralClient(LLMClient):
    """Mistral API client."""

    def __init__(self, api_key: str, model: str = "mistral-small"):
        super().__init__(api_key, model, "https://api.mistral.ai/v1")
        self.client.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    async def generate_response(
        self, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate response using Mistral API."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "stream": False,
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "random_seed": kwargs.get("random_seed", None),
            }

            # Remove None values to avoid API errors
            payload = {k: v for k, v in payload.items() if v is not None}

            response = await self.client.post(
                f"{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()

            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                model=data.get("model", self.model),
            )

        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "mistral"


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model, "https://api.openai.com/v1")
        self.client.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    async def generate_response(
        self, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }

            response = await self.client.post(
                f"{self.base_url}/chat/completions", json=payload
            )
            response.raise_for_status()

            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                model=data.get("model", self.model),
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "openai"


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model, "https://api.anthropic.com/v1")
        self.client.headers.update(
            {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
        )

    async def generate_response(
        self, messages: list[dict[str, str]], **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append(
                        {"role": msg["role"], "content": msg["content"]}
                    )

            payload = {
                "model": self.model,
                "messages": conversation_messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }

            if system_message:
                payload["system"] = system_message

            response = await self.client.post(f"{self.base_url}/messages", json=payload)
            response.raise_for_status()

            data = response.json()

            return LLMResponse(
                content=data["content"][0]["text"],
                usage=data.get("usage", {}),
                model=data.get("model", self.model),
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def get_provider_name(self) -> str:
        return "anthropic"


class LLMClientFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_client(provider: str, api_key: str, model: str) -> LLMClient:
        """Create an LLM client based on the provider."""
        provider = provider.lower()

        if provider == LLMProvider.MISTRAL:
            return MistralClient(api_key, model)
        elif provider == LLMProvider.OPENAI:
            return OpenAIClient(api_key, model)
        elif provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(api_key, model)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class LLMManager:
    """Manager for LLM operations."""

    def __init__(self, provider: str, api_key: str, model: str):
        self.provider = provider
        self.client = LLMClientFactory.create_client(provider, api_key, model)

    async def generate_rag_answer(
        self,
        question: str,
        context_documents: list[dict[str, Any]],
        max_context_length: int = 8000,  # Increased initial limit
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> tuple[str, float, bool]:  # Added truncation flag
        """Generate a RAG answer using the LLM with hybrid context management."""
        try:
            from .smart_context import smart_truncate_text

            # Prepare context with chunked documents only
            context_texts = []
            for doc in context_documents[:5]:  # Limit to top 5 documents
                text = doc.get("text", "")
                filename = doc.get("filename", "Unknown")
                is_chunked = doc.get("is_chunked", False)
                chunk_index = doc.get("chunk_index")
                total_chunks = doc.get("total_chunks", 1)

                # All documents should be chunked
                if is_chunked and chunk_index is not None:
                    # Add chunk context information
                    chunk_info = f" (Chunk {chunk_index + 1} of {total_chunks})"
                    context_texts.append(f"Document: {filename}{chunk_info}\n{text}")
                else:
                    # Skip non-chunked documents
                    logger.warning(f"Skipping non-chunked document: {filename}")
                    continue

            context = "\n\n".join(context_texts)

            # Try with full context first
            try:
                answer, confidence, _ = await self._generate_with_context(
                    question, context, temperature, max_tokens
                )
                return answer, confidence, False  # No truncation

            except Exception as e:
                # Check if it's a context length error
                error_msg = str(e).lower()
                if any(
                    keyword in error_msg
                    for keyword in ["context", "length", "token", "limit", "too long"]
                ):
                    logger.warning(f"Context too long, retrying with truncation: {e}")

                    # Apply smart truncation
                    if len(context) > max_context_length:
                        context = smart_truncate_text(
                            context, max_context_length, question
                        )

                    # Retry with truncated context
                    try:
                        answer, confidence, _ = await self._generate_with_context(
                            question, context, temperature, max_tokens
                        )
                        return answer, confidence, True  # Truncation occurred
                    except Exception as retry_e:
                        logger.error(f"Failed even with truncated context: {retry_e}")
                        return (
                            "I apologize, but I couldn't generate a proper answer due to context limitations.",
                            0.1,
                            True,
                        )
                else:
                    # Re-raise if it's not a context length error
                    raise

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return (
                "I apologize, but I couldn't generate a proper answer at this time.",
                0.1,
                False,
            )

    async def _generate_with_context(
        self, question: str, context: str, temperature: float, max_tokens: int
    ) -> tuple[str, float, bool]:
        """Generate response with given context."""
        # Create messages
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided documents.
        Use only the information from the documents to answer the question. If the documents don't contain
        enough information to answer the question, say so clearly. Be concise and accurate."""

        user_prompt = f"""Based on the following documents, please answer this question: {question}

Documents:
{context}

Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate response
        response = await self.client.generate_response(
            messages, temperature=temperature, max_tokens=max_tokens
        )

        # Calculate confidence based on response length and context usage
        confidence = min(0.9, len(response.content) / 200.0)

        return response.content, confidence, False  # No truncation occurred

    async def close(self):
        """Close the LLM client."""
        await self.client.close()
