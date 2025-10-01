"""
LLM Client Module

Flexible interface for integrating with different LLM providers (Mistral, OpenAI, Anthropic).
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
import httpx
import json

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    MISTRAL = "mistral"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMResponse:
    """Standardized LLM response."""
    def __init__(self, content: str, usage: Optional[Dict[str, Any]] = None, model: Optional[str] = None):
        self.content = content
        self.usage = usage or {}
        self.model = model


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
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
        self.client.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
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
                "random_seed": kwargs.get("random_seed", None)
            }
            
            # Remove None values to avoid API errors
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                model=data.get("model", self.model)
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
        self.client.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                model=data.get("model", self.model)
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
        self.client.headers.update({
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        })
    
    async def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Convert messages to Anthropic format
            system_message = None
            conversation_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            payload = {
                "model": self.model,
                "messages": conversation_messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
            
            if system_message:
                payload["system"] = system_message
            
            response = await self.client.post(
                f"{self.base_url}/messages",
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data["content"][0]["text"],
                usage=data.get("usage", {}),
                model=data.get("model", self.model)
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
        context_documents: List[Dict[str, Any]], 
        max_context_length: int = 4000,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> tuple[str, float]:
        """Generate a RAG answer using the LLM."""
        try:
            # Prepare context
            context_texts = []
            for doc in context_documents[:5]:  # Limit to top 5 documents
                text = doc.get("text", "")
                filename = doc.get("filename", "Unknown")
                if len(text) > 2000:  # Increased from 500 to 2000 characters
                    text = text[:2000] + "..."
                context_texts.append(f"Document: {filename}\n{text}")
            
            context = "\n\n".join(context_texts)
            
            # Truncate context if too long
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
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
                {"role": "user", "content": user_prompt}
            ]
            
            # Generate response
            response = await self.client.generate_response(
                messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Calculate confidence based on response length and context usage
            confidence = min(0.9, len(response.content) / 200.0)
            
            return response.content, confidence
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I couldn't generate a proper answer at this time.", 0.1
    
    async def close(self):
        """Close the LLM client."""
        await self.client.close()
