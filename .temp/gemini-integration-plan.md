# Gemini Integration Plan for ElasticDash Python SDK

**Date**: 2026-02-09
**Status**: Planning
**Target**: Add full Google Gemini API support to ElasticDash SDK

---

## Executive Summary

This plan outlines the implementation of Gemini API support for the ElasticDash Python SDK. The integration will follow the established OpenAI wrapping pattern, providing automatic tracing and observability for Gemini API calls with minimal code changes for users.

**Two Integration Paths**:
1. **Direct Gemini SDK Integration** (Primary) - Function wrapping approach
2. **Langchain-Gemini Integration** (Secondary) - Already partially supported via existing callback handler

---

## 1. Background & Context

### 1.1 Current State
- ElasticDash SDK supports OpenAI via function wrapping (`elasticdash/openai.py`)
- Langchain integration via callback handler supports any LLM (including Gemini)
- Core tracing built on OpenTelemetry with ElasticDashGeneration observations
- Architecture supports multiple LLM providers through consistent patterns

### 1.2 Gemini API Overview

**Python Client**: `google-generativeai` package

**Key Classes**:
```python
import google.generativeai as genai

# Configuration
genai.configure(api_key="...")

# Main model class
model = genai.GenerativeModel('gemini-1.5-pro')

# Generation methods
response = model.generate_content("Hello")           # Sync
response = model.generate_content_async("Hello")    # Async
for chunk in model.generate_content("Hi", stream=True):  # Streaming
    print(chunk.text)
```

**Key Features to Support**:
- Text generation (chat-style with history)
- Function calling (tools)
- Multimodal input (images, audio, video, PDFs)
- Streaming responses
- Token counting
- Safety ratings
- Prompt caching (cost optimization)
- System instructions

---

## 2. Architecture Design

### 2.1 Integration Approach: Function Wrapping

Follow the OpenAI pattern with `wrapt.wrap_function_wrapper`:

```python
# elasticdash/gemini.py

from wrapt import wrap_function_wrapper

def register():
    """Register all Gemini wrappers"""
    for definition in GEMINI_DEFINITIONS:
        wrap_function_wrapper(
            module=definition.module,
            name=f"{definition.object}.{definition.method}",
            wrapper=_create_wrapper(definition)
        )
```

### 2.2 Method Definitions

```python
@dataclass
class GeminiDefinition:
    module: str            # "google.generativeai.generative_models"
    object: str            # "GenerativeModel"
    method: str            # "generate_content"
    type: str              # "chat" or "embedding"
    sync: bool             # True/False
    streaming: bool        # True/False

GEMINI_DEFINITIONS = [
    # Sync generation
    GeminiDefinition(
        module="google.generativeai.generative_models",
        object="GenerativeModel",
        method="generate_content",
        type="chat",
        sync=True,
        streaming=False
    ),
    # Async generation
    GeminiDefinition(
        module="google.generativeai.generative_models",
        object="GenerativeModel",
        method="generate_content_async",
        type="chat",
        sync=False,
        streaming=False
    ),
    # Count tokens (useful for tracking)
    GeminiDefinition(
        module="google.generativeai.generative_models",
        object="GenerativeModel",
        method="count_tokens",
        type="count",
        sync=True,
        streaming=False
    ),
    # Embeddings
    GeminiDefinition(
        module="google.generativeai",
        object="embed_content",
        method=None,  # It's a function, not a method
        type="embedding",
        sync=True,
        streaming=False
    ),
]
```

### 2.3 Data Flow

```
User Code
    ↓
genai.GenerativeModel.generate_content(**kwargs)
    ↓
[ElasticDash Wrapper Intercepts]
    ↓
1. Extract ElasticDash kwargs (elasticdash_trace_id, etc.)
2. Create ElasticDashGeneration observation
3. Extract model name, parameters, prompt
4. Call original Gemini API
5. Extract response, usage, safety ratings
6. Update observation with output & usage
7. End observation (async flush to backend)
    ↓
Return response to user (unchanged)
```

---

## 3. Detailed Implementation

### 3.1 File Structure

```
elasticdash/
├── gemini.py                    # NEW: Main Gemini integration (1,000-1,500 lines)
├── __init__.py                  # MODIFY: Export gemini module
├── _client/
│   └── span.py                  # NO CHANGE: Already supports all needed features
└── langchain/
    └── utils.py                 # MODIFY: Add Gemini model name parsing

tests/
├── test_gemini.py               # NEW: Integration tests (500-800 lines)
└── test_gemini_langchain.py     # NEW: Langchain + Gemini tests (200-300 lines)

.temp/
└── gemini-integration-plan.md   # THIS FILE
```

### 3.2 Core Implementation: elasticdash/gemini.py

#### 3.2.1 Module Structure

```python
"""
ElasticDash integration for Google Gemini API.

This module provides automatic tracing and observability for Gemini API calls
by wrapping the google-generativeai SDK methods.

Usage:
    import elasticdash
    from elasticdash.gemini import register

    elasticdash.init()
    register()  # Register Gemini wrappers

    import google.generativeai as genai
    genai.configure(api_key="...")

    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content("Hello")  # Automatically traced!
"""

from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Union, Generator, AsyncGenerator
from collections.abc import Sequence
import logging
from functools import wraps
import inspect

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse, ContentDict
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from elasticdash import ElasticDash
from elasticdash._client.span import ElasticDashGeneration, ElasticDashEmbedding

logger = logging.getLogger(__name__)

# Global state
_is_registered = False
_elasticdash_client: Optional[ElasticDash] = None
```

#### 3.2.2 Definition Classes

```python
@dataclass
class GeminiDefinition:
    """Definition of a Gemini method to wrap."""
    module: str
    object: str
    method: str
    type: str  # "chat", "embedding", "count"
    sync: bool
    streaming: bool = False

GEMINI_DEFINITIONS = [
    # ... as defined in section 2.2
]
```

#### 3.2.3 Argument Extraction

```python
class GeminiArgsExtractor:
    """Extracts ElasticDash-specific kwargs from Gemini API calls."""

    ELASTICDASH_KWARGS = {
        "elasticdash_trace_id",
        "elasticdash_parent_observation_id",
        "elasticdash_session_id",
        "elasticdash_user_id",
        "elasticdash_tags",
        "elasticdash_metadata",
        "elasticdash_name",
        "elasticdash_public_key",
        "elasticdash_secret_key",
    }

    @classmethod
    def extract_elasticdash_args(cls, kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Split kwargs into Gemini kwargs and ElasticDash kwargs.

        Returns:
            (gemini_kwargs, elasticdash_kwargs)
        """
        elasticdash_kwargs = {}
        gemini_kwargs = {}

        for key, value in kwargs.items():
            if key in cls.ELASTICDASH_KWARGS:
                # Remove prefix and store
                clean_key = key.replace("elasticdash_", "")
                elasticdash_kwargs[clean_key] = value
            else:
                gemini_kwargs[key] = value

        return gemini_kwargs, elasticdash_kwargs
```

#### 3.2.4 Prompt/Response Extraction

```python
class GeminiDataExtractor:
    """Extracts structured data from Gemini requests and responses."""

    @staticmethod
    def extract_prompt(contents: Any, model_instance: Any) -> Dict[str, Any]:
        """
        Extract normalized prompt from Gemini contents.

        Args:
            contents: Can be str, ContentDict, list of ContentDict
            model_instance: GenerativeModel instance (for system instruction)

        Returns:
            Normalized prompt dict with 'messages' and optional 'system_instruction'
        """
        prompt_data = {}

        # Extract system instruction if present
        if hasattr(model_instance, '_system_instruction') and model_instance._system_instruction:
            prompt_data['system_instruction'] = str(model_instance._system_instruction)

        # Normalize contents to list of messages
        if isinstance(contents, str):
            prompt_data['messages'] = [{"role": "user", "parts": [{"text": contents}]}]
        elif isinstance(contents, dict):
            prompt_data['messages'] = [contents]
        elif isinstance(contents, list):
            prompt_data['messages'] = contents
        else:
            # Handle Content objects
            prompt_data['messages'] = [{"role": "user", "parts": [{"text": str(contents)}]}]

        return prompt_data

    @staticmethod
    def extract_response(response: GenerateContentResponse) -> Dict[str, Any]:
        """
        Extract normalized response from Gemini.

        Returns:
            Dict with 'text', 'function_calls', 'safety_ratings', etc.
        """
        response_data = {}

        # Main text content
        if response.text:
            response_data['text'] = response.text

        # Function calls (if any)
        if response.candidates and response.candidates[0].content.parts:
            function_calls = []
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append({
                        'name': part.function_call.name,
                        'args': dict(part.function_call.args)
                    })
            if function_calls:
                response_data['function_calls'] = function_calls

        # Safety ratings
        if response.candidates and response.candidates[0].safety_ratings:
            response_data['safety_ratings'] = [
                {
                    'category': rating.category.name,
                    'probability': rating.probability.name
                }
                for rating in response.candidates[0].safety_ratings
            ]

        # Finish reason
        if response.candidates:
            response_data['finish_reason'] = response.candidates[0].finish_reason.name

        return response_data

    @staticmethod
    def extract_usage(response: GenerateContentResponse) -> Optional[Dict[str, int]]:
        """
        Extract token usage from Gemini response.

        Returns:
            Dict with 'input', 'output', 'total' token counts, or None
        """
        if not response.usage_metadata:
            return None

        usage = response.usage_metadata
        return {
            'input': usage.prompt_token_count,
            'output': usage.candidates_token_count,
            'total': usage.total_token_count,
            # Gemini also provides cached content tokens
            'cached': getattr(usage, 'cached_content_token_count', 0)
        }

    @staticmethod
    def extract_model_parameters(
        generation_config: Optional[Any],
        safety_settings: Optional[Any],
        tools: Optional[Any]
    ) -> Dict[str, Any]:
        """Extract model configuration parameters."""
        params = {}

        if generation_config:
            config = generation_config if isinstance(generation_config, dict) else generation_config.to_dict()
            for key in ['temperature', 'top_p', 'top_k', 'max_output_tokens', 'stop_sequences']:
                if key in config and config[key] is not None:
                    params[key] = config[key]

        if safety_settings:
            params['safety_settings'] = [
                {'category': s.category.name, 'threshold': s.threshold.name}
                for s in safety_settings
            ] if not isinstance(safety_settings, dict) else safety_settings

        if tools:
            params['tools'] = [
                {'name': t.function_declarations[0].name} if hasattr(t, 'function_declarations')
                else str(t)
                for t in tools
            ]

        return params
```

#### 3.2.5 Main Wrapper Function

```python
def _create_wrapper(definition: GeminiDefinition):
    """Create a wrapper function for a specific Gemini method."""

    def _wrapper(wrapped, instance, args, kwargs):
        """The actual wrapper that intercepts Gemini calls."""

        # Skip if ElasticDash not initialized
        if not _elasticdash_client:
            logger.debug("ElasticDash not initialized, passing through")
            return wrapped(*args, **kwargs)

        # Extract ElasticDash-specific kwargs
        gemini_kwargs, elasticdash_kwargs = GeminiArgsExtractor.extract_elasticdash_args(kwargs)

        # Get model name from instance
        model_name = getattr(instance, 'model_name', 'unknown')
        if model_name.startswith('models/'):
            model_name = model_name[7:]  # Remove "models/" prefix

        # Extract generation config, safety settings, tools
        generation_config = gemini_kwargs.get('generation_config')
        safety_settings = gemini_kwargs.get('safety_settings')
        tools = gemini_kwargs.get('tools')

        # Extract model parameters
        model_parameters = GeminiDataExtractor.extract_model_parameters(
            generation_config, safety_settings, tools
        )

        # Extract prompt (contents argument)
        contents = gemini_kwargs.get('contents') or (args[0] if args else None)
        prompt_data = GeminiDataExtractor.extract_prompt(contents, instance)

        # Create observation name
        observation_name = elasticdash_kwargs.pop('name', None) or f"gemini.{definition.method}"

        # Create ElasticDash generation observation
        try:
            generation = _elasticdash_client.start_observation(
                name=observation_name,
                as_type="generation",
                input=prompt_data,
                model=model_name,
                model_parameters=model_parameters,
                **elasticdash_kwargs
            )
        except Exception as e:
            logger.warning(f"Failed to create ElasticDash observation: {e}")
            return wrapped(*args, **gemini_kwargs)

        # Handle streaming vs non-streaming
        try:
            if definition.streaming or gemini_kwargs.get('stream', False):
                return _handle_streaming(
                    wrapped, args, gemini_kwargs, generation, definition.sync
                )
            else:
                return _handle_non_streaming(
                    wrapped, args, gemini_kwargs, generation, definition.sync
                )
        except Exception as e:
            # Log error to observation
            generation.update(
                level="ERROR",
                status_message=f"{type(e).__name__}: {str(e)}"
            )
            generation.end()
            raise

    return _wrapper

def _handle_non_streaming(wrapped, args, gemini_kwargs, generation, is_sync):
    """Handle non-streaming Gemini responses."""

    # Call Gemini API
    response = wrapped(*args, **gemini_kwargs)

    # Extract response data
    response_data = GeminiDataExtractor.extract_response(response)
    usage = GeminiDataExtractor.extract_usage(response)

    # Update observation
    generation.update(
        output=response_data,
        usage_details=usage,
        metadata={
            'finish_reason': response_data.get('finish_reason'),
            'safety_ratings': response_data.get('safety_ratings')
        }
    )
    generation.end()

    return response

def _handle_streaming(wrapped, args, gemini_kwargs, generation, is_sync):
    """Handle streaming Gemini responses."""

    if is_sync:
        return _wrap_sync_generator(wrapped, args, gemini_kwargs, generation)
    else:
        return _wrap_async_generator(wrapped, args, gemini_kwargs, generation)

def _wrap_sync_generator(wrapped, args, gemini_kwargs, generation):
    """Wrap synchronous streaming generator."""

    accumulated_text = []
    accumulated_function_calls = []
    final_usage = None

    try:
        for chunk in wrapped(*args, **gemini_kwargs):
            # Collect chunks
            if chunk.text:
                accumulated_text.append(chunk.text)

            # Check for function calls
            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        accumulated_function_calls.append({
                            'name': part.function_call.name,
                            'args': dict(part.function_call.args)
                        })

            # Track usage
            if chunk.usage_metadata:
                final_usage = GeminiDataExtractor.extract_usage(chunk)

            yield chunk

        # Finalize observation
        output_data = {}
        if accumulated_text:
            output_data['text'] = ''.join(accumulated_text)
        if accumulated_function_calls:
            output_data['function_calls'] = accumulated_function_calls

        generation.update(output=output_data, usage_details=final_usage)
        generation.end()

    except Exception as e:
        generation.update(level="ERROR", status_message=str(e))
        generation.end()
        raise

async def _wrap_async_generator(wrapped, args, gemini_kwargs, generation):
    """Wrap asynchronous streaming generator."""

    accumulated_text = []
    accumulated_function_calls = []
    final_usage = None

    try:
        async for chunk in wrapped(*args, **gemini_kwargs):
            # Same logic as sync, but async
            if chunk.text:
                accumulated_text.append(chunk.text)

            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        accumulated_function_calls.append({
                            'name': part.function_call.name,
                            'args': dict(part.function_call.args)
                        })

            if chunk.usage_metadata:
                final_usage = GeminiDataExtractor.extract_usage(chunk)

            yield chunk

        output_data = {}
        if accumulated_text:
            output_data['text'] = ''.join(accumulated_text)
        if accumulated_function_calls:
            output_data['function_calls'] = accumulated_function_calls

        generation.update(output=output_data, usage_details=final_usage)
        generation.end()

    except Exception as e:
        generation.update(level="ERROR", status_message=str(e))
        generation.end()
        raise
```

#### 3.2.6 Registration Function

```python
def register(
    client: Optional[ElasticDash] = None,
    *,
    force: bool = False
):
    """
    Register ElasticDash wrappers for Gemini API.

    Args:
        client: ElasticDash client instance. If None, uses global client.
        force: If True, re-register even if already registered.

    Raises:
        ImportError: If google-generativeai is not installed
        RuntimeError: If ElasticDash client is not initialized

    Example:
        import elasticdash
        from elasticdash.gemini import register

        elasticdash.init(public_key="...", secret_key="...")
        register()

        import google.generativeai as genai
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("Hello")  # Traced!
    """
    global _is_registered, _elasticdash_client

    if _is_registered and not force:
        logger.debug("Gemini integration already registered")
        return

    if not GEMINI_AVAILABLE:
        raise ImportError(
            "google-generativeai is not installed. "
            "Install it with: pip install google-generativeai"
        )

    # Get or validate client
    if client is None:
        from elasticdash import elasticdash_singleton
        if elasticdash_singleton is None:
            raise RuntimeError(
                "ElasticDash client not initialized. "
                "Call elasticdash.init() before registering Gemini integration."
            )
        _elasticdash_client = elasticdash_singleton
    else:
        _elasticdash_client = client

    # Register all wrappers
    try:
        from wrapt import wrap_function_wrapper

        for definition in GEMINI_DEFINITIONS:
            target = f"{definition.object}.{definition.method}"
            logger.debug(f"Registering wrapper for {definition.module}.{target}")

            wrap_function_wrapper(
                module=definition.module,
                name=target,
                wrapper=_create_wrapper(definition)
            )

        _is_registered = True
        logger.info("Gemini integration registered successfully")

    except Exception as e:
        logger.error(f"Failed to register Gemini integration: {e}")
        raise


def unregister():
    """
    Unregister Gemini wrappers (for testing purposes).

    Note: This requires reimporting the google.generativeai module.
    """
    global _is_registered, _elasticdash_client

    _is_registered = False
    _elasticdash_client = None

    # Force reimport to remove wrappers
    import sys
    if 'google.generativeai' in sys.modules:
        del sys.modules['google.generativeai']
```

### 3.3 Package Exports

#### Modify: elasticdash/__init__.py

```python
# Add to imports
from elasticdash import gemini

# Add to __all__
__all__ = [
    # ... existing exports
    "gemini",
]
```

### 3.4 Langchain Integration Enhancement

#### Modify: elasticdash/langchain/utils.py

```python
# In _extract_model_name function, add:

def _extract_model_name(serialized: dict, kwargs: dict, chat_model: bool = False) -> Optional[str]:
    # ... existing code ...

    # Add Gemini detection
    if "gemini" in model_id_str.lower():
        # Handle both "models/gemini-1.5-pro" and "gemini-1.5-pro"
        if "models/" in model_str:
            return model_str.split("models/")[1]
        return model_str

    # ... rest of function
```

---

## 4. Testing Strategy

### 4.1 Test File Structure

```python
# tests/test_gemini.py

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import google.generativeai as genai
from elasticdash import ElasticDash
from elasticdash.gemini import register, unregister

# Test fixtures
@pytest.fixture
def elasticdash_client():
    """Create test ElasticDash client."""
    client = ElasticDash(
        public_key=os.environ.get("ELASTICDASH_PUBLIC_KEY", "test-key"),
        secret_key=os.environ.get("ELASTICDASH_SECRET_KEY", "test-secret")
    )
    yield client
    client.shutdown()

@pytest.fixture
def gemini_model():
    """Create configured Gemini model."""
    api_key = os.environ.get("GOOGLE_API_KEY", "test-api-key")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-pro')

@pytest.fixture(autouse=True)
def setup_teardown():
    """Ensure clean state for each test."""
    unregister()
    yield
    unregister()
```

### 4.2 Test Cases

```python
class TestGeminiIntegration:
    """Test basic Gemini integration."""

    def test_simple_generation(self, elasticdash_client, gemini_model):
        """Test basic text generation is traced."""
        register(elasticdash_client)

        response = gemini_model.generate_content("Hello, how are you?")

        assert response.text
        # Verify observation was created
        elasticdash_client.flush()
        # Check via API or mock

    def test_streaming_generation(self, elasticdash_client, gemini_model):
        """Test streaming response is traced."""
        register(elasticdash_client)

        chunks = []
        for chunk in gemini_model.generate_content("Count to 5", stream=True):
            chunks.append(chunk.text)

        assert len(chunks) > 0
        full_text = ''.join(chunks)
        assert full_text

    @pytest.mark.asyncio
    async def test_async_generation(self, elasticdash_client, gemini_model):
        """Test async generation is traced."""
        register(elasticdash_client)

        response = await gemini_model.generate_content_async("What is AI?")

        assert response.text

    def test_function_calling(self, elasticdash_client, gemini_model):
        """Test function calling is captured."""
        register(elasticdash_client)

        # Define a function
        get_weather = genai.Tool(
            function_declarations=[
                genai.FunctionDeclaration(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                )
            ]
        )

        model = genai.GenerativeModel('gemini-1.5-pro', tools=[get_weather])
        response = model.generate_content("What's the weather in Paris?")

        # Should contain function call
        assert response.candidates[0].content.parts[0].function_call

    def test_multimodal_input(self, elasticdash_client, gemini_model):
        """Test multimodal input (image) is traced."""
        register(elasticdash_client)

        # Load test image
        import PIL.Image
        img = PIL.Image.new('RGB', (100, 100), color='red')

        response = gemini_model.generate_content([
            "What color is this image?",
            img
        ])

        assert response.text

    def test_custom_kwargs(self, elasticdash_client, gemini_model):
        """Test ElasticDash custom kwargs are passed."""
        register(elasticdash_client)

        response = gemini_model.generate_content(
            "Hello",
            elasticdash_trace_id="custom-trace-123",
            elasticdash_user_id="user-456",
            elasticdash_tags=["test", "gemini"]
        )

        assert response.text
        # Verify observation has correct metadata

    def test_error_handling(self, elasticdash_client, gemini_model):
        """Test errors are logged to observations."""
        register(elasticdash_client)

        with pytest.raises(Exception):
            # Trigger an error (e.g., invalid request)
            gemini_model.generate_content("")

        # Observation should have ERROR level

    def test_safety_ratings(self, elasticdash_client, gemini_model):
        """Test safety ratings are captured."""
        register(elasticdash_client)

        response = gemini_model.generate_content("Write a story about...")

        # Check safety ratings in metadata
        assert response.candidates[0].safety_ratings


class TestGeminiParameters:
    """Test model parameter extraction."""

    def test_temperature_parameter(self, elasticdash_client):
        """Test temperature is captured."""
        register(elasticdash_client)

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            'gemini-1.5-pro',
            generation_config=genai.GenerationConfig(temperature=0.7)
        )

        response = model.generate_content("Hello")

        # Verify observation has temperature parameter

    def test_max_tokens_parameter(self, elasticdash_client):
        """Test max_output_tokens is captured."""
        register(elasticdash_client)

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            'gemini-1.5-pro',
            generation_config=genai.GenerationConfig(max_output_tokens=100)
        )

        response = model.generate_content("Count to 100")

        # Verify parameter captured


class TestGeminiUsageTracking:
    """Test token usage tracking."""

    def test_usage_metadata(self, elasticdash_client, gemini_model):
        """Test token counts are captured."""
        register(elasticdash_client)

        response = gemini_model.generate_content("Hello, world!")

        assert response.usage_metadata
        assert response.usage_metadata.prompt_token_count > 0
        assert response.usage_metadata.candidates_token_count > 0
        # Verify observation has correct usage


@pytest.mark.skip("Requires real API key - run manually")
class TestGeminiE2E:
    """End-to-end tests with real API (skip by default)."""

    def test_real_api_call(self, elasticdash_client):
        """Test with real Gemini API."""
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        register(elasticdash_client)

        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content("What is the capital of France?")

        assert "Paris" in response.text

        # Flush and verify on ElasticDash backend
        elasticdash_client.flush()
```

### 4.3 Test Execution

```bash
# Run all tests (mocked)
poetry run pytest tests/test_gemini.py -v

# Run E2E tests (requires API keys)
export GOOGLE_API_KEY="your-key"
export ELASTICDASH_PUBLIC_KEY="your-key"
export ELASTICDASH_SECRET_KEY="your-secret"
poetry run pytest tests/test_gemini.py -v --run-e2e

# Run with coverage
poetry run pytest tests/test_gemini.py --cov=elasticdash.gemini --cov-report=html
```

---

## 5. Dependencies

### 5.1 Update pyproject.toml

```toml
[tool.poetry.dependencies]
# ... existing dependencies ...

# Optional Gemini support
google-generativeai = { version = "^0.8.0", optional = true }

[tool.poetry.extras]
# ... existing extras ...
gemini = ["google-generativeai"]

# Or add to "all" extra
all = [
    # ... existing
    "google-generativeai",
]
```

### 5.2 Installation

```bash
# Install with Gemini support
poetry install --extras gemini

# Or install all extras
poetry install --all-extras
```

---

## 6. Documentation

### 6.1 README Updates

Add to main README.md:

```markdown
### Google Gemini

```python
import elasticdash
from elasticdash.gemini import register
import google.generativeai as genai

# Initialize ElasticDash
elasticdash.init(
    public_key="pk-...",
    secret_key="sk-..."
)

# Register Gemini integration
register()

# Configure Gemini
genai.configure(api_key="YOUR_GOOGLE_API_KEY")

# Use Gemini normally - automatically traced!
model = genai.GenerativeModel('gemini-1.5-pro')
response = model.generate_content("Explain quantum computing")
print(response.text)

# Streaming
for chunk in model.generate_content("Count to 10", stream=True):
    print(chunk.text, end='')

# Function calling
tools = [...]
model_with_tools = genai.GenerativeModel('gemini-1.5-pro', tools=tools)
response = model_with_tools.generate_content("What's the weather?")
```

#### Custom Trace Attributes

```python
response = model.generate_content(
    "Hello",
    elasticdash_trace_id="custom-trace-id",
    elasticdash_user_id="user-123",
    elasticdash_session_id="session-456",
    elasticdash_tags=["production", "chatbot"],
    elasticdash_metadata={"context": "customer_support"}
)
```
```

### 6.2 API Documentation

Generate with pdoc:

```bash
poetry run pdoc elasticdash.gemini --docformat google
```

---

## 7. Alternative Approach: Using Existing @observe Decorator

### 7.1 User-Side Integration (Simpler)

If users prefer not to use automatic wrapping, they can use the `@observe` decorator:

```python
import elasticdash
from elasticdash import observe
import google.generativeai as genai

elasticdash.init(public_key="...", secret_key="...")
genai.configure(api_key="...")

@observe(as_type="generation")
def call_gemini(prompt: str):
    """Manually traced Gemini call."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

# Usage
result = call_gemini("What is AI?")
```

**Pros**:
- No wrapping needed
- Works immediately
- User has full control

**Cons**:
- Not automatic
- Requires wrapping every call
- Less detailed metadata (no automatic usage extraction)

### 7.2 Hybrid Approach

Document both approaches:
1. **Automatic** (recommended): Use `register()` for hands-off tracing
2. **Manual** (flexible): Use `@observe()` for custom control

---

## 8. Implementation Timeline

### Phase 1: Core Implementation (Week 1)
- [ ] Create `elasticdash/gemini.py` skeleton
- [ ] Implement `GeminiDefinition` and registration
- [ ] Implement basic wrapper for `generate_content`
- [ ] Test with simple text generation

### Phase 2: Advanced Features (Week 2)
- [ ] Streaming support (sync + async)
- [ ] Function calling support
- [ ] Multimodal input handling
- [ ] Usage and safety rating extraction

### Phase 3: Testing (Week 3)
- [ ] Unit tests with mocks
- [ ] Integration tests
- [ ] E2E tests with real API
- [ ] Coverage > 80%

### Phase 4: Documentation & Release (Week 4)
- [ ] Update README.md
- [ ] Generate API docs
- [ ] Create examples
- [ ] Update pyproject.toml
- [ ] Release as minor version bump

---

## 9. Potential Challenges & Solutions

### 9.1 Challenge: Gemini SDK Version Changes

**Problem**: Google frequently updates the generativeai SDK

**Solution**:
- Use version guards like OpenAI integration
- Test against multiple versions (0.7.x, 0.8.x)
- Use `hasattr()` checks for optional features

```python
if hasattr(response, 'usage_metadata'):
    usage = extract_usage(response.usage_metadata)
```

### 9.2 Challenge: Multimodal Content Serialization

**Problem**: Images/audio can't be JSON-serialized directly

**Solution**:
- Store metadata about media (type, size, format)
- Don't serialize raw bytes
- Provide URL if available

```python
if isinstance(part, PIL.Image.Image):
    prompt_data['media'].append({
        'type': 'image',
        'size': part.size,
        'format': part.format
    })
```

### 9.3 Challenge: Streaming Token Counts

**Problem**: Token counts only available at end of stream

**Solution**:
- Accumulate chunks
- Update observation after stream completes
- Use `completion_start_time` for time-to-first-token

### 9.4 Challenge: Function Calling Format Differences

**Problem**: Gemini's function calling format differs from OpenAI

**Solution**:
- Normalize to common structure
- Store original format in metadata
- Document format in API docs

---

## 10. Success Criteria

### 10.1 Functional Requirements
- ✅ Basic text generation traced automatically
- ✅ Streaming responses captured completely
- ✅ Function calls extracted and logged
- ✅ Token usage tracked accurately
- ✅ Custom ElasticDash kwargs supported
- ✅ Error handling with proper logging

### 10.2 Quality Requirements
- ✅ Test coverage > 80%
- ✅ No breaking changes to existing SDK
- ✅ Performance overhead < 5% per request
- ✅ Documentation complete and accurate

### 10.3 User Experience
- ✅ One-line registration: `register()`
- ✅ Zero code changes to existing Gemini usage
- ✅ Clear error messages
- ✅ Works with Langchain + Gemini

---

## 11. Future Enhancements

### 11.1 Cost Tracking
- Implement cost calculation for Gemini models
- Add pricing table for different model tiers
- Track cached content cost savings

### 11.2 Advanced Features
- Grounding with Google Search integration
- Code execution tracking
- Batch prediction support
- Tuned model support

### 11.3 Performance
- Async background processing for large responses
- Media upload to ElasticDash storage
- Sampling for high-volume applications

---

## 12. Questions for Clarification

Before implementation, clarify:

1. **Cost tracking**: Do we need to implement cost estimation for Gemini? (Pricing is complex with caching)
2. **Media handling**: Should we upload images/audio to ElasticDash storage or just metadata?
3. **Langchain priority**: Is Langchain + Gemini support sufficient for MVP?
4. **Version support**: Which google-generativeai versions should we support? (0.7.x, 0.8.x, or both?)
5. **Prompt caching**: Should we track cached content separately in observations?

---

## 13. Appendix

### 13.1 Gemini API Reference

- **Docs**: https://ai.google.dev/docs
- **Python SDK**: https://github.com/google/generative-ai-python
- **Models**: gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro

### 13.2 Related Issues

- Link to any GitHub issues requesting Gemini support
- Related feature requests

### 13.3 Example Code

See `examples/gemini_example.py` (to be created) for complete examples.

---

**End of Plan**

This plan provides a comprehensive roadmap for implementing Gemini integration in the ElasticDash Python SDK following established patterns and best practices.
