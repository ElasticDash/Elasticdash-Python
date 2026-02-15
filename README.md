# ElasticDash Python SDK

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/elasticdash.svg?style=flat-square&label=pypi+elasticdash)](https://pypi.python.org/pypi/elasticdash)

## About ElasticDash

ElasticDash is a fork of [Langfuse](https://github.com/langfuse/langfuse-python), extended with additional observability features for the ElasticDash platform. We maintain compatibility with core Langfuse concepts while adding specialized instrumentation for production LLM applications.

**Original Project:** [Langfuse Python SDK](https://github.com/langfuse/langfuse-python)
**License:** MIT License (see [LICENSE](LICENSE) and [NOTICE](NOTICE) files for full terms and attribution)

### Key Differences from Langfuse

- Integrated with ElasticDash Logger platform (https://logger.elasticdash.com)
- Enhanced OpenTelemetry support for distributed tracing
- Custom instrumentation for additional LLM providers
- Extended tracing and observability features for production environments

## Installation

```bash
pip install elasticdash
```

Then set your ElasticDash credentials as environment variables so the SDK knows who you are.

```bash
ELASTICDASH_SECRET_KEY="sk-lf-..."
ELASTICDASH_PUBLIC_KEY="pk-lf-..."
ELASTICDASH_BASE_URL="https://logger.elasticdash.com"
```

Or if you are setting it in your Python files:

```python
import os

# Get keys for your project from ElasticDash
# Keys are provided directly by ElasticDash or can be generated in self-hosted setup
os.environ["ELASTICDASH_PUBLIC_KEY"] = "pk-lf-..."
os.environ["ELASTICDASH_SECRET_KEY"] = "sk-lf-..."
os.environ["ELASTICDASH_BASE_URL"] = "https://logger.elasticdash.com"


# Your OpenAI key
os.environ["OPENAI_API_KEY"] = "sk-proj-..."
```

At the moment, the Secret Key and Public Key will be provided by ElasticDash directly.

If you are self-hosting, you can get your Secret Key and Public Key by creating them in Elasticdash Logger repository.

## Usage

### General Tracing

In most cases, you can start tracing by initialising the ElasticDash client.

```python
from elasticdash import get_client

elasticdash = get_client()

# Verify connection
if elasticdash.auth_check():
    print("ElasticDash client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
```

### OpenAI SDK

Swap the regular OpenAI import to ElasticDash's OpenAI drop-in. It behaves like the regular OpenAI client while also recording each call for you.

```python
from elasticdash.openai import openai
```

Use the OpenAI SDK as you normally would. The wrapper captures the prompt, model and output and forwards everything to ElasticDash.

```python
completion = openai.chat.completions.create(
    name="test-chat",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a very accurate calculator. You output only the result of the calculation."},
        {"role": "user", "content": "1 + 1 = "}
    ],
    metadata={"someMetadataKey": "someValue"},
)
```

### Manual Observation

This can help you cover the LLM models that we are not currently supporting.

```python
from elasticdash import get_client

elasticdash = get_client()

# Verify connection
if elasticdash.auth_check():
    print("ElasticDash client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# Create a span using a context manager
with elasticdash.start_as_current_observation(as_type="span", name="process-request") as span:
    # Your processing logic here
    span.update(input="Llm input here")

    # Create a nested generation for an LLM call
    with elasticdash.start_as_current_observation(as_type="generation", name="llm-response", model="gpt-3.5-turbo") as generation:
        generation.update(input="Llm input here")
        # Your LLM call logic here
        span.update(output="Processing complete")
        generation.update(output="Generated response")

# All spans are automatically closed when exiting their context blocks


# Flush events in short-lived applications
elasticdash.flush()
```

### Sample Usage

Always include the http.method, http.route and http.body (optional) in the span.

Make sure to include the input and output of the LLM in the observation.

Each LLM call should have an observation, while all LLM calls should be under (contained by) the span.

```python
with elasticdash.start_as_current_span(
    name="POST /chat/gemini/send/",
) as span:
    span.update(metadata={
        "http.method": "POST",
        "http.route": "/chat/gemini/send/",
        "http.body": body
    })
    with elasticdash.start_as_current_observation(
        as_type="generation",
        name="gemini-response",
        model="gemini-2.5-flash",
        input={
            "message": user_message,
            "history": history
        }
    ) as generation:

        history.insert(0, {
            "role": "user",
            "parts": [{"text": system_instruction}]
        })

        # Start chat with history
        chat = genaiClient.chats.create(
            model="gemini-2.5-flash",
            history=history,
        )

        # Send message and get response
        response = chat.send_message(user_message)
        bot_response = response.text

        generation.update(output=bot_response)
```
