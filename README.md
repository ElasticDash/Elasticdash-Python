# ElasticDash Python SDK

[![MIT License](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/elasticdash.svg?style=flat-square&label=pypi+elasticdash)](https://pypi.python.org/pypi/elasticdash)

## Installation

```
pip install elasticdash
```

Then set your ElasticDash credentials as environment variables so the SDK knows who you are.

```
ELASTICDASH_SECRET_KEY = "sk-lf-..."
ELASTICDASH_PUBLIC_KEY = "pk-lf-..."
ELASTICDASH_BASE_URL = "https://devserver-logger.elasticdash.com"
```

At the moment, the Secret Key and Public Key will be provided by ElasticDash directly.

## Usage

### OpenAI SDK

Swap the regular OpenAI import to Langfuse’s OpenAI drop-in. It behaves like the regular OpenAI client while also recording each call for you.

```
from elasticdash.openai import openai
```

Use the OpenAI SDK as you normally would. The wrapper captures the prompt, model and output and forwards everything to ElasticDash.

```
completion = openai.chat.completions.create(
  name="test-chat",
  model="gpt-4o",
  messages=[
      {"role": "system", "content": "You are a very accurate calculator. You output only the result of the calculation."},
      {"role": "user", "content": "1 + 1 = "}],
  metadata={"someMetadataKey": "someValue"},
)
```

### Manual Observation

This can help you cover the LLM models that we are not currently supporting.

```
from elasticdash import get_client
 
elasticdash = get_client()
 
# Create a span using a context manager
with elasticdash.start_as_current_observation(as_type="span", name="process-request") as span:
    # Your processing logic here
    span.update(output="Processing complete")
 
    # Create a nested generation for an LLM call
    with elasticdash.start_as_current_observation(as_type="generation", name="llm-response", model="gpt-3.5-turbo") as generation:
        # Your LLM call logic here
        generation.update(output="Generated response")
 
# All spans are automatically closed when exiting their context blocks
 
 
# Flush events in short-lived applications
elasticdash.flush()
```
