"""Environment variable definitions for ElasticDash OpenTelemetry integration.

This module defines environment variables used to configure the ElasticDash OpenTelemetry integration.
Each environment variable includes documentation on its purpose, expected values, and defaults.
"""

ELASTICDASH_TRACING_ENVIRONMENT = "ELASTICDASH_TRACING_ENVIRONMENT"
"""
.. envvar:: ELASTICDASH_TRACING_ENVIRONMENT

The tracing environment. Can be any lowercase alphanumeric string with hyphens and underscores that does not start with 'elasticdash'.

**Default value:** ``"default"``
"""

ELASTICDASH_RELEASE = "ELASTICDASH_RELEASE"
"""
.. envvar:: ELASTICDASH_RELEASE

Release number/hash of the application to provide analytics grouped by release.
"""


ELASTICDASH_PUBLIC_KEY = "ELASTICDASH_PUBLIC_KEY"
"""
.. envvar:: ELASTICDASH_PUBLIC_KEY

Public API key of ElasticDash project
"""

ELASTICDASH_SECRET_KEY = "ELASTICDASH_SECRET_KEY"
"""
.. envvar:: ELASTICDASH_SECRET_KEY

Secret API key of ElasticDash project
"""

ELASTICDASH_BASE_URL = "ELASTICDASH_BASE_URL"
"""
.. envvar:: ELASTICDASH_BASE_URL

Base URL of ElasticDash API. Can be set via `ELASTICDASH_BASE_URL` environment variable.

**Default value:** ``"https://logger.elasticdash.com"``
"""

ELASTICDASH_HOST = "ELASTICDASH_HOST"
"""
.. envvar:: ELASTICDASH_HOST

Deprecated. Use ELASTICDASH_BASE_URL instead. Host of ElasticDash API. Can be set via `ELASTICDASH_HOST` environment variable.

**Default value:** ``"https://logger.elasticdash.com"``
"""

ELASTICDASH_OTEL_TRACES_EXPORT_PATH = "ELASTICDASH_OTEL_TRACES_EXPORT_PATH"
"""
.. envvar:: ELASTICDASH_OTEL_TRACES_EXPORT_PATH

URL path on the configured host to export traces to.

**Default value:** ``/api/public/otel/v1/traces``
"""

ELASTICDASH_DEBUG = "ELASTICDASH_DEBUG"
"""
.. envvar:: ELASTICDASH_DEBUG

Enables debug mode for more verbose logging.

**Default value:** ``"False"``
"""

ELASTICDASH_TRACING_ENABLED = "ELASTICDASH_TRACING_ENABLED"
"""
.. envvar:: ELASTICDASH_TRACING_ENABLED

Enables or disables the ElasticDash client. If disabled, all observability calls to the backend will be no-ops. Default is True. Set to `False` to disable tracing.

**Default value:** ``"True"``
"""

ELASTICDASH_MEDIA_UPLOAD_THREAD_COUNT = "ELASTICDASH_MEDIA_UPLOAD_THREAD_COUNT"
"""
.. envvar:: ELASTICDASH_MEDIA_UPLOAD_THREAD_COUNT 

Number of background threads to handle media uploads from trace ingestion.

**Default value:** ``1``
"""

ELASTICDASH_FLUSH_AT = "ELASTICDASH_FLUSH_AT"
"""
.. envvar:: ELASTICDASH_FLUSH_AT

Max batch size until a new ingestion batch is sent to the API.
**Default value:** same as OTEL ``OTEL_BSP_MAX_EXPORT_BATCH_SIZE``
"""

ELASTICDASH_FLUSH_INTERVAL = "ELASTICDASH_FLUSH_INTERVAL"
"""
.. envvar:: ELASTICDASH_FLUSH_INTERVAL

Max delay in seconds until a new ingestion batch is sent to the API.
**Default value:** same as OTEL ``OTEL_BSP_SCHEDULE_DELAY``
"""

ELASTICDASH_SAMPLE_RATE = "ELASTICDASH_SAMPLE_RATE"
"""
.. envvar: ELASTICDASH_SAMPLE_RATE

Float between 0 and 1 indicating the sample rate of traces to bet sent to ElasticDash servers.

**Default value**: ``1.0``

"""
ELASTICDASH_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED = (
    "ELASTICDASH_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED"
)
"""
.. envvar: ELASTICDASH_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED

Default capture of function args, kwargs and return value when using the @observe decorator.

Having default IO capture enabled for observe decorated function may have a performance impact on your application
if large or deeply nested objects are attempted to be serialized. Set this value to `False` and use manual
input/output setting on your observation to avoid this.

**Default value**: ``True``
"""

ELASTICDASH_MEDIA_UPLOAD_ENABLED = "ELASTICDASH_MEDIA_UPLOAD_ENABLED"
"""
.. envvar: ELASTICDASH_MEDIA_UPLOAD_ENABLED

Controls whether media detection and upload is attempted by the SDK.

**Default value**: ``True``
"""

ELASTICDASH_TIMEOUT = "ELASTICDASH_TIMEOUT"
"""
.. envvar: ELASTICDASH_TIMEOUT

Controls the timeout for all API requests in seconds

**Default value**: ``5``
"""

ELASTICDASH_PROMPT_CACHE_DEFAULT_TTL_SECONDS = "ELASTICDASH_PROMPT_CACHE_DEFAULT_TTL_SECONDS"
"""
.. envvar: ELASTICDASH_PROMPT_CACHE_DEFAULT_TTL_SECONDS

Controls the default time-to-live (TTL) in seconds for cached prompts.
This setting determines how long prompt responses are cached before they expire.

**Default value**: ``60``
"""
