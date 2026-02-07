"""Span attribute management for ElasticDash OpenTelemetry integration.

This module defines constants and functions for managing OpenTelemetry span attributes
used by ElasticDash. It provides a structured approach to creating and manipulating
attributes for different span types (trace, span, generation) while ensuring consistency.

The module includes:
- Attribute name constants organized by category
- Functions to create attribute dictionaries for different entity types
- Utilities for serializing and processing attribute values
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from elasticdash._client.constants import (
    ObservationTypeGenerationLike,
    ObservationTypeSpanLike,
)
from elasticdash._utils.serializer import EventSerializer
from elasticdash.model import PromptClient
from elasticdash.types import MapValue, SpanLevel


class ElasticDashOtelSpanAttributes:
    # ElasticDash-Trace attributes
    TRACE_NAME = "elasticdash.trace.name"
    TRACE_USER_ID = "user.id"
    TRACE_SESSION_ID = "session.id"
    TRACE_TAGS = "elasticdash.trace.tags"
    TRACE_PUBLIC = "elasticdash.trace.public"
    TRACE_METADATA = "elasticdash.trace.metadata"
    TRACE_INPUT = "elasticdash.trace.input"
    TRACE_OUTPUT = "elasticdash.trace.output"

    # ElasticDash-observation attributes
    OBSERVATION_TYPE = "elasticdash.observation.type"
    OBSERVATION_METADATA = "elasticdash.observation.metadata"
    OBSERVATION_LEVEL = "elasticdash.observation.level"
    OBSERVATION_STATUS_MESSAGE = "elasticdash.observation.status_message"
    OBSERVATION_INPUT = "elasticdash.observation.input"
    OBSERVATION_OUTPUT = "elasticdash.observation.output"

    # ElasticDash-observation of type Generation attributes
    OBSERVATION_COMPLETION_START_TIME = "elasticdash.observation.completion_start_time"
    OBSERVATION_MODEL = "elasticdash.observation.model.name"
    OBSERVATION_MODEL_PARAMETERS = "elasticdash.observation.model.parameters"
    OBSERVATION_USAGE_DETAILS = "elasticdash.observation.usage_details"
    OBSERVATION_COST_DETAILS = "elasticdash.observation.cost_details"
    OBSERVATION_PROMPT_NAME = "elasticdash.observation.prompt.name"
    OBSERVATION_PROMPT_VERSION = "elasticdash.observation.prompt.version"

    # General
    ENVIRONMENT = "elasticdash.environment"
    RELEASE = "elasticdash.release"
    VERSION = "elasticdash.version"

    # Internal
    AS_ROOT = "elasticdash.internal.as_root"

    # Experiments
    EXPERIMENT_ID = "elasticdash.experiment.id"
    EXPERIMENT_NAME = "elasticdash.experiment.name"
    EXPERIMENT_DESCRIPTION = "elasticdash.experiment.description"
    EXPERIMENT_METADATA = "elasticdash.experiment.metadata"
    EXPERIMENT_DATASET_ID = "elasticdash.experiment.dataset.id"
    EXPERIMENT_ITEM_ID = "elasticdash.experiment.item.id"
    EXPERIMENT_ITEM_EXPECTED_OUTPUT = "elasticdash.experiment.item.expected_output"
    EXPERIMENT_ITEM_METADATA = "elasticdash.experiment.item.metadata"
    EXPERIMENT_ITEM_ROOT_OBSERVATION_ID = "elasticdash.experiment.item.root_observation_id"


def create_trace_attributes(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    version: Optional[str] = None,
    release: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    metadata: Optional[Any] = None,
    tags: Optional[List[str]] = None,
    public: Optional[bool] = None,
) -> dict:
    attributes = {
        ElasticDashOtelSpanAttributes.TRACE_NAME: name,
        ElasticDashOtelSpanAttributes.TRACE_USER_ID: user_id,
        ElasticDashOtelSpanAttributes.TRACE_SESSION_ID: session_id,
        ElasticDashOtelSpanAttributes.VERSION: version,
        ElasticDashOtelSpanAttributes.RELEASE: release,
        ElasticDashOtelSpanAttributes.TRACE_INPUT: _serialize(input),
        ElasticDashOtelSpanAttributes.TRACE_OUTPUT: _serialize(output),
        ElasticDashOtelSpanAttributes.TRACE_TAGS: tags,
        ElasticDashOtelSpanAttributes.TRACE_PUBLIC: public,
        **_flatten_and_serialize_metadata(metadata, "trace"),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def create_span_attributes(
    *,
    metadata: Optional[Any] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    level: Optional[SpanLevel] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    observation_type: Optional[
        Union[ObservationTypeSpanLike, Literal["event"]]
    ] = "span",
) -> dict:
    attributes = {
        ElasticDashOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        ElasticDashOtelSpanAttributes.OBSERVATION_LEVEL: level,
        ElasticDashOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        ElasticDashOtelSpanAttributes.VERSION: version,
        ElasticDashOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        ElasticDashOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        **_flatten_and_serialize_metadata(metadata, "observation"),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def create_generation_attributes(
    *,
    name: Optional[str] = None,
    completion_start_time: Optional[datetime] = None,
    metadata: Optional[Any] = None,
    level: Optional[SpanLevel] = None,
    status_message: Optional[str] = None,
    version: Optional[str] = None,
    model: Optional[str] = None,
    model_parameters: Optional[Dict[str, MapValue]] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    usage_details: Optional[Dict[str, int]] = None,
    cost_details: Optional[Dict[str, float]] = None,
    prompt: Optional[PromptClient] = None,
    observation_type: Optional[ObservationTypeGenerationLike] = "generation",
) -> dict:
    attributes = {
        ElasticDashOtelSpanAttributes.OBSERVATION_TYPE: observation_type,
        ElasticDashOtelSpanAttributes.OBSERVATION_LEVEL: level,
        ElasticDashOtelSpanAttributes.OBSERVATION_STATUS_MESSAGE: status_message,
        ElasticDashOtelSpanAttributes.VERSION: version,
        ElasticDashOtelSpanAttributes.OBSERVATION_INPUT: _serialize(input),
        ElasticDashOtelSpanAttributes.OBSERVATION_OUTPUT: _serialize(output),
        ElasticDashOtelSpanAttributes.OBSERVATION_MODEL: model,
        ElasticDashOtelSpanAttributes.OBSERVATION_PROMPT_NAME: prompt.name
        if prompt and not prompt.is_fallback
        else None,
        ElasticDashOtelSpanAttributes.OBSERVATION_PROMPT_VERSION: prompt.version
        if prompt and not prompt.is_fallback
        else None,
        ElasticDashOtelSpanAttributes.OBSERVATION_USAGE_DETAILS: _serialize(usage_details),
        ElasticDashOtelSpanAttributes.OBSERVATION_COST_DETAILS: _serialize(cost_details),
        ElasticDashOtelSpanAttributes.OBSERVATION_COMPLETION_START_TIME: _serialize(
            completion_start_time
        ),
        ElasticDashOtelSpanAttributes.OBSERVATION_MODEL_PARAMETERS: _serialize(
            model_parameters
        ),
        **_flatten_and_serialize_metadata(metadata, "observation"),
    }

    return {k: v for k, v in attributes.items() if v is not None}


def _serialize(obj: Any) -> Optional[str]:
    if obj is None or isinstance(obj, str):
        return obj

    return json.dumps(obj, cls=EventSerializer)


def _flatten_and_serialize_metadata(
    metadata: Any, type: Literal["observation", "trace"]
) -> dict:
    prefix = (
        ElasticDashOtelSpanAttributes.OBSERVATION_METADATA
        if type == "observation"
        else ElasticDashOtelSpanAttributes.TRACE_METADATA
    )

    metadata_attributes: Dict[str, Union[str, int, None]] = {}

    if not isinstance(metadata, dict):
        metadata_attributes[prefix] = _serialize(metadata)
    else:
        for key, value in metadata.items():
            metadata_attributes[f"{prefix}.{key}"] = (
                value
                if isinstance(value, str) or isinstance(value, int)
                else _serialize(value)
            )

    return metadata_attributes
