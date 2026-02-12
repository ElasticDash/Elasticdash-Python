""".. include:: ../README.md"""

from elasticdash.batch_evaluation import (
    BatchEvaluationResult,
    BatchEvaluationResumeToken,
    CompositeEvaluatorFunction,
    EvaluatorInputs,
    EvaluatorStats,
    MapperFunction,
)
from elasticdash.experiment import Evaluation

from ._client import client as _client_module
from ._client.attributes import ElasticDashOtelSpanAttributes
from ._client.constants import ObservationTypeLiteral
from ._client.get_client import get_client
from ._client.observe import observe
from ._client.propagation import propagate_attributes
from ._client.span import (
    ElasticDashAgent,
    ElasticDashChain,
    ElasticDashEmbedding,
    ElasticDashEvaluator,
    ElasticDashEvent,
    ElasticDashGeneration,
    ElasticDashGuardrail,
    ElasticDashRetriever,
    ElasticDashSpan,
    ElasticDashTool,
)

ElasticDash = _client_module.ElasticDash

__all__ = [
    "ElasticDash",
    "get_client",
    "observe",
    "propagate_attributes",
    "ObservationTypeLiteral",
    "ElasticDashSpan",
    "ElasticDashGeneration",
    "ElasticDashEvent",
    "ElasticDashOtelSpanAttributes",
    "ElasticDashAgent",
    "ElasticDashTool",
    "ElasticDashChain",
    "ElasticDashEmbedding",
    "ElasticDashEvaluator",
    "ElasticDashRetriever",
    "ElasticDashGuardrail",
    "Evaluation",
    "EvaluatorInputs",
    "MapperFunction",
    "CompositeEvaluatorFunction",
    "EvaluatorStats",
    "BatchEvaluationResumeToken",
    "BatchEvaluationResult",
    "experiment",
    "api",
]
