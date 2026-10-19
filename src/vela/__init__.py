"""Vela -- a small agent runtime with explicit boundaries.

Three layers, each depending only on the one below:

    AgentLoop     channel-facing: sessions, compaction, lifecycle
    AgentRunner   model-facing: the tool loop and its recovery paths
    Provider/Tool backend capabilities

Nothing imports upward. That constraint is what keeps a backend swap from
touching routing logic, and it is the first thing to preserve when extending
this codebase.
"""

from __future__ import annotations

from vela.core.contracts import ModelResponse, ToolCall, ToolResult, Usage
from vela.core.loop import AgentLoop, Session, SessionStore, build_default_loop
from vela.core.runner import AgentRunner, RunResult, RunSpec, StreamingEmitter
from vela.observability import Instrumentor, Metrics, Span, Tracer
from vela.providers.anthropic import AnthropicProvider
from vela.providers.base import (
    AuthError,
    BillingError,
    ContextWindowError,
    GenerationConfig,
    Provider,
    ProviderError,
    RateLimitError,
    TransientError,
)
from vela.providers.caching import (
    CachePolicy,
    CacheStats,
    CacheTTL,
    estimate_savings,
)
from vela.providers.factory import (
    REGISTRY,
    UnknownProviderError,
    available_providers,
    create_provider,
)
from vela.providers.openai_compat import OpenAICompatProvider
from vela.server import ServerConfig, VelaServer
from vela.tools.base import FunctionTool, Tool, tool
from vela.tools.registry import ToolRegistry

__version__ = "0.4.0"

__all__ = [
    "AgentLoop",
    "Instrumentor",
    "Metrics",
    "Span",
    "Tracer",
    "CachePolicy",
    "CacheStats",
    "CacheTTL",
    "AnthropicProvider",
    "AgentRunner",
    "AuthError",
    "BillingError",
    "ContextWindowError",
    "FunctionTool",
    "GenerationConfig",
    "ModelResponse",
    "OpenAICompatProvider",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "RunResult",
    "RunSpec",
    "ServerConfig",
    "Session",
    "SessionStore",
    "StreamingEmitter",
    "Tool",
    "ToolCall",
    "REGISTRY",
    "ToolRegistry",
    "ToolResult",
    "TransientError",
    "VelaServer",
    "UnknownProviderError",
    "available_providers",
    "create_provider",
    "Usage",
    "estimate_savings",
    "build_default_loop",
    "tool",
]
