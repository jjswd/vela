"""Backend selection.

Provider choice is configuration, not a code change: the rest of the runtime
addresses backends through the :class:`Provider` contract and never branches on
which one is in use.
"""

from __future__ import annotations

from dataclasses import dataclass

from vela.providers.base import GenerationConfig, Provider
from vela.providers.caching import CachePolicy


class UnknownProviderError(ValueError):
    """Raised when the requested backend has no registered implementation."""


@dataclass(frozen=True)
class ProviderInfo:
    """Describes one backend for configuration validation."""

    name: str
    default_base_url: str
    env_key: str
    #: Whether the backend needs an API key. Local servers usually do not.
    requires_key: bool = True


#: Registered backends. The OpenAI-compatible entry covers every service that
#: speaks that wire format, which includes most hosted providers and local
#: runtimes such as Ollama or vLLM.
REGISTRY: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo(
        name="anthropic",
        default_base_url="https://api.anthropic.com",
        env_key="ANTHROPIC_API_KEY",
    ),
    "openai": ProviderInfo(
        name="openai",
        default_base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
    ),
    "deepseek": ProviderInfo(
        name="deepseek",
        default_base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
    ),
    "moonshot": ProviderInfo(
        name="moonshot",
        default_base_url="https://api.moonshot.cn/v1",
        env_key="MOONSHOT_API_KEY",
    ),
    "openrouter": ProviderInfo(
        name="openrouter",
        default_base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
    ),
    "ollama": ProviderInfo(
        name="ollama",
        default_base_url="http://127.0.0.1:11434/v1",
        env_key="",
        requires_key=False,
    ),
}

#: Backends whose native message format differs from the OpenAI shape and
#: therefore need a dedicated implementation.
_NATIVE_BACKENDS = {"anthropic"}


def available_providers() -> list[str]:
    return sorted(REGISTRY)


def create_provider(
    backend: str,
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    cache_policy: CachePolicy | None = None,
    expected_turns: int = 1,
) -> Provider:
    """Instantiate a backend from configuration.

    Raises for unknown names instead of silently falling back: a typo in the
    backend name would otherwise surface much later as a confusing 404 from an
    unrelated endpoint.
    """
    info = REGISTRY.get(backend)
    if info is None:
        raise UnknownProviderError(
            f"unknown provider {backend!r}; available: {', '.join(available_providers())}"
        )

    config = GenerationConfig(model=model)
    resolved_url = base_url or info.default_base_url

    if info.requires_key and not api_key:
        raise ValueError(
            f"provider {backend!r} requires an API key (set {info.env_key})"
        )

    if backend in _NATIVE_BACKENDS:
        from vela.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            config,
            api_key=api_key or "",
            base_url=resolved_url,
            timeout=timeout,
            cache_policy=cache_policy,
            expected_turns=expected_turns,
        )

    from vela.providers.openai_compat import OpenAICompatProvider

    # Local servers accept a placeholder key but still require the header.
    return OpenAICompatProvider(
        config,
        api_key=api_key or "not-needed",
        base_url=resolved_url,
        timeout=timeout,
    )