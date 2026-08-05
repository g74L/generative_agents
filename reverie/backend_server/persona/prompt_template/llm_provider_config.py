"""Canonical, content-free configuration for LLM transport selection."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


LEGACY_OPENAI = "LEGACY_OPENAI"
MODERN_OPENAI = "MODERN_OPENAI"
FAKE = "FAKE"

LEGACY_TRANSPORT = "OPENAI_SDK_0_X"
MODERN_TRANSPORT = "OPENAI_SDK_1_PLUS"
FAKE_TRANSPORT = "IN_MEMORY_FAKE"

LEGACY_SDK_MODE = "LEGACY"
MODERN_SDK_MODE = "MODERN"
FAKE_SDK_MODE = "FAKE"


@dataclass(frozen=True)
class LLMProviderConfig:
  provider_kind: str = LEGACY_OPENAI
  transport_kind: str = LEGACY_TRANSPORT
  sdk_mode: str = LEGACY_SDK_MODE
  sdk_retry_count: int = 0
  request_timeout_seconds: float = 600.0
  store_responses: bool = False

  def __post_init__(self):
    if self.provider_kind not in (LEGACY_OPENAI, MODERN_OPENAI, FAKE):
      raise ValueError(f"Unsupported provider_kind: {self.provider_kind}")
    if self.transport_kind not in (
        LEGACY_TRANSPORT, MODERN_TRANSPORT, FAKE_TRANSPORT):
      raise ValueError(f"Unsupported transport_kind: {self.transport_kind}")
    if self.sdk_mode not in (
        LEGACY_SDK_MODE, MODERN_SDK_MODE, FAKE_SDK_MODE):
      raise ValueError(f"Unsupported sdk_mode: {self.sdk_mode}")
    canonical_transport_and_mode = {
      LEGACY_OPENAI: (LEGACY_TRANSPORT, LEGACY_SDK_MODE),
      MODERN_OPENAI: (MODERN_TRANSPORT, MODERN_SDK_MODE),
      FAKE: (FAKE_TRANSPORT, FAKE_SDK_MODE),
    }
    expected = canonical_transport_and_mode[self.provider_kind]
    if (self.transport_kind, self.sdk_mode) != expected:
      raise ValueError(
        "provider_kind, transport_kind, and sdk_mode are inconsistent")
    if type(self.sdk_retry_count) is not int or self.sdk_retry_count < 0:
      raise ValueError("sdk_retry_count must be a non-negative integer")
    if (not isinstance(self.request_timeout_seconds, (int, float))
        or isinstance(self.request_timeout_seconds, bool)
        or self.request_timeout_seconds <= 0):
      raise ValueError("request_timeout_seconds must be positive")
    if type(self.store_responses) is not bool:
      raise ValueError("store_responses must be boolean")


DEFAULT_LLM_PROVIDER_CONFIG = LLMProviderConfig()
_runtime_llm_provider_config = ContextVar(
  "runtime_llm_provider_config", default=DEFAULT_LLM_PROVIDER_CONFIG)


def get_llm_provider_config():
  return _runtime_llm_provider_config.get()


def set_llm_provider_config(config):
  if not isinstance(config, LLMProviderConfig):
    raise TypeError("LLM provider config must be LLMProviderConfig")
  return _runtime_llm_provider_config.set(config)


def reset_llm_provider_config(token=None):
  if token is None:
    _runtime_llm_provider_config.set(DEFAULT_LLM_PROVIDER_CONFIG)
  else:
    _runtime_llm_provider_config.reset(token)


@contextmanager
def use_llm_provider_config(config):
  token = set_llm_provider_config(config)
  try:
    yield config
  finally:
    reset_llm_provider_config(token)


def modern_openai_config(request_timeout_seconds=600.0):
  """Return the default-off modern configuration with SDK retries disabled."""
  return LLMProviderConfig(
    provider_kind=MODERN_OPENAI,
    transport_kind=MODERN_TRANSPORT,
    sdk_mode=MODERN_SDK_MODE,
    sdk_retry_count=0,
    request_timeout_seconds=request_timeout_seconds,
    store_responses=False,
  )


def fake_provider_config():
  return LLMProviderConfig(
    provider_kind=FAKE,
    transport_kind=FAKE_TRANSPORT,
    sdk_mode=FAKE_SDK_MODE,
    sdk_retry_count=0,
    request_timeout_seconds=600.0,
    store_responses=False,
  )
