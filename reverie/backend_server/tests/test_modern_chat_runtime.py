from contextlib import contextmanager, redirect_stdout
from decimal import Decimal
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template.chat_runtime import (
  M5_REPLAY_CALLER_ALLOWLIST,
  M5_CHAT_MODEL,
  ModernChatCallerNotAllowedError,
  ModernChatModelMismatchError,
  ModernChatRequest,
  ModernChatRequestError,
  ModernChatRuntimeConfig,
  ModernChatRuntimeInactiveError,
  ModernChatValidationError,
  build_modern_chat_runtime_config,
  get_modern_chat_runtime_config,
  run_modern_chat,
  use_modern_chat_runtime,
  validate_modern_chat_caller,
)
from persona.prompt_template import gpt_structure, run_gpt_prompt
from persona.prompt_template.cost_ledger import (
  COMPLETE,
  ModelPricing,
  PricingSnapshot,
  build_cost_ledger_records,
  summarize_cost_ledger,
)
from persona.prompt_template.embedding_runtime import (
  M2_EMBEDDING_CALLER_ALLOWLIST,
  TEXT_EMBEDDING_3_SMALL_MODEL,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
  validate_modern_embedding_caller,
)
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION,
  EMBEDDING,
  ERROR,
  SUCCESS,
  FakeProvider,
  OpenAILegacyProvider,
  clear_telemetry,
  chat_completion,
  embedding,
  get_chat_provider,
  get_completion_provider,
  get_embedding_provider,
  get_embedding_cache_stats,
  get_llm_replay_context,
  get_provider,
  get_telemetry,
  reset_embedding_cache,
  reset_provider,
  text_completion,
  use_llm_replay_context,
  use_provider,
  LLMReplayContext,
  PLANNING,
)
from persona.prompt_template.llm_provider_config import (
  DEFAULT_LLM_PROVIDER_CONFIG,
  LEGACY_OPENAI,
  LEGACY_SDK_MODE,
  LEGACY_TRANSPORT,
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
  get_llm_provider_config,
  reset_llm_provider_config,
)
from persona.prompt_template.modern_openai_provider import (
  LLMAuthenticationError,
  LLMInvalidRequestError,
  LLMMalformedResponseError,
  LLMRateLimitError,
  LLMServerError,
  LLMTimeoutError,
  ModernChatResponseValidationError,
  ModernOpenAIProvider,
  NormalizedEmbeddingResponse,
  NormalizedTextResponse,
  NormalizedUsage,
  map_modern_sdk_error,
  normalize_chat_response,
)
from persona.prompt_template.replay_cost_guard import ReplayCostGuardError
from persona.prompt_template.replay_cost_guard import (
  ReplayCostCeilingExceededError,
)
from controlled_replay import ReplayLogicalCallLimitExceededError


class FakeModernChatAdapter:
  def __init__(self, *responses):
    self.responses = list(responses)
    self.calls = []

  def create_chat(self, **kwargs):
    self.calls.append(kwargs)
    if not self.responses:
      raise AssertionError("No fake modern Chat response configured")
    response = self.responses.pop(0)
    if isinstance(response, Exception):
      raise response
    return response


class FakeModernEmbeddingAdapter:
  def __init__(self):
    self.calls = []

  def create_embedding(self, **kwargs):
    self.calls.append(kwargs)
    vector = (1.0,) + (0.0,) * 1535
    return NormalizedEmbeddingResponse(
      vector=vector, model=TEXT_EMBEDDING_3_SMALL_MODEL,
      request_id="req-embedding", usage=NormalizedUsage(1, None))

  def create_chat(self, **kwargs):
    raise AssertionError("Embedding adapter reached by Chat")


def response(content="answer", model=M5_CHAT_MODEL, request_id="req-m5",
             finish_reason="stop", status="completed", usage=None):
  return NormalizedTextResponse(
    text=content,
    model=model,
    request_id=request_id,
    finish_reason=finish_reason,
    status=status,
    usage=usage or NormalizedUsage(20, 8, 3, 2),
  )


class ModernChatRuntimeTests(unittest.TestCase):
  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_cache()
    clear_telemetry()

  def tearDown(self):
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_cache()
    clear_telemetry()

  def config(self, **changes):
    return build_modern_chat_runtime_config(**changes)

  def request(self, messages=None, **changes):
    values = {"messages": messages or (
      {"role": "system", "content": "You are an agent."},
      {"role": "user", "content": "What should I do?"},
    )}
    values.update(changes)
    return ModernChatRequest(**values)

  def _execute_case(self, adapter, request=None, config=None, validator=None):
    with use_modern_chat_runtime(config or self.config(), adapter):
      return run_modern_chat(request or self.request(), validator=validator)

  def _legacy_completion(self, provider):
    provider.queue_completion_response("legacy completion")
    return text_completion(
      model="text-davinci-003", prompt="prompt", temperature=0,
      max_tokens=4, top_p=1, frequency_penalty=0,
      presence_penalty=0, stream=False, stop=None)

  def test_01_valid_modern_chat_config_is_canonical(self):
    config = self.config()
    self.assertEqual((MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE,
                      M5_CHAT_MODEL, CHAT),
                     (config.provider_kind, config.transport_kind,
                      config.sdk_mode, config.chat_model, config.operation))

  def test_02_provider_transport_mismatch_is_rejected(self):
    with self.assertRaises(ValueError):
      ModernChatRuntimeConfig(transport_kind=LEGACY_TRANSPORT)

  def test_03_legacy_sdk_mode_is_rejected(self):
    with self.assertRaises(ValueError):
      ModernChatRuntimeConfig(sdk_mode=LEGACY_SDK_MODE)

  def test_04_missing_chat_model_is_rejected(self):
    with self.assertRaises(ValueError):
      ModernChatRuntimeConfig(chat_model="")

  def test_05_completion_model_as_chat_is_rejected(self):
    with self.assertRaises(ValueError):
      ModernChatRuntimeConfig(chat_model="text-davinci-003")

  def test_06_unsupported_operation_is_rejected(self):
    with self.assertRaises(ValueError):
      ModernChatRuntimeConfig(operation=COMPLETION)

  def test_07_factory_selects_modern_provider_for_chat(self):
    adapter = FakeModernChatAdapter(response())
    with use_modern_chat_runtime(self.config(), adapter) as provider:
      self.assertIsInstance(provider, ModernOpenAIProvider)
      self.assertIs(provider, get_chat_provider())

  def test_08_legacy_chat_provider_is_not_reached(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      self._execute_case(FakeModernChatAdapter(response()))
    self.assertEqual([], legacy.calls)

  def test_09_default_chat_provider_remains_legacy(self):
    self.assertIsInstance(get_chat_provider(), OpenAILegacyProvider)
    self.assertEqual(LEGACY_OPENAI, get_llm_provider_config().provider_kind)

  def test_10_completion_stays_on_general_provider(self):
    legacy = FakeProvider()
    legacy.queue_completion_response("legacy completion")
    adapter = FakeModernChatAdapter(response())
    with use_provider(legacy), use_modern_chat_runtime(self.config(), adapter):
      result = text_completion(
        model="text-davinci-003", prompt="p", temperature=0.1,
        max_tokens=4, top_p=1, frequency_penalty=0,
        presence_penalty=0, stream=False, stop=None)
    self.assertEqual("legacy completion", result.choices[0].text)
    self.assertEqual(COMPLETION, legacy.calls[0].operation)

  def test_11_embedding_stays_on_general_provider(self):
    legacy = FakeProvider(embedding_space_provider="openai")
    legacy.queue_embedding_response([1.0])
    before = get_embedding_cache_stats()
    with use_provider(legacy), use_modern_chat_runtime(
        self.config(), FakeModernChatAdapter(response())):
      result = embedding(input=["legacy embedding"],
                         model="text-embedding-ada-002")
    self.assertEqual([1.0], result["data"][0]["embedding"])
    self.assertEqual(EMBEDDING, legacy.calls[0].operation)
    self.assertEqual(before.logical_embedding_requests + 1,
                     get_embedding_cache_stats().logical_embedding_requests)

  def test_12_model_is_forwarded_unchanged(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter)
    self.assertEqual(M5_CHAT_MODEL, adapter.calls[0]["model"])

  def test_13_system_role_is_forwarded(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter)
    self.assertEqual("system", adapter.calls[0]["messages"][0]["role"])

  def test_14_user_role_is_forwarded(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter)
    self.assertEqual("user", adapter.calls[0]["messages"][1]["role"])

  def test_15_assistant_role_is_forwarded(self):
    messages = ({"role": "assistant", "content": "prior"},)
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(messages))
    self.assertEqual(messages[0], adapter.calls[0]["messages"][0])

  def test_16_message_order_is_preserved(self):
    messages = tuple({"role": role, "content": str(index)} for index, role in
                     enumerate(("system", "user", "assistant", "user")))
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(messages))
    self.assertEqual(list(messages), adapter.calls[0]["messages"])

  def test_17_message_content_and_whitespace_are_unchanged(self):
    text = "  Exact\nContent  "
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(({"role": "user", "content": text},)))
    self.assertEqual(text, adapter.calls[0]["messages"][0]["content"])

  def test_18_temperature_is_forwarded(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(temperature=0.25))
    self.assertEqual(0.25, adapter.calls[0]["temperature"])

  def test_19_max_tokens_is_forwarded(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(max_tokens=77))
    self.assertEqual(77, adapter.calls[0]["max_tokens"])

  def test_20_stop_is_forwarded(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(stop=("END", "STOP")))
    self.assertEqual(("END", "STOP"), adapter.calls[0]["stop"])

  def test_21_response_format_is_forwarded(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(response_format={"type": "json_object"}))
    self.assertEqual({"type": "json_object"},
                     adapter.calls[0]["response_format"])

  def test_22_response_content_is_normalized(self):
    result = self._execute_case(FakeModernChatAdapter(response(" exact output ")))
    self.assertEqual(" exact output ", result.content)

  def test_23_finish_reason_is_normalized(self):
    result = self._execute_case(FakeModernChatAdapter(response(finish_reason="stop")))
    self.assertEqual("stop", result.finish_reason)

  def test_24_request_id_is_normalized(self):
    result = self._execute_case(FakeModernChatAdapter(response(request_id="req-exact")))
    self.assertEqual("req-exact", result.request_id)

  def test_25_input_usage_is_normalized(self):
    self.assertEqual(20, self._execute_case(FakeModernChatAdapter(response())).usage.input_tokens)

  def test_26_output_usage_is_normalized(self):
    self.assertEqual(8, self._execute_case(FakeModernChatAdapter(response())).usage.output_tokens)

  def test_27_cached_usage_is_normalized(self):
    self.assertEqual(3, self._execute_case(FakeModernChatAdapter(response())).usage.cached_input_tokens)

  def test_28_reasoning_usage_is_normalized(self):
    self.assertEqual(2, self._execute_case(FakeModernChatAdapter(response())).usage.reasoning_tokens)

  def test_28b_total_usage_is_normalized(self):
    self.assertEqual(28, self._execute_case(
      FakeModernChatAdapter(response())).usage.total_tokens)

  def test_29_missing_usage_remains_unknown(self):
    result = self._execute_case(FakeModernChatAdapter(
      response(usage=NormalizedUsage())))
    self.assertEqual((None, None, None, None), (
      result.usage.input_tokens, result.usage.output_tokens,
      result.usage.cached_input_tokens, result.usage.reasoning_tokens))

  def test_30_success_telemetry_is_complete(self):
    self._execute_case(FakeModernChatAdapter(response()))
    event = get_telemetry()[0]
    self.assertEqual((CHAT, MODERN_OPENAI, MODERN_TRANSPORT, SUCCESS,
                      M5_CHAT_MODEL, "req-m5", 20, 8, 3, 2),
                     (event.operation, event.provider_kind,
                      event.transport_kind, event.outcome,
                      event.model_or_engine, event.request_id,
                      event.input_tokens, event.output_tokens,
                      event.cached_input_tokens, event.reasoning_tokens))

  def test_31_provider_error_telemetry_is_typed(self):
    adapter = FakeModernChatAdapter(LLMAuthenticationError("redacted"))
    with self.assertRaises(LLMAuthenticationError):
      self._execute_case(adapter)
    event = get_telemetry()[0]
    self.assertEqual((ERROR, "LLMAuthenticationError", MODERN_OPENAI),
                     (event.outcome, event.error_type, event.provider_kind))

  def test_32_timeout_is_typed_and_not_fallback(self):
    with self.assertRaises(LLMTimeoutError):
      self._execute_case(FakeModernChatAdapter(LLMTimeoutError("timeout")))

  def test_33_rate_limit_is_typed(self):
    with self.assertRaises(LLMRateLimitError):
      self._execute_case(FakeModernChatAdapter(LLMRateLimitError("limited")))

  def test_34_malformed_provider_result_is_typed(self):
    class MalformedProviderAdapter(FakeModernChatAdapter):
      pass
    adapter = MalformedProviderAdapter(response())
    with use_modern_chat_runtime(self.config(), adapter):
      with patch.object(get_chat_provider(), "chat_completion",
                        return_value={"choices": []}):
        with self.assertRaises(LLMMalformedResponseError):
          run_modern_chat(self.request())

  def test_35_transient_errors_retry_then_succeed(self):
    adapter = FakeModernChatAdapter(
      LLMRateLimitError("limited"), LLMTimeoutError("timeout"), response("ok"))
    result = self._execute_case(adapter, config=self.config(application_retry_count=2))
    self.assertEqual("ok", result.content)
    self.assertEqual(3, len(adapter.calls))

  def test_36_retry_exhaustion_propagates_last_error(self):
    adapter = FakeModernChatAdapter(
      LLMTimeoutError("one"), LLMTimeoutError("two"))
    with self.assertRaises(LLMTimeoutError):
      self._execute_case(adapter, config=self.config(application_retry_count=1))
    self.assertEqual(2, len(adapter.calls))

  def test_37_non_retryable_error_has_one_attempt(self):
    adapter = FakeModernChatAdapter(LLMInvalidRequestError("invalid"), response())
    with self.assertRaises(LLMInvalidRequestError):
      self._execute_case(adapter, config=self.config(application_retry_count=3))
    self.assertEqual(1, len(adapter.calls))

  def test_38_validator_failure_is_not_retried(self):
    adapter = FakeModernChatAdapter(response("invalid"), response("unused"))
    with self.assertRaises(ModernChatValidationError):
      self._execute_case(adapter, config=self.config(application_retry_count=2),
               validator=lambda content: False)
    self.assertEqual(1, len(adapter.calls))

  def test_39_retries_share_one_logical_call(self):
    adapter = FakeModernChatAdapter(LLMTimeoutError("one"), response())
    self._execute_case(adapter, config=self.config(application_retry_count=1))
    self.assertEqual(1, len({event.logical_call_id for event in get_telemetry()}))

  def test_40_physical_attempt_numbers_are_exact(self):
    adapter = FakeModernChatAdapter(LLMTimeoutError("one"), response())
    self._execute_case(adapter, config=self.config(application_retry_count=1))
    self.assertEqual([1, 2], [event.physical_attempt for event in get_telemetry()])

  def test_41_synthetic_cost_ledger_is_exact(self):
    usage = NormalizedUsage(100, 40, 10, 5)
    self._execute_case(FakeModernChatAdapter(response(usage=usage)))
    pricing = PricingSnapshot(
      "m5-synthetic", 1, "USD", "synthetic", (
        ModelPricing(M5_CHAT_MODEL, input_per_million=Decimal("2"),
                     cached_input_per_million=Decimal("1"),
                     output_per_million=Decimal("4")),), "synthetic only")
    records = build_cost_ledger_records(get_telemetry(), pricing)
    summary = summarize_cost_ledger(records)
    self.assertEqual(COMPLETE, records[0].token_usage_status)
    self.assertEqual(Decimal("0.000350000000"),
                     records[0].estimated_total_cost_usd)
    self.assertEqual(CHAT, summary.by_operation[0][0])
    self.assertEqual(MODERN_OPENAI, summary.by_provider[0][0])

  def test_42_telemetry_and_ledger_are_content_private(self):
    prompt = "M5-PRIVATE-PROMPT"
    output = "M5-PRIVATE-OUTPUT"
    self._execute_case(FakeModernChatAdapter(response(output)),
             self.request(({"role": "user", "content": prompt},)))
    serialized = repr((get_telemetry(), build_cost_ledger_records(get_telemetry())))
    self.assertNotIn(prompt, serialized)
    self.assertNotIn(output, serialized)

  def test_43_nested_context_restores_outer_runtime_and_provider(self):
    outer = self.config()
    outer_adapter = FakeModernChatAdapter(response("outer"), response("outer2"))
    inner_adapter = FakeModernChatAdapter(response("inner"))
    with use_modern_chat_runtime(outer, outer_adapter):
      outer_provider = get_chat_provider()
      with use_modern_chat_runtime(self.config(), inner_adapter):
        self.assertIsNot(outer_provider, get_chat_provider())
      self.assertIs(outer_provider, get_chat_provider())
      self.assertIs(outer, get_modern_chat_runtime_config())

  def test_44_exception_restores_legacy_context(self):
    before_provider = get_chat_provider()
    with self.assertRaises(RuntimeError):
      with use_modern_chat_runtime(self.config(), FakeModernChatAdapter()):
        raise RuntimeError("injected")
    self.assertIs(before_provider, get_chat_provider())
    self.assertIsNone(get_modern_chat_runtime_config())

  def test_45_general_provider_and_config_instances_are_unchanged(self):
    before_provider = get_provider()
    before_config = get_llm_provider_config()
    with use_modern_chat_runtime(self.config(), FakeModernChatAdapter(response())):
      self.assertIs(before_provider, get_provider())
      self.assertIs(before_config, get_llm_provider_config())
    self.assertIs(before_provider, get_provider())
    self.assertIs(before_config, get_llm_provider_config())

  def test_46_modern_error_never_falls_back_to_legacy(self):
    legacy = FakeProvider()
    legacy.queue_chat_response("forbidden fallback")
    adapter = FakeModernChatAdapter(LLMTimeoutError("modern failure"))
    with use_provider(legacy):
      with self.assertRaises(LLMTimeoutError):
        self._execute_case(adapter)
    self.assertEqual([], legacy.calls)

  def test_47_import_isolated_from_cognition_credentials_and_network(self):
    script = (
      "import importlib,json,socket,sys\n"
      f"sys.path.insert(0,{str(BACKEND_SERVER)!r})\n"
      "def blocked(*args,**kwargs): raise AssertionError('network')\n"
      "socket.getaddrinfo=blocked\n"
      "socket.create_connection=blocked\n"
      "importlib.import_module('persona.prompt_template.chat_runtime')\n"
      "print(json.dumps(sorted(sys.modules)))\n")
    completed = subprocess.run(
      [sys.executable, "-I", "-c", script], check=True,
      capture_output=True, text=True, timeout=30)
    modules = set(__import__("json").loads(completed.stdout))
    forbidden = {"gpt_structure", "planning", "reflection",
                 "conversation", "retrieve", "associative_memory"}
    self.assertFalse(any(name.rsplit(".", 1)[-1] in forbidden
                         for name in modules))
    self.assertFalse(any(name in (
      "utils", "reverie.backend_server.utils") for name in modules))

  def test_48_module_contains_no_sensitive_or_environment_lookup(self):
    source = (BACKEND_SERVER / "persona" / "prompt_template" /
              "chat_runtime.py").read_text(encoding="utf-8")
    self.assertNotIn("OPENAI" + "_API_KEY", source)
    self.assertNotIn("dotenv", source.lower())

  def test_49_socket_tripwire_confirms_zero_real_api_calls(self):
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("DNS reached")) as dns, patch.object(
        socket, "create_connection",
        side_effect=AssertionError("network reached")) as connection:
      self._execute_case(FakeModernChatAdapter(response()))
    dns.assert_not_called()
    connection.assert_not_called()

  def test_50_import_and_runtime_leave_legacy_defaults_unchanged(self):
    before = (get_provider(), get_chat_provider(), get_llm_provider_config())
    self._execute_case(FakeModernChatAdapter(response()))
    after = (get_provider(), get_chat_provider(), get_llm_provider_config())
    self.assertEqual(before, after)
    self.assertIs(DEFAULT_LLM_PROVIDER_CONFIG, after[2])

  def test_51_inactive_runtime_fails_before_provider_call(self):
    with self.assertRaises(ModernChatRuntimeInactiveError):
      run_modern_chat(self.request())

  def test_52_request_model_mismatch_fails_before_provider_call(self):
    adapter = FakeModernChatAdapter(response())
    with use_modern_chat_runtime(self.config(), adapter):
      with self.assertRaises(ModernChatModelMismatchError):
        run_modern_chat(self.request(model="gpt-3.5-turbo"))
    self.assertEqual([], adapter.calls)

  def test_53_invalid_message_roles_are_rejected(self):
    with self.assertRaises(ModernChatRequestError):
      self.request(({"role": "tool", "content": "no"},))

  def test_54_storage_mtime_is_unchanged(self):
    repository = BACKEND_SERVER.parents[1]
    storage = repository / "environment" / "frontend_server" / "storage"
    before = storage.stat().st_mtime_ns
    self._execute_case(FakeModernChatAdapter(response()))
    self.assertEqual(before, storage.stat().st_mtime_ns)

  def test_55_request_id_dict_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(request_id={"id": 1})))

  def test_56_request_id_list_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(request_id=["id"])))

  def test_57_request_id_bool_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(request_id=True)))

  def test_58_request_id_blank_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(request_id="   ")))

  def test_59_request_id_sentinel_is_private(self):
    sentinel = "RAW-REQUEST-ID-SENTINEL-M5R"
    raw = {"secret": sentinel}
    with self.assertRaises(ModernChatResponseValidationError) as raised:
      self._execute_case(FakeModernChatAdapter(response(request_id=raw)))
    exported = repr((raised.exception, get_telemetry(),
                     build_cost_ledger_records(get_telemetry())))
    self.assertNotIn(sentinel, exported)
    self.assertIsNone(get_telemetry()[0].request_id)

  def test_60_finish_reason_int_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(finish_reason=7)))

  def test_61_finish_reason_dict_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(
        response(finish_reason={"reason": "stop"})))

  def test_62_finish_reason_blank_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(finish_reason="\n")))

  def test_63_response_status_malformed_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(status=["done"])))

  def test_64_usage_input_string_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(
        response(usage=NormalizedUsage("20", 8))))

  def test_65_usage_bool_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(
        response(usage=NormalizedUsage(True, 8))))

  def test_66_usage_negative_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(
        response(usage=NormalizedUsage(20, -1))))

  def test_67_incoherent_provider_total_is_rejected(self):
    raw = {
      "choices": [{"message": {"content": "answer"},
                   "finish_reason": "stop"}],
      "model": M5_CHAT_MODEL,
      "usage": {"prompt_tokens": 2, "completion_tokens": 3,
                "total_tokens": 99},
    }
    with self.assertRaises(LLMMalformedResponseError):
      normalize_chat_response(raw)

  def test_68_malformed_metadata_records_error_only(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(request_id=False)))
    events = get_telemetry()
    self.assertEqual([ERROR], [event.outcome for event in events])

  def test_69_empty_injected_content_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(content="")))

  def test_70_blank_injected_content_is_rejected(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(content=" \n ")))

  def test_71_nonblank_whitespace_is_preserved(self):
    content = " \n Unicode ✓ and Markdown **ok** \n"
    self.assertEqual(content, self._execute_case(
      FakeModernChatAdapter(response(content=content))).content)

  def test_72_stop_list_is_accepted(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(stop=["END", "STOP"]))
    self.assertEqual(("END", "STOP"), adapter.calls[0]["stop"])

  def test_73_stop_list_order_is_preserved(self):
    adapter = FakeModernChatAdapter(response())
    self._execute_case(adapter, self.request(stop=["z", "a", "m"]))
    self.assertEqual(("z", "a", "m"), adapter.calls[0]["stop"])

  def test_74_stop_list_is_defensively_copied(self):
    source = ["END", "STOP"]
    request = self.request(stop=source)
    source[0] = "MUTATED"
    self.assertEqual(("END", "STOP"), request.stop)

  def test_75_non_string_stop_item_is_rejected(self):
    with self.assertRaises(ModernChatRequestError):
      self.request(stop=["END", 3])

  def test_76_empty_stop_list_is_rejected(self):
    with self.assertRaises(ModernChatRequestError):
      self.request(stop=[])

  def test_77_messages_are_deeply_immutable(self):
    message = {"role": "user", "content": "original"}
    source = [message]
    request = self.request(source)
    message["content"] = "mutated"
    source.append({"role": "assistant", "content": "new"})
    self.assertEqual("original", request.messages[0]["content"])
    self.assertEqual(1, len(request.messages))
    with self.assertRaises(TypeError):
      request.messages[0]["content"] = "blocked"

  def test_78_response_format_is_deeply_immutable(self):
    source = {"schema": {"required": ["name"]}}
    request = self.request(response_format=source)
    source["schema"]["required"].append("secret")
    self.assertEqual(("name",), request.response_format["schema"]["required"])
    with self.assertRaises(TypeError):
      request.response_format["schema"]["new"] = True

  def test_79_embedding_only_leaves_chat_legacy(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(),
          FakeModernEmbeddingAdapter()):
        self.assertIs(legacy, get_chat_provider())

  def test_80_embedding_only_leaves_completion_operational(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(),
          FakeModernEmbeddingAdapter()):
        self.assertEqual("legacy completion", self._legacy_completion(
          legacy).choices[0].text)

  def test_81_chat_only_leaves_embedding_legacy(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_modern_chat_runtime(
          self.config(), FakeModernChatAdapter(response())):
        self.assertIs(legacy, get_embedding_provider())

  def test_82_chat_only_leaves_completion_legacy(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_modern_chat_runtime(
          self.config(), FakeModernChatAdapter(response())):
        self.assertIs(legacy, get_completion_provider())
        self.assertEqual("legacy completion", self._legacy_completion(
          legacy).choices[0].text)

  def test_83_chat_and_embedding_providers_are_independent(self):
    legacy = FakeProvider()
    embedding_adapter = FakeModernEmbeddingAdapter()
    chat_adapter = FakeModernChatAdapter(response())
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), embedding_adapter):
        embedding_provider = get_embedding_provider()
        with use_modern_chat_runtime(self.config(), chat_adapter):
          self.assertIsNot(embedding_provider, get_chat_provider())
          self.assertIs(embedding_provider, get_embedding_provider())
          self.assertIs(legacy, get_completion_provider())

  def test_84_chat_reset_preserves_embedding_outer(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(),
          FakeModernEmbeddingAdapter()):
        outer = get_embedding_provider()
        with use_modern_chat_runtime(
            self.config(), FakeModernChatAdapter(response())):
          pass
        self.assertIs(outer, get_embedding_provider())
        self.assertIs(legacy, get_chat_provider())

  def test_85_embedding_reset_preserves_chat_outer(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_modern_chat_runtime(
          self.config(), FakeModernChatAdapter(response())):
        outer = get_chat_provider()
        with use_embedding_runtime(
            build_modern_embedding_runtime_config(),
            FakeModernEmbeddingAdapter()):
          pass
        self.assertIs(outer, get_chat_provider())
        self.assertIs(legacy, get_embedding_provider())

  def test_86_combined_exception_resets_all_providers(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with self.assertRaises(RuntimeError):
        with use_embedding_runtime(
            build_modern_embedding_runtime_config(),
            FakeModernEmbeddingAdapter()):
          with use_modern_chat_runtime(
              self.config(), FakeModernChatAdapter(response())):
            raise RuntimeError("injected")
      self.assertIs(legacy, get_chat_provider())
      self.assertIs(legacy, get_completion_provider())
      self.assertIs(legacy, get_embedding_provider())

  def test_87_combined_nesting_keeps_completion_legacy(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(),
          FakeModernEmbeddingAdapter()):
        with use_modern_chat_runtime(
            self.config(), FakeModernChatAdapter(response())):
          self.assertEqual("legacy completion", self._legacy_completion(
            legacy).choices[0].text)

  def test_88_provider_instances_do_not_leak(self):
    before = (get_chat_provider(), get_completion_provider(),
              get_embedding_provider(), get_provider())
    self._execute_case(FakeModernChatAdapter(response()))
    self.assertEqual(before, (get_chat_provider(), get_completion_provider(),
                             get_embedding_provider(), get_provider()))

  def test_89_valid_sdk_error_request_id_is_preserved(self):
    sdk_error = type("APITimeoutError", (Exception,), {})()
    sdk_error.request_id = "req-valid-error"
    mapped = map_modern_sdk_error(sdk_error)
    with self.assertRaises(LLMTimeoutError):
      self._execute_case(FakeModernChatAdapter(mapped))
    self.assertEqual("req-valid-error", get_telemetry()[0].request_id)

  def test_90_malformed_sdk_error_request_id_is_eliminated(self):
    sentinel = "SDK-RAW-ID-SENTINEL-M5R"
    sdk_error = type("RateLimitError", (Exception,), {})()
    sdk_error.request_id = {"secret": sentinel}
    mapped = map_modern_sdk_error(sdk_error)
    with self.assertRaises(LLMRateLimitError) as raised:
      self._execute_case(FakeModernChatAdapter(mapped))
    self.assertIsNone(mapped.request_id)
    self.assertNotIn(sentinel, repr((raised.exception, get_telemetry())))

  def test_91_requested_and_response_models_are_distinct(self):
    result = self._execute_case(FakeModernChatAdapter(
      response(model="gpt-4o-mini-provider-alias")))
    self.assertEqual(M5_CHAT_MODEL, result.requested_model)
    self.assertEqual("gpt-4o-mini-provider-alias", result.response_model)
    self.assertEqual(result.response_model, result.model)
    self.assertEqual(M5_CHAT_MODEL, get_telemetry()[0].model_or_engine)

  def test_92_raw_metadata_sentinels_are_not_exported(self):
    sentinels = ("RAW-FINISH-M5R", "RAW-STATUS-M5R", "RAW-USAGE-M5R")
    malformed = response(
      finish_reason={"secret": sentinels[0]},
      status={"secret": sentinels[1]},
      usage=NormalizedUsage({"secret": sentinels[2]}, 1))
    with self.assertRaises(ModernChatResponseValidationError) as raised:
      self._execute_case(FakeModernChatAdapter(malformed))
    exported = repr((raised.exception, get_telemetry(),
                     build_cost_ledger_records(get_telemetry())))
    for sentinel in sentinels:
      self.assertNotIn(sentinel, exported)

  def test_93_validation_failure_has_one_logical_event(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(status=True)))
    events = get_telemetry()
    self.assertEqual(1, len(events))
    self.assertEqual(1, len({event.logical_call_id for event in events}))

  def test_94_success_is_not_emitted_before_validation(self):
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(finish_reason=[])))
    self.assertNotIn(SUCCESS, [event.outcome for event in get_telemetry()])

  def test_95_metadata_validation_error_is_not_retried(self):
    adapter = FakeModernChatAdapter(
      response(request_id={"bad": True}), response("must-not-run"))
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(
        adapter, config=self.config(application_retry_count=3))
    self.assertEqual(1, len(adapter.calls))

  def test_96_m5r_matrix_performs_no_real_api_call(self):
    legacy = FakeProvider()
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("DNS reached")) as dns:
      with use_provider(legacy):
        self._execute_case(FakeModernChatAdapter(response()))
    dns.assert_not_called()

  def test_97_golden_defaults_remain_invariant(self):
    self.assertEqual("gpt-4o-mini", M5_CHAT_MODEL)
    self.assertIsInstance(get_chat_provider(), OpenAILegacyProvider)
    self.assertIs(DEFAULT_LLM_PROVIDER_CONFIG, get_llm_provider_config())

  def test_98_m3_embedding_runtime_remains_operational(self):
    legacy = FakeProvider()
    adapter = FakeModernEmbeddingAdapter()
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), adapter):
        result = embedding(
          input=["M5R embedding"], model=TEXT_EMBEDDING_3_SMALL_MODEL)
    self.assertEqual(1536, len(result["data"][0]["embedding"]))
    self.assertEqual(1, len(adapter.calls))

  def test_99_m4_bootstrap_module_is_not_imported(self):
    self.assertNotIn(
      "persona.memory_structures.embedding_store_bootstrap", sys.modules)

  def test_100_storage_tree_is_not_mutated(self):
    storage = BACKEND_SERVER.parents[1] / "environment" / "frontend_server" / "storage"
    before = sorted((path.relative_to(storage), path.stat().st_mtime_ns)
                    for path in storage.rglob("*") if path.is_file())
    with self.assertRaises(ModernChatResponseValidationError):
      self._execute_case(FakeModernChatAdapter(response(request_id={})))
    after = sorted((path.relative_to(storage), path.stat().st_mtime_ns)
                   for path in storage.rglob("*") if path.is_file())
    self.assertEqual(before, after)

  def test_101_testcase_run_is_not_overridden(self):
    source = Path(__file__).read_text(encoding="utf-8")
    self.assertNotIn("def " + "run(", source)
    self.assertNotIn("TestCase" + ".run", source)

  def test_102_rate_limit_error_preserves_valid_request_id(self):
    sdk_error = type("RateLimitError", (Exception,), {})()
    sdk_error.request_id = "req-valid-rate-limit"
    mapped = map_modern_sdk_error(sdk_error)
    self.assertIsInstance(mapped, LLMRateLimitError)
    self.assertEqual("req-valid-rate-limit", mapped.request_id)

  def test_103_server_error_preserves_valid_request_id_and_status(self):
    sdk_error = type("InternalServerError", (Exception,), {})()
    sdk_error.request_id = "req-valid-server"
    sdk_error.status_code = 503
    mapped = map_modern_sdk_error(sdk_error)
    self.assertIsInstance(mapped, LLMServerError)
    self.assertEqual("req-valid-server", mapped.request_id)
    self.assertEqual(503, mapped.provider_status)

  def _pronunciatio(self, adapter, action="painting"):
    persona = SimpleNamespace(name="Isabella Rodriguez")
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_pronunciatio(action, persona)
      return result[0] if result is not None else None

  @contextmanager
  def _backend_workdir(self):
    previous = Path.cwd()
    os.chdir(BACKEND_SERVER)
    try:
      yield
    finally:
      os.chdir(previous)

  def test_104_pronunciatio_uses_attributed_modern_chat(self):
    adapter = FakeModernChatAdapter(response('{"output":"🎨"}'))
    with use_llm_replay_context(LLMReplayContext(
        cognitive_category="UNSPECIFIED", actor_id="Isabella Rodriguez",
        simulation_id="offline-pronunciatio", simulation_step=0)):
      output = self._pronunciatio(adapter)
    event = get_telemetry()[0]
    self.assertEqual("🎨", output)
    self.assertEqual((M5_CHAT_MODEL, 0, 15, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    self.assertEqual(("pronunciatio", PLANNING),
                     (event.caller_id, event.cognitive_category))
    self.assertEqual((MODERN_OPENAI, MODERN_TRANSPORT),
                     (event.provider_kind, event.transport_kind))

  def test_105_pronunciatio_preserves_one_user_message_and_prompt(self):
    marker = "PRIVATE-PRONUNCIATIO-INPUT"
    adapter = FakeModernChatAdapter(response('{"output":"🎨"}'))
    self._pronunciatio(adapter, marker)
    messages = adapter.calls[0]["messages"]
    self.assertEqual(1, len(messages))
    self.assertEqual("user", messages[0]["role"])
    self.assertIn(marker, messages[0]["content"])
    self.assertIn('Example output json:', messages[0]["content"])
    self.assertNotIn(marker, repr(get_telemetry()))
    self.assertNotIn("🎨", repr(get_telemetry()))

  def test_106_pronunciatio_cleanup_remains_three_characters(self):
    adapter = FakeModernChatAdapter(response('{"output":"abcdef"}'))
    self.assertEqual("abc", self._pronunciatio(adapter))

  def test_107_invalid_response_uses_historical_three_attempts_and_fallback(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-not-json"), response("bad"))
    output = self._pronunciatio(adapter)
    self.assertIsNone(output)
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_108_invalid_then_valid_response_retries_semantically(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":"🎨"}'))
    self.assertEqual("🎨", self._pronunciatio(adapter))
    self.assertEqual(2, len(adapter.calls))

  def test_109_policy_error_is_rethrown_without_second_provider_call(self):
    adapter = FakeModernChatAdapter(
      LLMAuthenticationError("denied"), response('{"output":"must-not-run"}'))
    with self.assertRaises(LLMAuthenticationError):
      self._pronunciatio(adapter)
    self.assertEqual(1, len(adapter.calls))

  def test_110_replay_guard_error_is_not_swallowed(self):
    adapter = FakeModernChatAdapter(
      ReplayCostGuardError("tripped"), response('{"output":"must-not-run"}'))
    with self.assertRaises(ReplayCostGuardError):
      self._pronunciatio(adapter)
    self.assertEqual(1, len(adapter.calls))

  def test_110a_replay_logical_limit_error_is_not_swallowed(self):
    adapter = FakeModernChatAdapter(
      ReplayLogicalCallLimitExceededError(1),
      response('{"output":"must-not-run"}'))
    with self.assertRaises(ReplayLogicalCallLimitExceededError):
      self._pronunciatio(adapter)
    self.assertEqual(1, len(adapter.calls))

  def test_110b_cost_ceiling_error_is_not_swallowed(self):
    adapter = FakeModernChatAdapter(
      ReplayCostCeilingExceededError(
        Decimal("0.002"), Decimal("0.001"), CHAT, M5_CHAT_MODEL),
      response('{"output":"must-not-run"}'))
    with self.assertRaises(ReplayCostCeilingExceededError):
      self._pronunciatio(adapter)
    self.assertEqual(1, len(adapter.calls))

  def test_111_provider_configuration_error_is_not_swallowed(self):
    adapter = FakeModernChatAdapter(
      LLMInvalidRequestError("invalid configuration"),
      response('{"output":"must-not-run"}'))
    with self.assertRaises(LLMInvalidRequestError):
      self._pronunciatio(adapter)
    self.assertEqual(1, len(adapter.calls))

  def test_112_transient_provider_error_keeps_historical_retry(self):
    adapter = FakeModernChatAdapter(
      LLMTimeoutError("temporary"), response('{"output":"🎨"}'))
    self.assertEqual("🎨", self._pronunciatio(adapter))
    self.assertEqual(2, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_113_inactive_runtime_fails_closed_without_legacy_fallback(self):
    with patch.object(gpt_structure, "chat_completion",
                      side_effect=AssertionError("legacy Chat reached")) as legacy:
      with self._backend_workdir(), redirect_stdout(io.StringIO()), (
          self.assertRaises(ModernChatRuntimeInactiveError)):
        run_gpt_prompt.run_gpt_prompt_pronunciatio(
          "painting", SimpleNamespace(name="Isabella Rodriguez"))
    legacy.assert_not_called()

  def _event_poignancy(self, adapter, event="event",
                       persona_name="Isabella Rodriguez"):
    scratch = SimpleNamespace(
      name=persona_name, get_str_iss=lambda: "identity")
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_event_poignancy(
        SimpleNamespace(scratch=scratch), event)
      return result[0] if result is not None else None

  def test_114_event_poignancy_uses_attributed_modern_chat(self):
    adapter = FakeModernChatAdapter(response('{"output":"7"}'))
    context = LLMReplayContext(
      cognitive_category="PERCEPTION", actor_id="Isabella Rodriguez",
      simulation_id="offline-event-poignancy", simulation_step=1)
    with use_llm_replay_context(context):
      output = self._event_poignancy(adapter)
    event = get_telemetry()[0]
    ledger = build_cost_ledger_records(get_telemetry())[0]
    self.assertEqual(7, output)
    self.assertEqual((M5_CHAT_MODEL, 0, 15, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    self.assertEqual(("event_poignancy", CHAT, M5_CHAT_MODEL), (
      event.caller_id, event.operation, event.model_or_engine))
    self.assertEqual(("Isabella Rodriguez", "offline-event-poignancy", 1), (
      event.actor_id, event.simulation_id, event.simulation_step))
    self.assertEqual(("event_poignancy", PLANNING),
                     (ledger.caller_id, ledger.cognitive_category))
    self.assertEqual(
      ("pronunciatio", "act_obj_desc", "event_poignancy", "focal_pt",
       "agent_chat_summarize_relationship", "iterative_chat_utterance",
       "summarize_conversation", "chat_poignancy", "memo_on_convo"),
      M5_REPLAY_CALLER_ALLOWLIST)

  def test_115_pronunciatio_micro_run_is_one_call_without_legacy_detection(self):
    adapter = FakeModernChatAdapter(response('{"output":"🎨"}'))
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("network reached")) as dns:
      self.assertEqual("🎨", self._pronunciatio(adapter))
    events = get_telemetry()
    self.assertEqual(1, len(events))
    self.assertEqual((1, 1), (
      len({event.logical_call_id for event in events}),
      sum(event.physical_attempt for event in events)))
    self.assertEqual([M5_CHAT_MODEL], [event.model_or_engine for event in events])
    self.assertNotIn("gpt-3.5-turbo", [event.model_or_engine for event in events])
    dns.assert_not_called()
    self.assertIsNone(get_modern_chat_runtime_config())
    self.assertEqual(LLMReplayContext(), get_llm_replay_context())

  def _act_obj_desc(self, adapter, game_object="bed", action="sleeping",
                    persona_name="Isabella Rodriguez"):
    persona = SimpleNamespace(name=persona_name)
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_act_obj_desc(
        game_object, action, persona)
      return result[0] if result is not None else None

  def test_116_act_obj_desc_uses_attributed_modern_chat(self):
    adapter = FakeModernChatAdapter(response('{"output":"bed is made."}'))
    output = self._act_obj_desc(adapter)
    event = get_telemetry()[0]
    self.assertEqual("bed is made", output)
    self.assertIsInstance(output, str)
    self.assertEqual((M5_CHAT_MODEL, 0, 15, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    self.assertEqual(("act_obj_desc", PLANNING),
                     (event.caller_id, event.cognitive_category))
    self.assertEqual((MODERN_OPENAI, MODERN_TRANSPORT),
                     (event.provider_kind, event.transport_kind))

  def test_117_act_obj_desc_preserves_prompt_inputs_and_message_shape(self):
    markers = ("PRIVATE-OBJECT", "PRIVATE-ACTION", "PRIVATE-PERSONA")
    adapter = FakeModernChatAdapter(response('{"output":"state"}'))
    self._act_obj_desc(adapter, *markers)
    messages = adapter.calls[0]["messages"]
    self.assertEqual(1, len(messages))
    self.assertEqual("user", messages[0]["role"])
    for marker in markers:
      self.assertIn(marker, messages[0]["content"])
      self.assertNotIn(marker, repr(get_telemetry()))
    self.assertNotIn("state", repr(get_telemetry()))

  def test_118_act_obj_desc_exhaustion_returns_typed_failsafe(self):
    # R1CHAT-P5 / live B7 (Klaus Mueller, step 348): 3 physical attempts, all
    # invalid, must restore the historical fail-safe ("<object> is idle")
    # instead of the implicit ``None`` that produced 'NoneType' object is not
    # subscriptable in plan.py's ``run_gpt_prompt_act_obj_desc(...)[0]``
    # consumer.
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-invalid"), response("bad"))
    persona = SimpleNamespace(name="Isabella Rodriguez")
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_act_obj_desc(
        "bed", "sleeping", persona)
    self.assertIsNotNone(result)
    self.assertEqual(("bed is idle", "bed is idle"), (result[0], result[1][4]))
    self.assertIs(type(result[0]), str)
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))
    self.assertEqual(2, sum(
      1 for event in get_telemetry() if event.physical_attempt > 1))

  def test_118b_act_obj_desc_consumer_survives_retry_exhaustion(self):
    # Exercises the exact expression from plan.py's generate_act_obj_desc --
    # run_gpt_prompt_act_obj_desc(...)[0] -- the real call site that raised
    # 'NoneType' object is not subscriptable in live run
    # r1cli-a2-b-process-b7 (Klaus Mueller, step 348, caller=act_obj_desc).
    from persona.cognitive_modules import plan as plan_module
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-invalid"), response("bad"))
    persona = SimpleNamespace(name="Klaus Mueller")
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      desc = plan_module.generate_act_obj_desc(
        "pool table", "playing pool", persona)
    self.assertEqual("pool table is idle", desc)
    self.assertIs(type(desc), str)

  def test_119_act_obj_desc_invalid_then_valid_retries_semantically(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":"bed is made."}'))
    self.assertEqual("bed is made", self._act_obj_desc(adapter))
    self.assertEqual(2, len(adapter.calls))

  def test_120_act_obj_desc_policy_error_hard_fails(self):
    adapter = FakeModernChatAdapter(
      LLMAuthenticationError("denied"), response('{"output":"must-not-run"}'))
    with self.assertRaises(LLMAuthenticationError):
      self._act_obj_desc(adapter)
    self.assertEqual(1, len(adapter.calls))
    self.assertIsNone(get_modern_chat_runtime_config())
    self.assertEqual(LLMReplayContext(), get_llm_replay_context())

  def test_121_act_obj_desc_cost_and_logical_limits_are_not_swallowed(self):
    errors = (
      ReplayLogicalCallLimitExceededError(1),
      ReplayCostCeilingExceededError(
        Decimal("0.002"), Decimal("0.001"), CHAT, M5_CHAT_MODEL),
    )
    for error in errors:
      with self.subTest(error=type(error).__name__):
        clear_telemetry()
        adapter = FakeModernChatAdapter(
          error, response('{"output":"must-not-run"}'))
        with self.assertRaises(type(error)):
          self._act_obj_desc(adapter)
        self.assertEqual(1, len(adapter.calls))

  def test_122_act_obj_desc_transient_error_keeps_historical_retry(self):
    adapter = FakeModernChatAdapter(
      LLMTimeoutError("temporary"), response('{"output":"state"}'))
    self.assertEqual("state", self._act_obj_desc(adapter))
    self.assertEqual(2, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_123_act_obj_desc_one_call_no_network_or_legacy(self):
    adapter = FakeModernChatAdapter(response('{"output":"state"}'))
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("network reached")) as dns:
      self.assertEqual("state", self._act_obj_desc(adapter))
    events = get_telemetry()
    self.assertEqual(1, len(events))
    self.assertEqual((1, 1), (
      len({event.logical_call_id for event in events}), len(events)))
    self.assertEqual("act_obj_desc", events[0].caller_id)
    self.assertEqual(M5_CHAT_MODEL, events[0].model_or_engine)
    self.assertNotEqual("gpt-3.5-turbo", events[0].model_or_engine)
    dns.assert_not_called()

  def test_124_pronunciatio_remains_operational_with_exact_allowlist(self):
    self.assertEqual(
      "🎨", self._pronunciatio(
        FakeModernChatAdapter(response('{"output":"🎨"}'))))
    self.assertEqual(
      ("pronunciatio", "act_obj_desc", "event_poignancy", "focal_pt",
       "agent_chat_summarize_relationship", "iterative_chat_utterance",
       "summarize_conversation", "chat_poignancy", "memo_on_convo"),
      M5_REPLAY_CALLER_ALLOWLIST)

  def test_125_event_poignancy_preserves_prompt_and_privacy(self):
    marker = "PRIVATE-EVENT-POIGNANCY"
    adapter = FakeModernChatAdapter(response('{"output":"6"}'))
    self.assertEqual(6, self._event_poignancy(adapter, marker))
    messages = adapter.calls[0]["messages"]
    self.assertEqual((1, "user"), (len(messages), messages[0]["role"]))
    self.assertIn(marker, messages[0]["content"])
    self.assertIn('Example output json:', messages[0]["content"])
    self.assertNotIn(marker, repr(get_telemetry()))
    self.assertNotIn('{"output":"6"}', repr(get_telemetry()))

  def test_126_event_poignancy_exhaustion_returns_typed_failsafe(self):
    # R1CHAT-P4 / live B4 (Maria Lopez, step 107): 3 physical attempts, all
    # invalid, must restore the historical fail-safe (4) instead of the
    # implicit ``None`` that produced 'NoneType' object is not subscriptable
    # in reflect.py's ``run_gpt_prompt_event_poignancy(...)[0]`` consumer.
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":"not-an-int"}'),
      response("still-invalid"))
    scratch = SimpleNamespace(
      name="Maria Lopez", get_str_iss=lambda: "identity")
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_event_poignancy(
        SimpleNamespace(scratch=scratch), "event")
    self.assertIsNotNone(result)
    self.assertEqual((4, 4), (result[0], result[1][4]))
    self.assertIs(type(result[0]), int)
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))
    self.assertEqual(2, sum(
      1 for event in get_telemetry() if event.physical_attempt > 1))

  def test_126b_event_poignancy_policy_error_still_hard_fails(self):
    adapter = FakeModernChatAdapter(
      LLMAuthenticationError("denied"), response('{"output":"must-not-run"}'))
    with self.assertRaises(LLMAuthenticationError):
      self._event_poignancy(adapter)
    self.assertEqual(1, len(adapter.calls))

  def test_127_event_poignancy_invalid_then_valid_preserves_parser(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":"8"}'))
    self.assertEqual(8, self._event_poignancy(adapter))
    self.assertEqual(2, len(adapter.calls))

  def test_128_missing_and_unknown_callers_hard_fail_before_provider(self):
    adapter = FakeModernChatAdapter(response('{"output":"must-not-run"}'))
    with use_modern_chat_runtime(self.config(), adapter):
      for caller in (None, "unknown_chat_caller"):
        with self.subTest(caller=caller), (
            self.assertRaises(ModernChatCallerNotAllowedError)):
          with gpt_structure.use_modern_chat_caller(
              caller, M5_CHAT_MODEL, 0, 15, None):
            pass
    self.assertEqual([], adapter.calls)

  def test_129_event_poignancy_one_success_is_one_physical_request(self):
    adapter = FakeModernChatAdapter(response('{"output":"5"}'))
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("network reached")) as dns:
      self.assertEqual(5, self._event_poignancy(adapter))
    events = get_telemetry()
    self.assertEqual((1, 1), (len(adapter.calls), len(events)))
    self.assertEqual(("event_poignancy", CHAT, M5_CHAT_MODEL), (
      events[0].caller_id, events[0].operation, events[0].model_or_engine))
    self.assertNotIn("gpt-3.5-turbo", [event.model_or_engine
                                      for event in events])
    dns.assert_not_called()

  def test_129b_event_poignancy_consumer_survives_retry_exhaustion(self):
    # R1CHAT-P4 consumer regression: exercises the exact expression from
    # reflect.py's generate_poig_score -- run_gpt_prompt_event_poignancy(
    # persona, description)[0] -- which is the real call site that raised
    # 'NoneType' object is not subscriptable in live run
    # r1cli-a2-b-process-b4 (Maria Lopez, step 107, caller=event_poignancy).
    from persona.cognitive_modules import reflect as reflect_module
    for event_type in ("event", "thought"):
      with self.subTest(event_type=event_type):
        clear_telemetry()
        adapter = FakeModernChatAdapter(
          response("not-json"), response('{"output":"not-an-int"}'),
          response("still-invalid"))
        scratch = SimpleNamespace(
          name="Maria Lopez", get_str_iss=lambda: "identity")
        persona = SimpleNamespace(scratch=scratch)
        with self._backend_workdir(), redirect_stdout(io.StringIO()), (
            use_modern_chat_runtime(self.config(), adapter)):
          score = reflect_module.generate_poig_score(
            persona, event_type, "a chat happened")
        self.assertEqual(4, score)
        self.assertIs(type(score), int)

  def _focal_pt(self, adapter, statements="Klaus is reading\n", n=3,
               persona_name="Klaus Mueller"):
    persona = SimpleNamespace(name=persona_name)
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_focal_pt(persona, statements, n)
      return result

  def test_129c_focal_pt_authorized_attribution(self):
    # R1CHAT-P7 / live B8 (Klaus Mueller, tick 560, caller mislabeled
    # event_poignancy by the failure-report fallback): the real culprit was
    # run_gpt_prompt_focal_pt calling ChatGPT_safe_generate_response with no
    # use_modern_chat_caller wrapper at all, so the modern runtime received
    # caller_id=None and the guard fail-closed with
    # ModernChatCallerNotAllowedError. This must now be attributed as
    # "focal_pt" and authorized.
    adapter = FakeModernChatAdapter(response(
      '{"output": "[\\"What should Klaus do for lunch\\", '
      '\\"Does Klaus like tea\\", \\"Who is Klaus\\"]"}'))
    result = self._focal_pt(adapter)
    output, metadata = result
    self.assertEqual(
      ["What should Klaus do for lunch", "Does Klaus like tea",
       "Who is Klaus"], output)
    event = get_telemetry()[0]
    self.assertEqual(("focal_pt", CHAT, M5_CHAT_MODEL), (
      event.caller_id, event.operation, event.model_or_engine))
    self.assertEqual((M5_CHAT_MODEL, 0, 15, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    self.assertIn("focal_pt", M5_REPLAY_CALLER_ALLOWLIST)

  def test_129d_focal_pt_return_contract_preserved(self):
    adapter = FakeModernChatAdapter(response(
      '{"output": "[\\"a\\", \\"b\\", \\"c\\"]"}'))
    output, metadata = self._focal_pt(adapter)
    self.assertEqual(["a", "b", "c"], output)
    self.assertIs(type(output), list)
    self.assertEqual(5, len(metadata))
    self.assertEqual(output, metadata[0])

  def test_129e_focal_pt_exhaustion_returns_typed_failsafe(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-invalid"), response("bad"))
    output, metadata = self._focal_pt(adapter, n=3)
    self.assertIsNotNone(output)
    self.assertEqual(["Who am I", "Who am I", "Who am I"], output)
    self.assertEqual(output, metadata[4])
    self.assertIs(type(output), list)
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))
    self.assertEqual(2, sum(
      1 for event in get_telemetry() if event.physical_attempt > 1))

  def test_129f_focal_pt_consumer_survives_retry_exhaustion(self):
    # Exercises the exact expression from reflect.py's generate_focal_points
    # -- run_gpt_prompt_focal_pt(persona, statements, n)[0] -- the real
    # consumer reached from reflection_trigger -> run_reflect that crashed
    # live in r1cli-a2-b-process-b8. reflect.py itself is not modified.
    from persona.cognitive_modules import reflect as reflect_module
    node_a = SimpleNamespace(last_accessed=1, embedding_key="Klaus reads")
    node_b = SimpleNamespace(last_accessed=2, embedding_key="Klaus walks")
    persona = SimpleNamespace(
      a_mem=SimpleNamespace(seq_event=[node_a, node_b], seq_thought=[]),
      scratch=SimpleNamespace(importance_ele_n=2))
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-invalid"), response("bad"))
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      focal_points = reflect_module.generate_focal_points(persona, 3)
    self.assertEqual(["Who am I", "Who am I", "Who am I"], focal_points)

  def test_129g_focal_pt_policy_error_still_hard_fails(self):
    adapter = FakeModernChatAdapter(
      LLMAuthenticationError("denied"), response('{"output":"must-not-run"}'))
    with self.assertRaises(LLMAuthenticationError):
      self._focal_pt(adapter)
    self.assertEqual(1, len(adapter.calls))

  def _summarize_relationship(self, adapter, statements="shared context"):
    persona = SimpleNamespace(scratch=SimpleNamespace(name="Maria Lopez"))
    target = SimpleNamespace(scratch=SimpleNamespace(name="Klaus Mueller"))
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_agent_chat_summarize_relationship(
        persona, target, statements)
      return result[0] if result is not None else None

  def test_130_relationship_summary_preserves_legacy_transport_contract(self):
    adapter = FakeModernChatAdapter(
      response('{"output":"Klaus is discussing a shared project"}'))
    context = LLMReplayContext(
      cognitive_category="CONVERSATION", actor_id="Maria Lopez",
      simulation_id="offline-relationship", simulation_step=2)
    with use_llm_replay_context(context), patch.object(
        gpt_structure, "chat_completion",
        side_effect=AssertionError("legacy Chat reached")) as legacy:
      output = self._summarize_relationship(adapter)
    self.assertEqual("Klaus is discussing a shared project", output)
    self.assertEqual((M5_CHAT_MODEL, 0, None, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    event = get_telemetry()[0]
    self.assertEqual(
      ("agent_chat_summarize_relationship", CHAT, M5_CHAT_MODEL,
       "Maria Lopez", "offline-relationship", 2),
      (event.caller_id, event.operation, event.model_or_engine,
       event.actor_id, event.simulation_id, event.simulation_step))
    legacy.assert_not_called()

  def test_131_relationship_summary_wrong_key_keeps_three_attempts(self):
    adapter = FakeModernChatAdapter(*[
      response('{"wrong":"relationship"}') for _ in range(3)])
    self.assertIsNone(self._summarize_relationship(adapter))
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_132_relationship_summary_invalid_json_keeps_three_attempts(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-invalid"), response("bad"))
    self.assertIsNone(self._summarize_relationship(adapter))
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(
      ["agent_chat_summarize_relationship"] * 3,
      [event.caller_id for event in get_telemetry()])

  def test_133_relationship_summary_invalid_then_valid_uses_existing_parser(self):
    adapter = FakeModernChatAdapter(
      response("not-json"),
      response('{"output":"Klaus is discussing a shared project"}'))
    self.assertEqual(
      "Klaus is discussing a shared project",
      self._summarize_relationship(adapter))
    self.assertEqual(2, len(adapter.calls))

  def _summarize_conversation(self, adapter):
    persona = SimpleNamespace(scratch=SimpleNamespace(name="Maria Lopez"))
    conversation = [
      ["Maria Lopez", "synthetic first turn"],
      ["Klaus Mueller", "synthetic second turn"],
    ]
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_prompt_summarize_conversation(
        persona, conversation)
      return result[0] if result is not None else None

  def test_134_conversation_summary_preserves_legacy_transport_contract(self):
    adapter = FakeModernChatAdapter(response('{"output":"a shared topic"}'))
    context = LLMReplayContext(
      cognitive_category="CONVERSATION", actor_id="Maria Lopez",
      simulation_id="offline-conversation-summary", simulation_step=2)
    with use_llm_replay_context(context), patch.object(
        gpt_structure, "chat_completion",
        side_effect=AssertionError("legacy Chat reached")) as legacy:
      output = self._summarize_conversation(adapter)
    self.assertEqual("conversing about a shared topic", output)
    self.assertEqual((M5_CHAT_MODEL, 0, None, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    event = get_telemetry()[0]
    self.assertEqual(
      ("summarize_conversation", CHAT, M5_CHAT_MODEL,
       "Maria Lopez", "offline-conversation-summary", 2),
      (event.caller_id, event.operation, event.model_or_engine,
       event.actor_id, event.simulation_id, event.simulation_step))
    legacy.assert_not_called()

  def test_135_conversation_summary_invalid_then_valid_uses_existing_parser(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":"a shared topic"}'))
    self.assertEqual(
      "conversing about a shared topic",
      self._summarize_conversation(adapter))
    self.assertEqual(2, len(adapter.calls))

  def test_136_conversation_summary_three_invalid_attempts_fail_clearly(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"wrong":"summary"}'),
      response("still-invalid"))
    self.assertIsNone(self._summarize_conversation(adapter))
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_137_conversation_summary_retries_truncated_modern_response(self):
    adapter = FakeModernChatAdapter(
      response('{"output":"truncated', finish_reason="length"),
      response('{"output":"a shared topic"}'))
    self.assertEqual(
      "conversing about a shared topic",
      self._summarize_conversation(adapter))
    self.assertEqual(2, len(adapter.calls))

  def _iterative_utterance(self, adapter):
    scratch = SimpleNamespace(
      name="Maria Lopez", curr_time=__import__("datetime").datetime(
        2023, 2, 13, 10, 0, 0), curr_tile=(117, 49),
      get_str_iss=lambda: "synthetic identity")
    persona = SimpleNamespace(
      scratch=scratch, a_mem=SimpleNamespace(seq_chat=[]))
    target = SimpleNamespace(
      scratch=SimpleNamespace(name="Klaus Mueller"))
    maze = SimpleNamespace(access_tile=lambda tile: {
      "sector": "Dorm for Oak Hill College", "arena": "common room"})
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      result = run_gpt_prompt.run_gpt_generate_iterative_chat_utt(
        maze, persona, target, {}, "synthetic context", [])
      return result[0]

  def test_138_iterative_utterance_valid_continue_preserves_contract(self):
    adapter = FakeModernChatAdapter(response(
      '{"Maria Lopez":"hello","Did the conversation end?":false}'))
    with patch.object(
        gpt_structure, "chat_completion",
        side_effect=AssertionError("legacy Chat reached")) as legacy:
      output = self._iterative_utterance(adapter)
    self.assertEqual({"utterance": "hello", "end": False}, output)
    self.assertEqual((M5_CHAT_MODEL, 0, None, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    event = get_telemetry()[0]
    self.assertEqual(("iterative_chat_utterance", CHAT),
                     (event.caller_id, event.operation))
    legacy.assert_not_called()

  def test_139_iterative_utterance_valid_end_preserves_boolean(self):
    adapter = FakeModernChatAdapter(response(
      '{"Maria Lopez":"goodbye","Did the conversation end?":true}'))
    self.assertEqual(
      {"utterance": "goodbye", "end": True},
      self._iterative_utterance(adapter))

  def test_140_iterative_utterance_invalid_then_valid_retries(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response(
        '{"Maria Lopez":"hello","Did the conversation end?":false}'))
    self.assertEqual(False, self._iterative_utterance(adapter)["end"])
    self.assertEqual(2, len(adapter.calls))

  def test_141_iterative_utterance_three_invalid_uses_continue_failsafe(self):
    adapter = FakeModernChatAdapter(
      response("bad"), response("still-bad"), response("invalid"))
    self.assertEqual(
      {"utterance": "...", "end": False},
      self._iterative_utterance(adapter))
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_142_iterative_utterance_retries_truncated_response(self):
    adapter = FakeModernChatAdapter(
      response('{"Maria Lopez":"truncated', finish_reason="length"),
      response('{"Maria Lopez":"done","Did the conversation end?":true}'))
    self.assertTrue(self._iterative_utterance(adapter)["end"])
    self.assertEqual(2, len(adapter.calls))

  def _chat_poignancy_result(self, adapter, event="synthetic chat",
                             persona_name="Klaus Mueller"):
    scratch = SimpleNamespace(
      name=persona_name, get_str_iss=lambda: "synthetic identity")
    with self._backend_workdir(), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(self.config(), adapter)):
      return run_gpt_prompt.run_gpt_prompt_chat_poignancy(
        SimpleNamespace(scratch=scratch), event)

  def test_143_chat_poignancy_historical_string_output_is_typed(self):
    result = self._chat_poignancy_result(
      FakeModernChatAdapter(response('{"output":"7"}')))
    self.assertEqual(7, result[0])
    self.assertIs(type(result[0]), int)

  def test_144_chat_poignancy_modern_numeric_json_is_typed(self):
    result = self._chat_poignancy_result(
      FakeModernChatAdapter(response('{"output":7}')))
    self.assertEqual(7, result[0])
    self.assertIs(type(result[0]), int)

  def test_145_chat_poignancy_invalid_then_valid_retries_parser(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":8}'))
    result = self._chat_poignancy_result(adapter)
    self.assertEqual(8, result[0])
    self.assertEqual(2, len(adapter.calls))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_146_chat_poignancy_exhaustion_returns_typed_failsafe(self):
    adapter = FakeModernChatAdapter(
      response("not-json"), response('{"output":"not-an-int"}'),
      response("still-invalid"))
    result = self._chat_poignancy_result(adapter)
    self.assertIsNotNone(result)
    self.assertEqual((4, 4), (result[0], result[1][4]))
    self.assertIs(type(result[0]), int)
    self.assertEqual(3, len(adapter.calls))

  def test_147_chat_poignancy_preserves_actual_transport_and_attribution(self):
    adapter = FakeModernChatAdapter(response('{"output":6}'))
    context = LLMReplayContext(
      cognitive_category="PERCEPTION", actor_id="Klaus Mueller",
      simulation_id="offline-chat-poignancy", simulation_step=1)
    with use_llm_replay_context(context), patch.object(
        gpt_structure, "chat_completion",
        side_effect=AssertionError("legacy Chat reached")) as legacy:
      result = self._chat_poignancy_result(adapter)
    self.assertEqual(6, result[0])
    self.assertEqual((M5_CHAT_MODEL, 0, None, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))
    self.assertEqual(15, result[1][2]["max_tokens"])
    event = get_telemetry()[0]
    self.assertEqual(
      ("chat_poignancy", CHAT, M5_CHAT_MODEL, "Klaus Mueller",
       "offline-chat-poignancy", 1),
      (event.caller_id, event.operation, event.model_or_engine,
       event.actor_id, event.simulation_id, event.simulation_step))
    legacy.assert_not_called()

  def test_148_chat_poignancy_policy_error_still_hard_fails(self):
    adapter = FakeModernChatAdapter(
      LLMAuthenticationError("denied"), response('{"output":6}'))
    with self.assertRaises(LLMAuthenticationError):
      self._chat_poignancy_result(adapter)
    self.assertEqual(1, len(adapter.calls))

  def _memo_on_convo_result(self, adapter,
                            all_utt="Maria Lopez: Hi\nKlaus Mueller: Hi\n",
                            persona_name="Maria Lopez", config=None):
    scratch = SimpleNamespace(name=persona_name)
    persona = SimpleNamespace(name=persona_name, scratch=scratch)
    with patch.object(run_gpt_prompt, "debug", False), (
        self._backend_workdir()), redirect_stdout(io.StringIO()), (
        use_modern_chat_runtime(config or self.config(), adapter)):
      return run_gpt_prompt.run_gpt_prompt_memo_on_convo(persona, all_utt)

  def test_149_memo_on_convo_uses_attributed_modern_chat(self):
    adapter = FakeModernChatAdapter(
      response('{"output": "An interesting evening."}'))
    context = LLMReplayContext(
      cognitive_category="WORLD_TICK", actor_id="Maria Lopez",
      simulation_id="r1chat-p3-offline", simulation_step=107)
    with use_llm_replay_context(context):
      output, unused_extras = self._memo_on_convo_result(adapter)
    self.assertEqual("An interesting evening.", output)
    event = get_telemetry()[0]
    self.assertEqual(
      ("memo_on_convo", CHAT, M5_CHAT_MODEL, "Maria Lopez",
       "r1chat-p3-offline", 107),
      (event.caller_id, event.operation, event.model_or_engine,
       event.actor_id, event.simulation_id, event.simulation_step))

  def test_150_memo_on_convo_actor_context_is_not_hardcoded(self):
    adapter = FakeModernChatAdapter(response('{"output": "Klaus seemed busy."}'))
    context = LLMReplayContext(actor_id="Klaus Mueller", simulation_step=42)
    with use_llm_replay_context(context):
      self._memo_on_convo_result(adapter, persona_name="Klaus Mueller")
    event = get_telemetry()[0]
    self.assertEqual("memo_on_convo", event.caller_id)
    self.assertEqual("Klaus Mueller", event.actor_id)
    self.assertEqual(42, event.simulation_step)

  def test_151_memo_on_convo_anonymous_chat_caller_fails_closed(self):
    with self.assertRaises(ModernChatCallerNotAllowedError):
      validate_modern_chat_caller(None)

  def test_152_memo_on_convo_unknown_chat_caller_fails_closed(self):
    with self.assertRaises(ModernChatCallerNotAllowedError):
      validate_modern_chat_caller("unknown")

  def test_153_memo_on_convo_authorized_caller_passes_policy(self):
    self.assertEqual(
      "memo_on_convo", validate_modern_chat_caller("memo_on_convo"))
    self.assertIn("memo_on_convo", M5_REPLAY_CALLER_ALLOWLIST)

  def test_154_memo_on_convo_prompt_and_parameters_are_unchanged(self):
    adapter = FakeModernChatAdapter(
      response('{"output": "Noted the market stall."}'))
    output, extras = self._memo_on_convo_result(adapter)
    returned_output, prompt, gpt_param, prompt_input, fail_safe = extras
    self.assertEqual(output, returned_output)
    self.assertEqual({
      "engine": "text-davinci-002", "max_tokens": 15, "temperature": 0,
      "top_p": 1, "stream": False, "frequency_penalty": 0,
      "presence_penalty": 0, "stop": None}, gpt_param)
    self.assertEqual("...", fail_safe)
    self.assertEqual(
      ["Maria Lopez: Hi\nKlaus Mueller: Hi\n", "Maria Lopez",
       "Maria Lopez", "Maria Lopez"], prompt_input)
    self.assertIsInstance(prompt, str)
    self.assertEqual((M5_CHAT_MODEL, 0, 15, None), (
      adapter.calls[0]["model"], adapter.calls[0]["temperature"],
      adapter.calls[0]["max_tokens"], adapter.calls[0]["stop"]))

  def test_155_memo_on_convo_chatgpt_exhaustion_falls_through_unchanged(self):
    """Attribution must not alter the historical fallthrough: exhausting the
    ChatGPT branch (repeat=3, unchanged) still hands off to the existing,
    already-attributed legacy CompletionCompat branch below with the exact
    same prompt/gpt_param/fail_safe -- this wave touches only the ChatGPT
    branch that used to reach the modern runtime anonymously."""
    adapter = FakeModernChatAdapter(
      response("not-json"), response("still-not-json"), response("nope"))
    captured = {}

    def fake_legacy_fallback(prompt, gpt_param, repeat, fail_safe,
                             func_validate, func_clean_up, caller_id=None):
      captured.update(
        prompt=prompt, gpt_param=dict(gpt_param), fail_safe=fail_safe,
        caller_id=caller_id)
      return fail_safe

    with patch.object(run_gpt_prompt, "safe_generate_response",
                      side_effect=fake_legacy_fallback):
      output, extras = self._memo_on_convo_result(adapter)

    self.assertEqual(3, len(adapter.calls))
    self.assertEqual("memo_on_convo", captured["caller_id"])
    self.assertEqual("...", captured["fail_safe"])
    self.assertEqual("...", output)
    self.assertEqual({
      "engine": "text-davinci-003", "max_tokens": 50, "temperature": 0,
      "top_p": 1, "stream": False, "frequency_penalty": 0,
      "presence_penalty": 0, "stop": None}, captured["gpt_param"])

  def test_156_memo_on_convo_return_contract_is_a_two_item_tuple(self):
    adapter = FakeModernChatAdapter(response('{"output": "Kept it brief."}'))
    result = self._memo_on_convo_result(adapter)
    self.assertIsInstance(result, tuple)
    self.assertEqual(2, len(result))
    output, extras = result
    self.assertIsInstance(output, str)
    self.assertIsInstance(extras, list)
    self.assertEqual(5, len(extras))

  def test_157_both_r1emb_p1_embedding_callers_remain_authorized(self):
    """R1EMB-P1 regression: planning_thought_on_convo and memo_on_convo are
    still the only two authorized EMBEDDING callers after this wave's CHAT
    allowlist change (a distinct allowlist -- adding memo_on_convo to CHAT
    must not touch or widen the EMBEDDING policy)."""
    self.assertEqual(
      ("planning_thought_on_convo", "memo_on_convo"),
      M2_EMBEDDING_CALLER_ALLOWLIST)
    self.assertEqual(
      "planning_thought_on_convo",
      validate_modern_embedding_caller("planning_thought_on_convo"))
    self.assertEqual(
      "memo_on_convo", validate_modern_embedding_caller("memo_on_convo"))
    with self.assertRaises(ModernChatCallerNotAllowedError):
      validate_modern_embedding_caller(None)


if __name__ == "__main__":
  unittest.main()
