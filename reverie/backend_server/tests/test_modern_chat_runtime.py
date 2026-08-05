from contextlib import redirect_stdout
from decimal import Decimal
import io
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template.chat_runtime import (
  M5_CHAT_MODEL,
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
)
from persona.prompt_template.cost_ledger import (
  COMPLETE,
  ModelPricing,
  PricingSnapshot,
  build_cost_ledger_records,
  summarize_cost_ledger,
)
from persona.prompt_template.embedding_runtime import (
  TEXT_EMBEDDING_3_SMALL_MODEL,
  build_modern_embedding_runtime_config,
  use_embedding_runtime,
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
  get_provider,
  get_telemetry,
  reset_embedding_cache,
  reset_provider,
  text_completion,
  use_provider,
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


if __name__ == "__main__":
  unittest.main()
