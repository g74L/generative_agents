from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.memory_structures.embedding_space import (
  LEGACY_ADA_002_MANIFEST,
  EmbeddingSpaceMismatchError,
  reset_runtime_embedding_manifest,
  use_runtime_embedding_manifest,
)
from persona.prompt_template import gpt_structure
from persona.prompt_template.llm_provider import (
  CHAT,
  ERROR,
  SUCCESS,
  FakeProvider,
  OpenAILegacyProvider,
  chat_completion,
  clear_telemetry,
  create_llm_provider,
  embedding,
  get_embedding_cache_stats,
  get_provider,
  get_telemetry,
  reset_embedding_cache,
  reset_provider,
  use_configured_provider,
  use_provider,
)
from persona.prompt_template.llm_provider_config import (
  DEFAULT_LLM_PROVIDER_CONFIG,
  FAKE,
  FAKE_SDK_MODE,
  FAKE_TRANSPORT,
  LEGACY_OPENAI,
  LEGACY_SDK_MODE,
  LEGACY_TRANSPORT,
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
  LLMProviderConfig,
  fake_provider_config,
  get_llm_provider_config,
  modern_openai_config,
  reset_llm_provider_config,
  use_llm_provider_config,
)
from persona.prompt_template.modern_openai_provider import (
  LLMAuthenticationError,
  LLMAuthorizationError,
  LLMConnectionError,
  LLMEmptyOutputError,
  LLMIncompleteResponseError,
  LLMInvalidRequestError,
  LLMMalformedResponseError,
  LLMModelNotFoundError,
  LLMProviderError,
  LLMRateLimitError,
  LLMRefusalError,
  LLMServerError,
  LLMTimeoutError,
  LLMUnsupportedParameterError,
  ModernCompletionNotEnabledError,
  ModernOpenAIClientAdapter,
  ModernOpenAIProvider,
  ModernOpenAISdkUnavailableError,
  map_modern_sdk_error,
  normalize_chat_response,
  normalize_embedding_response,
)


class _Endpoint:
  def __init__(self, *results):
    self.results = list(results)
    self.calls = []

  def create(self, **kwargs):
    self.calls.append(dict(kwargs))
    if not self.results:
      raise AssertionError("No fake SDK response configured")
    result = self.results.pop(0)
    if isinstance(result, Exception):
      raise result
    return result


class _Client:
  def __init__(self, chat_results=(), embedding_results=()):
    self.chat_endpoint = _Endpoint(*chat_results)
    self.embedding_endpoint = _Endpoint(*embedding_results)
    self.chat = SimpleNamespace(
      completions=self.chat_endpoint)
    self.embeddings = self.embedding_endpoint


def _chat(content="ok", **changes):
  response = {
    "choices": [{
      "message": {"content": content, "refusal": None},
      "finish_reason": "stop",
    }],
    "model": "gpt-3.5-turbo",
    "_request_id": "req_fixture",
    "status": "completed",
    "usage": {
      "prompt_tokens": 11,
      "completion_tokens": 7,
      "prompt_tokens_details": {"cached_tokens": 3},
      "completion_tokens_details": {"reasoning_tokens": 2},
    },
  }
  response.update(changes)
  return response


def _embedding(vector=None, **changes):
  response = {
    "data": [{"embedding": vector or [0.25, -0.5]}],
    "model": "text-embedding-ada-002",
    "_request_id": "req_embedding",
    "usage": {"prompt_tokens": 4},
  }
  response.update(changes)
  return response


def _sdk_error(name, status=None, code=None):
  error_type = type(name, (Exception,), {})
  error = error_type("sensitive upstream detail")
  error.status_code = status
  error.code = code
  return error


class ModernSDKCompatibilitySeamTests(unittest.TestCase):
  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    reset_runtime_embedding_manifest()
    reset_embedding_cache()
    clear_telemetry()
    self.network_patches = [
      patch.object(gpt_structure.openai.ChatCompletion, "create",
                   side_effect=AssertionError("real chat transport reached")),
      patch.object(gpt_structure.openai.Completion, "create",
                   side_effect=AssertionError(
                     "real completion transport reached")),
      patch.object(gpt_structure.openai.Embedding, "create",
                   side_effect=AssertionError(
                     "real embedding transport reached")),
    ]
    for network_patch in self.network_patches:
      network_patch.start()

  def tearDown(self):
    for network_patch in reversed(self.network_patches):
      network_patch.stop()
    reset_provider()
    reset_llm_provider_config()
    reset_runtime_embedding_manifest()
    reset_embedding_cache()
    clear_telemetry()

  def adapter(self, client):
    return ModernOpenAIClientAdapter(modern_openai_config(), client=client)

  def provider(self, client):
    return ModernOpenAIProvider(
      modern_openai_config(), self.adapter(client))

  def test_01_legacy_provider_remains_runtime_default(self):
    self.assertIsInstance(get_provider(), OpenAILegacyProvider)

  def test_02_default_config_is_legacy(self):
    self.assertEqual(LEGACY_OPENAI, DEFAULT_LLM_PROVIDER_CONFIG.provider_kind)
    self.assertEqual(LEGACY_TRANSPORT,
                     DEFAULT_LLM_PROVIDER_CONFIG.transport_kind)
    self.assertEqual(LEGACY_SDK_MODE, DEFAULT_LLM_PROVIDER_CONFIG.sdk_mode)

  def test_03_factory_selects_legacy(self):
    self.assertIsInstance(create_llm_provider(), OpenAILegacyProvider)

  def test_04_factory_selects_fake(self):
    self.assertIsInstance(create_llm_provider(fake_provider_config()),
                          FakeProvider)

  def test_05_factory_selects_injected_modern_adapter(self):
    adapter = self.adapter(_Client())
    provider = create_llm_provider(modern_openai_config(), adapter)
    self.assertIsInstance(provider, ModernOpenAIProvider)

  def test_06_modern_sdk_absence_is_explicit(self):
    with patch(
        "persona.prompt_template.modern_openai_provider.importlib.import_module",
        side_effect=ImportError):
      with self.assertRaises(ModernOpenAISdkUnavailableError):
        create_llm_provider(modern_openai_config())

  def test_07_config_context_restores_previous_value(self):
    before = get_llm_provider_config()
    with use_llm_provider_config(modern_openai_config()):
      self.assertEqual(MODERN_OPENAI,
                       get_llm_provider_config().provider_kind)
    self.assertIs(before, get_llm_provider_config())

  def test_08_nested_config_contexts_restore_in_order(self):
    with use_llm_provider_config(modern_openai_config()):
      with use_llm_provider_config(fake_provider_config()):
        self.assertEqual(FAKE, get_llm_provider_config().provider_kind)
      self.assertEqual(MODERN_OPENAI,
                       get_llm_provider_config().provider_kind)

  def test_09_config_rejects_unknown_provider(self):
    with self.assertRaises(ValueError):
      LLMProviderConfig(provider_kind="UNKNOWN")

  def test_10_config_rejects_negative_sdk_retries(self):
    with self.assertRaises(ValueError):
      LLMProviderConfig(sdk_retry_count=-1)

  def test_11_config_rejects_nonpositive_timeout(self):
    with self.assertRaises(ValueError):
      LLMProviderConfig(request_timeout_seconds=0)

  def test_12_modern_chat_preserves_legacy_shape(self):
    with use_provider(self.provider(_Client(chat_results=[_chat("hello")]))):
      result = chat_completion(model="gpt-3.5-turbo", messages=[])
    self.assertEqual("hello", result["choices"][0]["message"]["content"])

  def test_13_modern_embedding_preserves_legacy_shape(self):
    client = _Client(embedding_results=[_embedding([0.1, 0.2])])
    with use_provider(self.provider(client)):
      result = embedding(input=["text"], model="text-embedding-ada-002")
    self.assertEqual([0.1, 0.2], result["data"][0]["embedding"])

  def test_14_modern_completion_is_explicitly_unsupported(self):
    provider = self.provider(_Client())
    with self.assertRaises(ModernCompletionNotEnabledError):
      provider.text_completion(model="legacy")

  def test_15_chat_request_forces_store_false(self):
    client = _Client(chat_results=[_chat()])
    self.adapter(client).create_chat(model="m", messages=[])
    self.assertIs(False, client.chat_endpoint.calls[0]["store"])

  def test_16_chat_request_forwards_only_expected_fields(self):
    client = _Client(chat_results=[_chat()])
    self.adapter(client).create_chat(
      model="model", messages=[{"role": "user", "content": "prompt"}])
    self.assertEqual({"model", "messages", "store"},
                     set(client.chat_endpoint.calls[0]))

  def test_17_embedding_request_forwards_expected_fields(self):
    client = _Client(embedding_results=[_embedding()])
    self.adapter(client).create_embedding(model="model", input=["text"])
    self.assertEqual({"model": "model", "input": ["text"]},
                     client.embedding_endpoint.calls[0])

  def test_18_sdk_factory_receives_zero_retries_and_timeout(self):
    captured = {}
    def factory(config):
      captured.update(retries=config.sdk_retry_count,
                      timeout=config.request_timeout_seconds)
      return _Client()
    config = modern_openai_config(request_timeout_seconds=12.5)
    ModernOpenAIClientAdapter(config=config, client_factory=factory)
    self.assertEqual({"retries": 0, "timeout": 12.5}, captured)

  def test_19_authentication_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("AuthenticationError", 401)), LLMAuthenticationError)

  def test_20_authorization_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("PermissionDeniedError", 403)), LLMAuthorizationError)

  def test_21_model_not_found_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("NotFoundError", 404)), LLMModelNotFoundError)

  def test_22_invalid_request_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("BadRequestError", 400)), LLMInvalidRequestError)

  def test_23_unsupported_parameter_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("BadRequestError", 400, "unsupported_parameter")),
      LLMUnsupportedParameterError)

  def test_24_timeout_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("APITimeoutError")), LLMTimeoutError)

  def test_25_connection_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("APIConnectionError")), LLMConnectionError)

  def test_26_rate_limit_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("RateLimitError", 429)), LLMRateLimitError)

  def test_27_server_error_mapping(self):
    self.assertIsInstance(map_modern_sdk_error(
      _sdk_error("InternalServerError", 500)), LLMServerError)

  def test_28_generic_error_mapping_redacts_upstream_message(self):
    mapped = map_modern_sdk_error(_sdk_error("OpenAIError"))
    self.assertIsInstance(mapped, LLMProviderError)
    self.assertNotIn("sensitive", str(mapped))

  def test_29_adapter_maps_sdk_errors(self):
    client = _Client(chat_results=[_sdk_error("RateLimitError", 429)])
    with self.assertRaises(LLMRateLimitError):
      self.adapter(client).create_chat(model="m", messages=[])

  def test_30_refusal_is_typed(self):
    response = _chat()
    response["choices"][0]["message"]["refusal"] = "not returned"
    with self.assertRaises(LLMRefusalError):
      normalize_chat_response(response)

  def test_31_incomplete_status_is_typed(self):
    with self.assertRaises(LLMIncompleteResponseError):
      normalize_chat_response(_chat(status="incomplete"))

  def test_32_incomplete_finish_reason_is_typed(self):
    response = _chat()
    response["choices"][0]["finish_reason"] = "length"
    with self.assertRaises(LLMIncompleteResponseError):
      normalize_chat_response(response)

  def test_33_none_output_is_typed(self):
    with self.assertRaises(LLMEmptyOutputError):
      normalize_chat_response(_chat(None))

  def test_34_blank_output_is_typed(self):
    with self.assertRaises(LLMEmptyOutputError):
      normalize_chat_response(_chat("   "))

  def test_35_missing_choices_is_malformed(self):
    with self.assertRaises(LLMMalformedResponseError):
      normalize_chat_response({"choices": []})

  def test_36_missing_message_is_malformed(self):
    with self.assertRaises(LLMMalformedResponseError):
      normalize_chat_response({"choices": [{}]})

  def test_37_malformed_embedding_is_typed(self):
    with self.assertRaises(LLMMalformedResponseError):
      normalize_embedding_response(_embedding([1.0, float("nan")]))

  def test_38_chat_telemetry_contains_normalized_metadata(self):
    with use_provider(self.provider(_Client(chat_results=[_chat()]))):
      chat_completion(model="gpt-3.5-turbo", messages=[])
    event = get_telemetry()[0]
    self.assertEqual((MODERN_OPENAI, MODERN_TRANSPORT, "req_fixture"),
                     (event.provider_kind, event.transport_kind,
                      event.request_id))
    self.assertEqual((11, 7, 3, 2),
                     (event.input_tokens, event.output_tokens,
                      event.cached_input_tokens, event.reasoning_tokens))
    self.assertEqual(("gpt-3.5-turbo", "stop", "completed"),
                     (event.response_model, event.finish_reason,
                      event.response_status))

  def test_39_embedding_telemetry_contains_request_id_and_usage(self):
    client = _Client(embedding_results=[_embedding()])
    with use_provider(self.provider(client)):
      embedding(input=["text"], model="text-embedding-ada-002")
    event = get_telemetry()[0]
    self.assertEqual("req_embedding", event.request_id)
    self.assertEqual(4, event.input_tokens)

  def test_40_telemetry_contains_no_raw_prompt_or_output(self):
    secret_prompt = "TOP-SECRET-PROMPT"
    secret_output = "TOP-SECRET-OUTPUT"
    with use_provider(self.provider(
        _Client(chat_results=[_chat(secret_output)]))):
      chat_completion(model="gpt-3.5-turbo", messages=[{
        "role": "user", "content": secret_prompt}])
    serialized = repr(get_telemetry())
    self.assertNotIn(secret_prompt, serialized)
    self.assertNotIn(secret_output, serialized)

  def test_41_application_retry_count_is_not_multiplied(self):
    client = _Client(chat_results=[
      _sdk_error("RateLimitError", 429),
      _sdk_error("APITimeoutError"),
      _chat('{"output": "ok"}'),
    ])
    with use_provider(self.provider(client)), patch.object(
        gpt_structure, "temp_sleep"), redirect_stdout(io.StringIO()):
      result = gpt_structure.ChatGPT_safe_generate_response(
        "prompt", "example", "instruction", repeat=3,
        func_validate=lambda value, prompt: value == "ok",
        func_clean_up=lambda value, prompt: value)
    self.assertEqual("ok", result)
    self.assertEqual(3, len(client.chat_endpoint.calls))
    events = get_telemetry()
    self.assertEqual([ERROR, ERROR, SUCCESS], [event.outcome for event in events])
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual([1, 2, 3], [event.physical_attempt for event in events])

  def test_42_manifest_mismatch_prevents_modern_sdk_attempt(self):
    client = _Client(embedding_results=[_embedding()])
    incompatible = replace(LEGACY_ADA_002_MANIFEST, model="other-space")
    before = get_embedding_cache_stats()
    with use_provider(self.provider(client)), use_runtime_embedding_manifest(
        incompatible):
      with self.assertRaises(EmbeddingSpaceMismatchError):
        embedding(input=["text"], model="text-embedding-ada-002")
    self.assertEqual([], client.embedding_endpoint.calls)
    self.assertEqual(before, get_embedding_cache_stats())

  def test_43_scoped_configured_provider_restores_both_states(self):
    old_provider = get_provider()
    old_config = get_llm_provider_config()
    adapter = self.adapter(_Client())
    with use_configured_provider(modern_openai_config(), adapter) as provider:
      self.assertIs(provider, get_provider())
      self.assertEqual(MODERN_OPENAI,
                       get_llm_provider_config().provider_kind)
    self.assertIs(old_provider, get_provider())
    self.assertIs(old_config, get_llm_provider_config())

  def test_44_importing_seam_does_not_require_modern_openai_class(self):
    import openai
    self.assertFalse(hasattr(openai, "OpenAI"))
    self.assertIsInstance(get_provider(), OpenAILegacyProvider)

  def test_45_lazy_sdk_client_is_built_with_zero_retries(self):
    captured = {}
    def client_class(**kwargs):
      captured.update(kwargs)
      return _Client()
    fake_module = SimpleNamespace(OpenAI=client_class)
    config = modern_openai_config(request_timeout_seconds=17.0)
    with patch(
        "persona.prompt_template.modern_openai_provider.importlib.import_module",
        return_value=fake_module):
      ModernOpenAIClientAdapter(config=config)
    self.assertEqual({"max_retries": 0, "timeout": 17.0}, captured)

  def test_46_canonical_provider_configurations_are_valid(self):
    configurations = (
      LLMProviderConfig(),
      LLMProviderConfig(
        provider_kind=LEGACY_OPENAI,
        transport_kind=LEGACY_TRANSPORT,
        sdk_mode=LEGACY_SDK_MODE),
      modern_openai_config(),
      fake_provider_config(),
    )
    self.assertEqual(
      (
        (LEGACY_OPENAI, LEGACY_TRANSPORT, LEGACY_SDK_MODE),
        (LEGACY_OPENAI, LEGACY_TRANSPORT, LEGACY_SDK_MODE),
        (MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE),
        (FAKE, FAKE_TRANSPORT, FAKE_SDK_MODE),
      ),
      tuple((config.provider_kind, config.transport_kind, config.sdk_mode)
            for config in configurations))

  def test_47_contradictory_provider_configurations_are_rejected(self):
    invalid = (
      (MODERN_OPENAI, LEGACY_TRANSPORT, MODERN_SDK_MODE),
      (MODERN_OPENAI, MODERN_TRANSPORT, LEGACY_SDK_MODE),
      (LEGACY_OPENAI, MODERN_TRANSPORT, LEGACY_SDK_MODE),
      (LEGACY_OPENAI, LEGACY_TRANSPORT, MODERN_SDK_MODE),
      (FAKE, LEGACY_TRANSPORT, FAKE_SDK_MODE),
      (FAKE, FAKE_TRANSPORT, LEGACY_SDK_MODE),
      (FAKE, FAKE_TRANSPORT, MODERN_SDK_MODE),
    )
    for provider_kind, transport_kind, sdk_mode in invalid:
      with self.subTest(provider_kind=provider_kind,
                        transport_kind=transport_kind, sdk_mode=sdk_mode):
        with self.assertRaises(ValueError):
          LLMProviderConfig(
            provider_kind=provider_kind,
            transport_kind=transport_kind,
            sdk_mode=sdk_mode)

  def test_48_unknown_transport_and_sdk_mode_are_rejected(self):
    with self.assertRaises(ValueError):
      LLMProviderConfig(transport_kind="UNKNOWN_TRANSPORT")
    with self.assertRaises(ValueError):
      LLMProviderConfig(sdk_mode="UNKNOWN_SDK_MODE")

  def test_49_invalid_config_never_reaches_provider_factory(self):
    with patch(
        "persona.prompt_template.llm_provider.create_llm_provider") as factory:
      with self.assertRaises(ValueError):
        config = LLMProviderConfig(
          provider_kind=MODERN_OPENAI,
          transport_kind=LEGACY_TRANSPORT,
          sdk_mode=MODERN_SDK_MODE)
        factory(config)
    factory.assert_not_called()

  def test_50_absent_and_none_usage_are_optional(self):
    absent = _chat()
    del absent["usage"]
    for response in (absent, _chat(usage=None)):
      with self.subTest(response=response):
        self.assertEqual(
          (None, None, None, None),
          tuple(vars(normalize_chat_response(response).usage).values()))

  def test_51_mapping_usage_aliases_and_zero_are_normalized(self):
    response = _chat(usage={
      "input_tokens": 0,
      "output_tokens": 5,
      "total_tokens": 5,
      "prompt_tokens_details": {"cached_tokens": 0},
      "completion_tokens_details": {"reasoning_tokens": 2},
    })
    usage = normalize_chat_response(response).usage
    self.assertEqual((0, 5, 0, 2), (
      usage.input_tokens, usage.output_tokens,
      usage.cached_input_tokens, usage.reasoning_tokens))

  def test_52_object_usage_is_normalized(self):
    usage_object = SimpleNamespace(
      prompt_tokens=4,
      completion_tokens=3,
      total_tokens=7,
      prompt_tokens_details=SimpleNamespace(cached_tokens=1),
      completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    usage = normalize_chat_response(_chat(usage=usage_object)).usage
    self.assertEqual((4, 3, 1, 0), (
      usage.input_tokens, usage.output_tokens,
      usage.cached_input_tokens, usage.reasoning_tokens))

  def test_53_malformed_usage_containers_are_rejected(self):
    for malformed in ("bad", [], True):
      with self.subTest(malformed=malformed):
        with self.assertRaises(LLMMalformedResponseError):
          normalize_chat_response(_chat(usage=malformed))

  def test_54_malformed_top_level_token_counts_are_rejected(self):
    malformed_fields = (
      ("prompt_tokens", "11"),
      ("prompt_tokens", True),
      ("prompt_tokens", -1),
      ("completion_tokens", 11.0),
      ("input_tokens", {}),
      ("output_tokens", []),
      ("total_tokens", False),
    )
    for field, malformed in malformed_fields:
      with self.subTest(field=field, malformed=malformed):
        with self.assertRaises(LLMMalformedResponseError):
          normalize_chat_response(_chat(usage={field: malformed}))

  def test_55_malformed_usage_detail_containers_are_rejected(self):
    malformed_fields = (
      ("prompt_tokens_details", "bad"),
      ("prompt_tokens_details", []),
      ("completion_tokens_details", "bad"),
    )
    for field, malformed in malformed_fields:
      with self.subTest(field=field, malformed=malformed):
        with self.assertRaises(LLMMalformedResponseError):
          normalize_chat_response(_chat(usage={field: malformed}))

  def test_56_malformed_usage_detail_counts_are_rejected(self):
    malformed_details = (
      ("prompt_tokens_details", "cached_tokens", True),
      ("prompt_tokens_details", "cached_tokens", -1),
      ("completion_tokens_details", "reasoning_tokens", "1"),
      ("completion_tokens_details", "reasoning_tokens", True),
      ("completion_tokens_details", "reasoning_tokens", -1),
    )
    for detail_field, token_field, malformed in malformed_details:
      with self.subTest(detail_field=detail_field, token_field=token_field,
                        malformed=malformed):
        with self.assertRaises(LLMMalformedResponseError):
          normalize_chat_response(_chat(usage={
            detail_field: {token_field: malformed}}))

  def test_57_malformed_usage_error_does_not_expose_payload(self):
    secret = "TOP-SECRET-USAGE"
    with self.assertRaises(LLMMalformedResponseError) as caught:
      normalize_chat_response(_chat(usage={"prompt_tokens": secret}))
    self.assertEqual(
      "Malformed usage metadata in modern OpenAI response",
      str(caught.exception))
    self.assertNotIn(secret, str(caught.exception))

  def test_58_absent_request_id_remains_optional(self):
    response = _chat()
    del response["_request_id"]
    self.assertIsNone(normalize_chat_response(response).request_id)


if __name__ == "__main__":
  unittest.main()
