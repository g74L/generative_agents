from contextlib import redirect_stdout
from datetime import datetime
from decimal import Decimal
import ast
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.cognitive_modules import reflect as reflect_module
from persona.prompt_template import gpt_structure
from persona.prompt_template import run_gpt_prompt
from persona.prompt_template.chat_runtime import (
  M5_CHAT_MODEL,
  ModernChatCallerNotAllowedError,
  build_modern_chat_runtime_config,
  use_modern_chat_runtime,
)
from persona.prompt_template.completion_runtime import (
  COMPLETION_COMPAT_MODEL,
  CompletionCompatCallerNotAllowedError,
  M6_DEFERRED_CALLERS,
  M6_REPLAY_CALLER_ALLOWLIST,
  LegacyModelInvocationDetectedError,
  ModernCompletionCompatRequest,
  ModernCompletionCompatRequestError,
  ModernCompletionRuntimeConfig,
  assert_no_legacy_model_invocation,
  build_modern_completion_runtime_config,
  get_modern_completion_runtime_config,
  is_modern_completion_runtime_active,
  request_from_legacy_parameters,
  run_modern_completion_compat,
  use_modern_completion_runtime,
  validate_completion_compat_caller,
)
from persona.prompt_template.cost_ledger import (
  COMPLETE,
  CostLedgerContext,
  ModelPricing,
  PricingSnapshot,
  build_cost_ledger_records,
  summarize_cost_ledger,
  use_cost_ledger_context,
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
  COMPLETION_COMPAT,
  EMBEDDING,
  ERROR,
  MEMORY_WRITE,
  SUCCESS,
  FakeProvider,
  LLMReplayContext,
  OpenAILegacyProvider,
  clear_telemetry,
  get_chat_provider,
  get_completion_compat_provider,
  get_completion_provider,
  get_embedding_provider,
  get_llm_replay_context,
  get_provider,
  get_telemetry,
  reset_embedding_cache,
  reset_provider,
  use_provider,
  use_llm_replay_context,
)
from persona.prompt_template.llm_provider_config import (
  LEGACY_TRANSPORT,
  MODERN_OPENAI,
  MODERN_SDK_MODE,
  MODERN_TRANSPORT,
  reset_llm_provider_config,
)
from persona.prompt_template.modern_openai_provider import (
  LLMAuthenticationError,
  LLMIncompleteResponseError,
  LLMInvalidRequestError,
  LLMRefusalError,
  LLMTimeoutError,
  ModernChatResponseValidationError,
  ModernOpenAIClientAdapter,
  ModernOpenAIProvider,
  NormalizedEmbeddingResponse,
  NormalizedTextResponse,
  NormalizedUsage,
)


def normalized_response(content="answer", model=COMPLETION_COMPAT_MODEL,
                        request_id="req-m6", finish_reason="stop",
                        status="completed", usage=None):
  return NormalizedTextResponse(
    content, model, request_id, finish_reason, status,
    usage or NormalizedUsage(20, 8, 3, 2))


class FakeCompletionCompatAdapter:
  def __init__(self, *responses):
    self.responses = list(responses)
    self.calls = []

  def create_chat(self, **kwargs):
    self.calls.append(kwargs)
    if not self.responses:
      raise AssertionError("No CompletionCompat response configured")
    response = self.responses.pop(0)
    if isinstance(response, Exception):
      raise response
    return response


class FakeModernChatEndpoint:
  def __init__(self, *responses):
    self.responses = list(responses)
    self.calls = []

  def create(self, **kwargs):
    self.calls.append(kwargs)
    if not self.responses:
      raise AssertionError("No modern Chat response configured")
    response = self.responses.pop(0)
    if isinstance(response, Exception):
      raise response
    return response


def sdk_chat_response(content="answer", finish_reason="stop", status=None,
                      input_tokens=20, output_tokens=8):
  response = {
    "choices": [{
      "message": {"content": content, "refusal": None},
      "finish_reason": finish_reason,
    }],
    "model": COMPLETION_COMPAT_MODEL,
    "_request_id": "req-planning-compat",
    "usage": {
      "prompt_tokens": input_tokens,
      "completion_tokens": output_tokens,
      "total_tokens": input_tokens + output_tokens,
    },
  }
  if status is not None:
    response["status"] = status
  return response


def modern_sdk_adapter(*responses):
  endpoint = FakeModernChatEndpoint(*responses)
  client = SimpleNamespace(
    chat=SimpleNamespace(completions=endpoint),
    embeddings=SimpleNamespace(create=lambda **unused: None))
  return ModernOpenAIClientAdapter(client=client), endpoint


class FakeEmbeddingAdapter:
  def create_embedding(self, **kwargs):
    return NormalizedEmbeddingResponse(
      (1.0,) + (0.0,) * 1535, TEXT_EMBEDDING_3_SMALL_MODEL,
      "req-embedding", NormalizedUsage(1, None))


EXPECTED_COMPLETION_CALLERS = {
  "run_gpt_prompt_wake_up_hour": "wake_up_hour",
  "run_gpt_prompt_daily_plan": "daily_plan",
  "run_gpt_prompt_generate_hourly_schedule": "generate_hourly_schedule",
  "run_gpt_prompt_task_decomp": "task_decomp",
  "run_gpt_prompt_action_sector": "action_sector",
  "run_gpt_prompt_action_arena": "action_arena",
  "run_gpt_prompt_action_game_object": "action_game_object",
  "run_gpt_prompt_event_triple": "event_triple",
  "run_gpt_prompt_act_obj_event_triple": "act_obj_event_triple",
  "run_gpt_prompt_new_decomp_schedule": "new_decomp_schedule",
  "run_gpt_prompt_decide_to_talk": "decide_to_talk",
  "run_gpt_prompt_decide_to_react": "decide_to_react",
  "run_gpt_prompt_create_conversation": "create_conversation",
  "run_gpt_prompt_extract_keywords": "extract_keywords",
  "run_gpt_prompt_keyword_to_thoughts": "keyword_to_thoughts",
  "run_gpt_prompt_convo_to_thoughts": "convo_to_thoughts",
  "run_gpt_prompt_focal_pt": "focal_pt",
  "run_gpt_prompt_insight_and_guidance": "insight_and_guidance",
  "run_gpt_prompt_generate_next_convo_line": "generate_next_convo_line",
  "run_gpt_prompt_generate_whisper_inner_thought": (
    "generate_whisper_inner_thought"),
  "run_gpt_prompt_planning_thought_on_convo": "planning_thought_on_convo",
  "run_gpt_prompt_memo_on_convo": "memo_on_convo",
}


class ModernCompletionMigrationTests(unittest.TestCase):
  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_cache()
    clear_telemetry()
    self.sleep_patch = patch.object(gpt_structure, "temp_sleep")
    self.sleep_patch.start()

  def tearDown(self):
    self.sleep_patch.stop()
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_cache()
    clear_telemetry()

  def parameters(self, **changes):
    values = {
      "engine": "text-davinci-003",
      "temperature": 0.25,
      "max_tokens": 50,
      "top_p": 0.9,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.2,
      "stream": False,
      "stop": None,
    }
    values.update(changes)
    return values

  def request(self, **changes):
    values = {
      "prompt": "legacy prompt",
      "source_model": "text-davinci-003",
      "caller_id": "daily_plan",
      "temperature": 0.25,
      "max_tokens": 50,
      "top_p": 0.9,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.2,
      "stream": False,
      "stop": None,
    }
    values.update(changes)
    return ModernCompletionCompatRequest(**values)

  def execute(self, adapter, request=None, config=None):
    with use_modern_completion_runtime(
        config or build_modern_completion_runtime_config(), adapter):
      return run_modern_completion_compat(request or self.request())

  def execute_wrapper(self, adapter, prompt="legacy prompt", parameters=None,
                      config=None, caller_id="daily_plan"):
    with use_modern_completion_runtime(
        config or build_modern_completion_runtime_config(), adapter):
      return gpt_structure.GPT_request(
        prompt, parameters or self.parameters(), caller_id=caller_id)

  def test_01_config_is_canonical(self):
    config = build_modern_completion_runtime_config()
    self.assertEqual(
      (MODERN_OPENAI, MODERN_TRANSPORT, MODERN_SDK_MODE,
       M5_CHAT_MODEL, COMPLETION_COMPAT),
      (config.provider_kind, config.transport_kind, config.sdk_mode,
       config.model, config.operation))

  def test_02_incoherent_config_is_rejected(self):
    with self.assertRaises(ValueError):
      ModernCompletionRuntimeConfig(transport_kind=LEGACY_TRANSPORT)

  def test_03_default_legacy_completion_is_unchanged(self):
    legacy = FakeProvider()
    legacy.queue_completion_response("legacy output")
    with use_provider(legacy):
      result = gpt_structure.GPT_request("prompt", self.parameters())
    self.assertEqual("legacy output", result)
    self.assertEqual(COMPLETION, legacy.calls[0].operation)

  def test_04_runtime_is_explicit_opt_in(self):
    self.assertFalse(is_modern_completion_runtime_active())
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(),
        FakeCompletionCompatAdapter(normalized_response())):
      self.assertTrue(is_modern_completion_runtime_active())
    self.assertFalse(is_modern_completion_runtime_active())

  def test_05_runtime_selects_modern_provider(self):
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(),
        FakeCompletionCompatAdapter(normalized_response())) as provider:
      self.assertIsInstance(provider, ModernOpenAIProvider)
      self.assertIs(provider, get_completion_compat_provider())

  def test_06_legacy_provider_tripwire_is_not_reached(self):
    legacy = FakeProvider()
    adapter = FakeCompletionCompatAdapter(normalized_response())
    with use_provider(legacy):
      self.assertEqual("answer", self.execute_wrapper(adapter))
    self.assertEqual([], legacy.calls)

  def test_07_modern_error_never_falls_back(self):
    legacy = FakeProvider()
    legacy.queue_completion_response("forbidden")
    adapter = FakeCompletionCompatAdapter(
      LLMInvalidRequestError("modern failure"))
    with use_provider(legacy), redirect_stdout(io.StringIO()):
      result = self.execute_wrapper(adapter)
    self.assertEqual("TOKEN LIMIT EXCEEDED", result)
    self.assertEqual([], legacy.calls)

  def test_08_pinned_model_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter)
    self.assertEqual(M5_CHAT_MODEL, adapter.calls[0]["model"])

  def test_09_prompt_is_byte_exact_user_content(self):
    prompt = "  legacy\r\nprompt ✓  "
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(prompt=prompt))
    self.assertEqual(prompt, adapter.calls[0]["messages"][0]["content"])

  def test_10_no_system_message_is_added(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter)
    self.assertEqual(["user"], [
      item["role"] for item in adapter.calls[0]["messages"]])

  def test_11_temperature_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(temperature=0.75))
    self.assertEqual(0.75, adapter.calls[0]["temperature"])

  def test_12_max_tokens_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(max_tokens=77))
    self.assertEqual(77, adapter.calls[0]["max_tokens"])

  def test_13_stop_string_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(stop="END"))
    self.assertEqual("END", adapter.calls[0]["stop"])

  def test_14_stop_tuple_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(stop=("A", "B")))
    self.assertEqual(("A", "B"), adapter.calls[0]["stop"])

  def test_15_stop_list_is_copied_and_forwarded(self):
    source = ["A", "B"]
    request = self.request(stop=source)
    source[0] = "changed"
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, request)
    self.assertEqual(("A", "B"), adapter.calls[0]["stop"])

  def test_15b_historical_newline_stop_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(stop=["\n"]))
    self.assertEqual(("\n",), adapter.calls[0]["stop"])

  def test_15c_blank_stop_sequences_fail_before_provider(self):
    rejected = (
      "", " ", "   ", "\t", "\t\t", " \t ", "\r",
      [""], ["   "], ["\t"], ["END", "   "], ("\t", "\n"), [], ())
    for stop in rejected:
      with self.subTest(stop=repr(stop)):
        clear_telemetry()
        adapter = FakeCompletionCompatAdapter(normalized_response())
        with self.assertRaises(ModernCompletionCompatRequestError):
          self.execute(adapter, self.request(stop=stop))
        self.assertEqual([], adapter.calls)
        self.assertEqual((), get_telemetry())

  def test_15d_valid_stop_sequences_are_preserved_exactly(self):
    accepted = (
      "\n", "\r\n", "\n\n", "END", " END ", "\tEND", "END\t",
      ["\n"], ["\r\n"], ["END"], ["\n", "END"],
      ("\n", "\r\n"), (" END ", "\n"))
    for stop in accepted:
      with self.subTest(stop=repr(stop)):
        clear_telemetry()
        adapter = FakeCompletionCompatAdapter(normalized_response())
        self.execute(adapter, self.request(stop=stop))
        expected = stop if isinstance(stop, str) else tuple(stop)
        self.assertEqual(expected, adapter.calls[0]["stop"])

  def test_15e_stop_list_is_defensively_frozen_after_validation(self):
    source = ["\n", "END"]
    request = self.request(stop=source)
    source[:] = ["changed"]
    self.assertEqual(("\n", "END"), request.stop)

  def test_15f_none_stop_is_accepted(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    request = self.request(stop=None)
    self.assertIsNone(request.stop)
    self.assertEqual("answer", self.execute(adapter, request))

  def test_16_top_p_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(top_p=0.7))
    self.assertEqual(0.7, adapter.calls[0]["top_p"])

  def test_17_frequency_penalty_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(frequency_penalty=-0.5))
    self.assertEqual(-0.5, adapter.calls[0]["frequency_penalty"])

  def test_18_presence_penalty_is_forwarded(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute(adapter, self.request(presence_penalty=0.6))
    self.assertEqual(0.6, adapter.calls[0]["presence_penalty"])

  def test_19_output_string_is_preserved(self):
    self.assertEqual("exact", self.execute(
      FakeCompletionCompatAdapter(normalized_response("exact"))))

  def test_20_output_whitespace_is_preserved(self):
    content = "  exact output \n"
    self.assertEqual(content, self.execute(
      FakeCompletionCompatAdapter(normalized_response(content))))

  def test_21_unicode_output_is_preserved(self):
    content = "caffè ✓ 東京"
    self.assertEqual(content, self.execute(
      FakeCompletionCompatAdapter(normalized_response(content))))

  def test_22_markdown_output_is_preserved(self):
    content = "**bold**\n- item"
    self.assertEqual(content, self.execute(
      FakeCompletionCompatAdapter(normalized_response(content))))

  def test_23_json_like_output_is_preserved(self):
    content = '{"answer": [1, 2]}'
    self.assertEqual(content, self.execute(
      FakeCompletionCompatAdapter(normalized_response(content))))

  def test_24_historical_parser_remains_authoritative(self):
    adapter = FakeCompletionCompatAdapter(normalized_response(" 42 "))
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(), adapter):
      result = gpt_structure.safe_generate_response(
        "prompt", self.parameters(), 1, -1,
        lambda value, prompt="": value.strip().isdigit(),
        lambda value, prompt="": int(value.strip()), caller_id="daily_plan")
    self.assertEqual(42, result)

  def test_25_historical_fail_safe_remains_authoritative(self):
    adapter = FakeCompletionCompatAdapter(
      normalized_response("bad"), normalized_response("still bad"))
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(), adapter):
      result = gpt_structure.safe_generate_response(
        "prompt", self.parameters(), 2, "existing fallback",
        lambda value, prompt="": False,
        lambda value, prompt="": value, caller_id="daily_plan")
    self.assertEqual("existing fallback", result)

  def test_26_semantic_retry_count_is_unchanged(self):
    adapter = FakeCompletionCompatAdapter(
      normalized_response("bad"), normalized_response("good"))
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(), adapter):
      result = gpt_structure.safe_generate_response(
        "prompt", self.parameters(), 2, "fallback",
        lambda value, prompt="": value == "good",
        lambda value, prompt="": value, caller_id="daily_plan")
    self.assertEqual("good", result)
    self.assertEqual(2, len(adapter.calls))

  def test_27_provider_retry_is_explicit(self):
    adapter = FakeCompletionCompatAdapter(
      LLMTimeoutError("timeout"), normalized_response("ok"))
    result = self.execute(
      adapter, config=build_modern_completion_runtime_config(
        application_retry_count=1))
    self.assertEqual("ok", result)
    self.assertEqual(2, len(adapter.calls))

  def test_28_sdk_retry_is_zero(self):
    self.assertEqual(
      0, build_modern_completion_runtime_config().provider_config().sdk_retry_count)

  def test_29_provider_retries_share_one_logical_call(self):
    adapter = FakeCompletionCompatAdapter(
      LLMTimeoutError("timeout"), normalized_response())
    self.execute(adapter, config=build_modern_completion_runtime_config(
      application_retry_count=1))
    self.assertEqual(1, len({event.logical_call_id
                            for event in get_telemetry()}))

  def test_30_physical_attempts_are_exact(self):
    adapter = FakeCompletionCompatAdapter(
      LLMTimeoutError("timeout"), normalized_response())
    self.execute(adapter, config=build_modern_completion_runtime_config(
      application_retry_count=1))
    self.assertEqual([1, 2], [event.physical_attempt
                             for event in get_telemetry()])

  def test_31_malformed_metadata_fails_closed(self):
    adapter = FakeCompletionCompatAdapter(
      normalized_response(request_id={"bad": True}))
    with self.assertRaises(ModernChatResponseValidationError):
      self.execute(adapter)
    self.assertEqual([ERROR], [event.outcome for event in get_telemetry()])

  def test_32_telemetry_and_ledger_are_private(self):
    prompt = "M6-PRIVATE-PROMPT"
    output = "M6-PRIVATE-OUTPUT"
    self.execute(FakeCompletionCompatAdapter(normalized_response(output)),
                 self.request(prompt=prompt))
    exported = repr((get_telemetry(),
                     build_cost_ledger_records(get_telemetry())))
    self.assertNotIn(prompt, exported)
    self.assertNotIn(output, exported)

  def test_33_success_telemetry_is_complete(self):
    self.execute(FakeCompletionCompatAdapter(normalized_response(
      model="gpt-4o-mini-provider-alias")))
    event = get_telemetry()[0]
    self.assertEqual((COMPLETION_COMPAT, SUCCESS, M5_CHAT_MODEL,
                      "gpt-4o-mini-provider-alias", "req-m6",
                      "stop", "completed", 20, 8),
                     (event.operation, event.outcome, event.model_or_engine,
                      event.response_model, event.request_id,
                      event.finish_reason, event.response_status,
                      event.input_tokens, event.output_tokens))

  def test_34_error_telemetry_is_typed(self):
    adapter = FakeCompletionCompatAdapter(
      LLMInvalidRequestError("invalid", request_id="req-error"))
    with self.assertRaises(LLMInvalidRequestError):
      self.execute(adapter)
    event = get_telemetry()[0]
    self.assertEqual((ERROR, "LLMInvalidRequestError", "req-error"),
                     (event.outcome, event.error_type, event.request_id))

  def test_35_cost_ledger_supports_completion_compat(self):
    call_context = LLMReplayContext(
      simulation_id="sim-m6", simulation_step=7,
      actor_id="actor-m6", cognitive_category="PLANNING")
    with use_llm_replay_context(call_context):
      self.execute(FakeCompletionCompatAdapter(normalized_response(
        usage=NormalizedUsage(100, 40, 10, 5))))
    pricing = PricingSnapshot(
      "m6-synthetic", 1, "USD", "synthetic", (
        ModelPricing(M5_CHAT_MODEL,
          input_per_million=Decimal("2"),
          cached_input_per_million=Decimal("1"),
          output_per_million=Decimal("4")),), "synthetic only")
    records = build_cost_ledger_records(get_telemetry(), pricing)
    summary = summarize_cost_ledger(records)
    self.assertEqual(COMPLETE, records[0].token_usage_status)
    self.assertEqual((100, 40, 10, 5, 140), (
      records[0].input_tokens, records[0].output_tokens,
      records[0].cached_input_tokens, records[0].reasoning_tokens,
      records[0].total_tokens))
    self.assertIsNotNone(records[0].logical_call_id)
    self.assertEqual(1, records[0].attempt)
    self.assertEqual(("PLANNING", "actor-m6", "sim-m6", 7), (
      records[0].cognitive_category, records[0].actor_id,
      records[0].simulation_id, records[0].simulation_step))
    self.assertEqual("daily_plan", records[0].caller_id)
    self.assertEqual(Decimal("0.000350000000"),
                     records[0].estimated_total_cost_usd)
    self.assertEqual(COMPLETION_COMPAT, summary.by_operation[0][0])
    self.assertEqual(M5_CHAT_MODEL, summary.by_model[0][0])
    self.assertEqual(MODERN_OPENAI, summary.by_provider[0][0])
    self.assertEqual(SUCCESS, summary.by_outcome[0][0])

  def test_36_context_reset_restores_inactive_state(self):
    config = build_modern_completion_runtime_config()
    with use_modern_completion_runtime(
        config, FakeCompletionCompatAdapter(normalized_response())):
      self.assertIs(config, get_modern_completion_runtime_config())
    self.assertIsNone(get_modern_completion_runtime_config())
    self.assertIsNone(get_completion_compat_provider())

  def test_37_exception_resets_context(self):
    with self.assertRaises(RuntimeError):
      with use_modern_completion_runtime(
          build_modern_completion_runtime_config(),
          FakeCompletionCompatAdapter(normalized_response())):
        raise RuntimeError("injected")
    self.assertIsNone(get_modern_completion_runtime_config())
    self.assertIsNone(get_completion_compat_provider())

  def test_38_completion_compat_does_not_activate_chat(self):
    before = get_chat_provider()
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(),
        FakeCompletionCompatAdapter(normalized_response())):
      self.assertIs(before, get_chat_provider())

  def test_39_completion_compat_does_not_activate_embedding(self):
    before = get_embedding_provider()
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(),
        FakeCompletionCompatAdapter(normalized_response())):
      self.assertIs(before, get_embedding_provider())

  def test_40_chat_and_embedding_do_not_activate_completion_compat(self):
    with use_modern_chat_runtime(
        build_modern_chat_runtime_config(),
        FakeCompletionCompatAdapter(normalized_response())):
      self.assertIsNone(get_completion_compat_provider())
    with use_embedding_runtime(
        build_modern_embedding_runtime_config(), FakeEmbeddingAdapter()):
      self.assertIsNone(get_completion_compat_provider())

  def test_41_combined_context_matrix_is_independent(self):
    legacy = FakeProvider()
    with use_provider(legacy):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), FakeEmbeddingAdapter()):
        embedding_provider = get_embedding_provider()
        with use_modern_chat_runtime(
            build_modern_chat_runtime_config(),
            FakeCompletionCompatAdapter(normalized_response())):
          chat_provider = get_chat_provider()
          with use_modern_completion_runtime(
              build_modern_completion_runtime_config(),
              FakeCompletionCompatAdapter(normalized_response())):
            compat_provider = get_completion_compat_provider()
            self.assertEqual(3, len({id(chat_provider), id(embedding_provider),
                                     id(compat_provider)}))
            self.assertIs(legacy, get_completion_provider())

  def test_42_legacy_model_detection_seam_is_typed(self):
    with self.assertRaises(LegacyModelInvocationDetectedError):
      assert_no_legacy_model_invocation("text-davinci-003")

  def test_43_transport_never_uses_text_davinci(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    self.execute_wrapper(adapter, parameters=self.parameters(
      engine="text-davinci-002"))
    self.assertNotIn("text-davinci", adapter.calls[0]["model"])

  def test_44_gpt_35_is_forbidden_in_modern_config(self):
    with self.assertRaises(LegacyModelInvocationDetectedError):
      ModernCompletionRuntimeConfig(model="gpt-3.5-turbo")

  def test_45_ada_is_forbidden_by_detection_seam(self):
    with self.assertRaises(LegacyModelInvocationDetectedError):
      assert_no_legacy_model_invocation("text-embedding-ada-002")

  def test_46_no_real_network_is_used(self):
    with patch.object(socket, "getaddrinfo",
                      side_effect=AssertionError("DNS reached")) as dns, patch.object(
        socket, "create_connection",
        side_effect=AssertionError("network reached")) as connection, patch.object(
          gpt_structure.openai.Completion, "create",
          side_effect=AssertionError("legacy Completion reached")) as completion, patch.object(
          gpt_structure.openai.ChatCompletion, "create",
          side_effect=AssertionError("legacy Chat reached")) as chat, patch(
          "persona.prompt_template.modern_openai_provider._create_modern_sdk_client",
          side_effect=AssertionError("real modern client reached")) as factory:
      self.execute(FakeCompletionCompatAdapter(normalized_response()))
    dns.assert_not_called()
    connection.assert_not_called()
    completion.assert_not_called()
    chat.assert_not_called()
    factory.assert_not_called()

  def test_47_storage_is_not_mutated(self):
    storage = BACKEND_SERVER.parents[1] / "environment" / "frontend_server" / "storage"
    before = storage.stat().st_mtime_ns
    self.execute(FakeCompletionCompatAdapter(normalized_response()))
    self.assertEqual(before, storage.stat().st_mtime_ns)

  def test_48_clean_import_does_not_load_utils_or_cognition(self):
    script = (
      "import importlib,json,sys\n"
      f"sys.path.insert(0,{str(BACKEND_SERVER)!r})\n"
      "importlib.import_module('persona.prompt_template.completion_runtime')\n"
      "importlib.import_module('persona.prompt_template.gpt_structure')\n"
      "print(json.dumps(sorted(sys.modules)))\n")
    completed = subprocess.run(
      [sys.executable, "-I", "-c", script], check=True,
      capture_output=True, text=True, timeout=30)
    modules = set(json.loads(completed.stdout))
    forbidden = {"planning", "reflection", "conversation",
                 "associative_memory", "Reverie"}
    self.assertNotIn("utils", modules)
    self.assertNotIn("reverie.backend_server.utils", modules)
    self.assertFalse(any(name.rsplit(".", 1)[-1] in forbidden
                         for name in modules))

  def test_49_legacy_golden_request_shape_is_unchanged(self):
    legacy = FakeProvider()
    legacy.queue_completion_response("legacy")
    parameters = self.parameters(stop=["\n"])
    with use_provider(legacy):
      gpt_structure.GPT_request("exact prompt", parameters)
    self.assertEqual("text-davinci-003",
                     legacy.calls[0].arguments["model"])
    self.assertEqual("exact prompt", legacy.calls[0].arguments["prompt"])

  def test_50_parallel_parser_compatibility_harness(self):
    fixtures = (
      ("42", lambda x: int(x), 42),
      ("yes", lambda x: x == "yes", True),
      ('{"x": 1}', json.loads, {"x": 1}),
      ("a,b", lambda x: x.split(","), ["a", "b"]),
      ("2) choice", lambda x: x.split(") ", 1)[1], "choice"),
      ("short answer", lambda x: x, "short answer"),
      (" extra prose ", lambda x: x.strip(), "extra prose"),
    )
    for content, parser, expected in fixtures:
      adapter = FakeCompletionCompatAdapter(normalized_response(content))
      with use_modern_completion_runtime(
          build_modern_completion_runtime_config(), adapter):
        result = gpt_structure.safe_generate_response(
          "prompt", self.parameters(), 1, "fallback",
          lambda value, prompt="", p=parser: self._parser_accepts(p, value),
          lambda value, prompt="", p=parser: p(value),
          caller_id="daily_plan")
      self.assertEqual(expected, result)

  @staticmethod
  def _parser_accepts(parser, value):
    try:
      parser(value)
      return True
    except Exception:
      return False

  def test_51_cognitive_modules_are_not_wired_to_runtime(self):
    cognitive = BACKEND_SERVER / "persona" / "cognitive_modules"
    combined = "".join(path.read_text(encoding="utf-8")
                       for path in cognitive.glob("*.py"))
    self.assertNotIn("completion_runtime", combined)
    self.assertNotIn("COMPLETION_COMPAT", combined)

  def test_52_replay_caller_inventory_is_explicit_and_disjoint(self):
    self.assertEqual(18, len(M6_REPLAY_CALLER_ALLOWLIST))
    self.assertEqual(4, len(M6_DEFERRED_CALLERS))
    self.assertFalse(set(M6_REPLAY_CALLER_ALLOWLIST) & set(M6_DEFERRED_CALLERS))

  def test_53_wrapper_to_caller_mapping_is_literal_and_complete(self):
    source_path = (BACKEND_SERVER / "persona" / "prompt_template"
                   / "run_gpt_prompt.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    actual = {}
    for function in (node for node in tree.body
                     if isinstance(node, ast.FunctionDef)):
      calls = [node for node in ast.walk(function)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "safe_generate_response"]
      if not calls:
        continue
      self.assertEqual(1, len(calls), function.name)
      self.assertEqual(6, len(calls[0].args), function.name)
      keywords = {item.arg: item.value for item in calls[0].keywords}
      self.assertEqual({"caller_id"}, set(keywords), function.name)
      self.assertIsInstance(keywords["caller_id"], ast.Constant)
      self.assertIsInstance(keywords["caller_id"].value, str)
      actual[function.name] = keywords["caller_id"].value
    self.assertEqual(EXPECTED_COMPLETION_CALLERS, actual)
    self.assertEqual(set(M6_REPLAY_CALLER_ALLOWLIST),
                     set(actual.values()) - set(M6_DEFERRED_CALLERS))
    self.assertEqual(set(M6_DEFERRED_CALLERS),
                     set(actual.values()) & set(M6_DEFERRED_CALLERS))

  def test_54_caller_policy_uses_no_stack_introspection(self):
    paths = (
      BACKEND_SERVER / "persona" / "prompt_template" / "completion_runtime.py",
      BACKEND_SERVER / "persona" / "prompt_template" / "gpt_structure.py",
    )
    combined = "".join(path.read_text(encoding="utf-8") for path in paths)
    for forbidden in ("inspect.stack", "sys._getframe", "traceback.extract_stack"):
      self.assertNotIn(forbidden, combined)

  def test_55_all_authorized_callers_reach_provider_telemetry_and_ledger(self):
    for caller_id in M6_REPLAY_CALLER_ALLOWLIST:
      with self.subTest(caller_id=caller_id):
        clear_telemetry()
        adapter = FakeCompletionCompatAdapter(normalized_response("accepted"))
        result = self.execute(adapter, self.request(caller_id=caller_id))
        self.assertEqual("accepted", result)
        self.assertEqual(1, len(adapter.calls))
        event = get_telemetry()[0]
        record = build_cost_ledger_records((event,))[0]
        self.assertEqual(caller_id, event.caller_id)
        self.assertEqual(caller_id, record.caller_id)

  def test_56_deferred_callers_hard_fail_without_attempt_or_fallback(self):
    for caller_id in M6_DEFERRED_CALLERS:
      with self.subTest(caller_id=caller_id):
        clear_telemetry()
        adapter = FakeCompletionCompatAdapter(normalized_response())
        legacy = FakeProvider()
        legacy.queue_completion_response("forbidden fallback")
        with use_provider(legacy), use_modern_completion_runtime(
            build_modern_completion_runtime_config(), adapter):
          with self.assertRaises(CompletionCompatCallerNotAllowedError):
            gpt_structure.GPT_request(
              "private prompt", self.parameters(), caller_id=caller_id)
        self.assertEqual([], adapter.calls)
        self.assertEqual([], legacy.calls)
        self.assertEqual((), get_telemetry())

  def test_57_unknown_missing_and_invalid_callers_hard_fail(self):
    invalid = (
      "definitely_not_allowlisted_caller", "future_dynamic_caller",
      None, "", "   ", 1, True, [], {}, object())
    for caller_id in invalid:
      with self.subTest(caller_id=repr(caller_id)):
        with self.assertRaises(CompletionCompatCallerNotAllowedError):
          validate_completion_compat_caller(caller_id)

  def test_58_policy_failure_bypasses_five_semantic_attempts(self):
    adapter = FakeCompletionCompatAdapter(normalized_response())
    with use_modern_completion_runtime(
        build_modern_completion_runtime_config(), adapter):
      with self.assertRaises(CompletionCompatCallerNotAllowedError):
        gpt_structure.safe_generate_response(
          "private prompt", self.parameters(), 5, "fallback",
          lambda value, prompt="": False,
          lambda value, prompt="": value,
          caller_id="future_dynamic_caller")
    self.assertEqual([], adapter.calls)
    self.assertEqual((), get_telemetry())

  def test_59_real_authorized_and_deferred_wrappers_apply_policy(self):
    allowed_adapter = FakeCompletionCompatAdapter(normalized_response("7 am"))
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt",
        return_value="exact wrapper prompt"), use_modern_completion_runtime(
        build_modern_completion_runtime_config(), allowed_adapter):
      output = run_gpt_prompt.run_gpt_prompt_wake_up_hour(
        object(), test_input=["fixture"])[0]
    self.assertEqual(7, output)
    self.assertEqual("wake_up_hour", get_telemetry()[0].caller_id)
    self.assertEqual(20, allowed_adapter.calls[0]["max_tokens"])
    self.assertEqual(("\n",), allowed_adapter.calls[0]["stop"])

    clear_telemetry()
    deferred_adapter = FakeCompletionCompatAdapter(normalized_response())
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt",
        return_value="private wrapper prompt"), use_modern_completion_runtime(
        build_modern_completion_runtime_config(), deferred_adapter):
      with self.assertRaises(CompletionCompatCallerNotAllowedError):
        run_gpt_prompt.run_gpt_prompt_extract_keywords(
          object(), "description", test_input=["fixture"])
    self.assertEqual([], deferred_adapter.calls)
    self.assertEqual((), get_telemetry())

  def test_60_call_time_context_overrides_late_ledger_attribution(self):
    call_context = LLMReplayContext(
      caller_id="external-caller", cognitive_category="PLANNING",
      actor_id="isabella_rodriguez", simulation_id="ego-vivens-lab-01",
      simulation_step=0)
    with use_llm_replay_context(call_context):
      self.execute_wrapper(
        FakeCompletionCompatAdapter(normalized_response("captured")),
        caller_id="daily_plan")
    event = get_telemetry()[0]
    with use_cost_ledger_context(CostLedgerContext(
        caller_id="late-caller", cognitive_category="REFLECTION",
        actor_id="other", simulation_id="other-sim", simulation_step=99)):
      record = build_cost_ledger_records((event,))[0]
    expected = ("daily_plan", "PLANNING", "isabella_rodriguez",
                "ego-vivens-lab-01", 0)
    self.assertEqual(expected, (
      event.caller_id, event.cognitive_category, event.actor_id,
      event.simulation_id, event.simulation_step))
    self.assertEqual(expected, (
      record.caller_id, record.cognitive_category, record.actor_id,
      record.simulation_id, record.simulation_step))

  def test_61_replay_context_nests_and_resets_after_exception(self):
    baseline = get_llm_replay_context()
    outer = LLMReplayContext(simulation_id="outer")
    inner = LLMReplayContext(
      simulation_id="inner", actor_id="actor", simulation_step=0)
    with use_llm_replay_context(outer):
      self.assertEqual(outer, get_llm_replay_context())
      with use_llm_replay_context(inner):
        self.assertEqual(inner, get_llm_replay_context())
      self.assertEqual(outer, get_llm_replay_context())
      with self.assertRaises(RuntimeError):
        with use_llm_replay_context(inner):
          raise RuntimeError("injected")
      self.assertEqual(outer, get_llm_replay_context())
    self.assertEqual(baseline, get_llm_replay_context())

  def test_62_semantic_retries_keep_call_time_attribution(self):
    adapter = FakeCompletionCompatAdapter(
      normalized_response("bad-1"), normalized_response("bad-2"),
      normalized_response("bad-3"), normalized_response("good"))
    context = LLMReplayContext(
      cognitive_category="PLANNING", actor_id="actor",
      simulation_id="simulation", simulation_step=0)
    with use_llm_replay_context(context), use_modern_completion_runtime(
        build_modern_completion_runtime_config(), adapter):
      result = gpt_structure.safe_generate_response(
        "prompt", self.parameters(), 5, "fallback",
        lambda value, prompt="": value == "good",
        lambda value, prompt="": value,
        caller_id="daily_plan")
    self.assertEqual("good", result)
    events = get_telemetry()
    self.assertEqual(4, len(events))
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual([1, 2, 3, 4],
                     [event.physical_attempt for event in events])
    self.assertEqual({("daily_plan", "PLANNING", "actor", "simulation", 0)}, {
      (event.caller_id, event.cognitive_category, event.actor_id,
       event.simulation_id, event.simulation_step) for event in events})

  def test_63_prompt_and_output_edge_cases_remain_byte_exact(self):
    fixtures = (
      "  leading and trailing  ", "line1\r\nline2", "line1\nline2",
      "Unicode caf\u00e8 \u2713", '{"json": [1]}', "**markdown**",
      "<DELIMITER>value</DELIMITER>", " \u200b ")
    for value in fixtures:
      with self.subTest(value=repr(value)):
        clear_telemetry()
        adapter = FakeCompletionCompatAdapter(normalized_response(value))
        result = self.execute(adapter, self.request(prompt=value))
        self.assertEqual(value, adapter.calls[0]["messages"][0]["content"])
        self.assertEqual(value, result)

  def test_64_m6_attribution_remains_content_private(self):
    prompt = "PRIVATE-PROMPT-M6RH2"
    output = "PRIVATE-OUTPUT-M6RH2"
    context = LLMReplayContext(
      cognitive_category="PLANNING", actor_id="actor",
      simulation_id="simulation", simulation_step=0)
    with use_llm_replay_context(context):
      self.execute(FakeCompletionCompatAdapter(normalized_response(output)),
                   self.request(prompt=prompt))
    exported = repr((get_telemetry(),
                     build_cost_ledger_records(get_telemetry())))
    self.assertNotIn(prompt, exported)
    self.assertNotIn(output, exported)
    for forbidden in ("messages", "parser input", "fail-safe", "transcript"):
      self.assertNotIn(forbidden, exported)

  def test_64a_wake_up_failure_classifier_matches_historical_validator(self):
    cases = (
      ("", "EMPTY_RESPONSE", False, 8),
      ("8am", None, True, 8),
      ("8 am", None, True, 8),
      ("8:00 am", "MINUTES_FORMAT_PRESENT", False, 8),
      ("Wake up at 8 am", "TEXT_BEFORE_NUMBER", False, 8),
      ("8", None, True, 8),
      ("8 pm", "PM_FORMAT_PRESENT", False, 8),
      ("13 am", None, True, 13),
      ("8 am because...", None, True, 8),
      (object(), "UNKNOWN_WAKE_UP_FORMAT", False, 8),
      ("eight", "AM_MARKER_MISSING", False, 8),
      ("eight am", "NON_INTEGER_PREFIX", False, 8),
    )

    for value, expected_code, expected_valid, expected_output in cases:
      with self.subTest(value=repr(value)):
        clear_telemetry()
        observed = {}

        def capture_without_provider(
            prompt, parameters, repeat, fail_safe, validate, clean_up,
            **kwargs):
          observed["repeat"] = repeat
          observed["valid"] = validate(value, prompt="private")
          return (clean_up(value, prompt="private")
                  if observed["valid"] else fail_safe)

        with patch.object(run_gpt_prompt, "debug", False), patch.object(
            run_gpt_prompt, "generate_prompt",
            return_value="private historical prompt"), patch.object(
              run_gpt_prompt, "safe_generate_response",
              side_effect=capture_without_provider):
          output = run_gpt_prompt.run_gpt_prompt_wake_up_hour(
            object(), test_input=["fixture"])[0]

        self.assertEqual(
          expected_code,
          run_gpt_prompt.classify_wake_up_format_failure(value))
        self.assertEqual(expected_valid, observed["valid"])
        self.assertEqual(expected_output, output)
        self.assertEqual(5, observed["repeat"])
        self.assertEqual((), get_telemetry())

  def test_64b_wake_up_diagnostic_is_content_private(self):
    secret = "PRIVATE-WAKE-UP-MARKER-9481"
    value = f"{secret} 8 am"
    classification = run_gpt_prompt.classify_wake_up_format_failure(value)
    self.assertEqual("TEXT_BEFORE_NUMBER", classification)
    exported = repr((classification, get_telemetry()))
    self.assertNotIn(secret, exported)

  def test_64c_wake_up_template_states_exact_output_contract(self):
    template = (
      BACKEND_SERVER / "persona" / "prompt_template" / "v2"
      / "wake_up_hour_v1.txt").read_text(encoding="utf-8")
    instruction = (
      'Respond only with an integer hour followed by "am", '
      'for example: 8 am.')
    exclusions = (
      "Do not include minutes, introductory text, explanations, "
      "or any additional text.")
    final_request = "!<INPUT 2>!'s wake up hour:"
    self.assertIn(instruction, template)
    self.assertIn(exclusions, template)
    self.assertIn(final_request, template)
    self.assertLess(template.index(instruction), template.index(final_request))
    self.assertLess(template.index(exclusions), template.index(final_request))

  def test_65_representative_real_completion_parsers_remain_authoritative(self):
    def invoke(response, callback):
      clear_telemetry()
      adapter = FakeCompletionCompatAdapter(normalized_response(response))
      with patch.object(run_gpt_prompt, "debug", False), patch.object(
          run_gpt_prompt, "generate_prompt", return_value="historical prompt"), (
          use_modern_completion_runtime(
            build_modern_completion_runtime_config(), adapter)):
        return callback()

    self.assertEqual(7, invoke(
      "7 am", lambda: run_gpt_prompt.run_gpt_prompt_wake_up_hour(
        object(), test_input=["fixture"])[0]))
    self.assertEqual(
      ["wake up and complete the morning routine at 7:00 am",
       "breakfast", "work"],
      invoke("Plan 1) breakfast, 2) work, 3) end", lambda:
        run_gpt_prompt.run_gpt_prompt_daily_plan(
          object(), 7, test_input=["fixture"])[0]))
    self.assertEqual("working", invoke(
      "working.", lambda:
        run_gpt_prompt.run_gpt_prompt_generate_hourly_schedule(
          object(), "09:00", [], [], test_input=["fixture"])[0]))

    scratch = SimpleNamespace(
      curr_time=datetime(2026, 1, 1, 9, 0), act_description="reading",
      planned_path=[], act_address="world:sector:room")
    persona = SimpleNamespace(
      name="Isabella", scratch=scratch,
      a_mem=SimpleNamespace(get_last_chat=lambda unused: None))
    target = SimpleNamespace(name="Klaus", scratch=scratch)
    retrieved = {"events": [], "thoughts": []}
    self.assertEqual("yes", invoke(
      "Answer in yes or no: yes", lambda:
        run_gpt_prompt.run_gpt_prompt_decide_to_talk(
          persona, target, retrieved)[0]))
    self.assertEqual("2", invoke(
      "Answer: Option2", lambda:
        run_gpt_prompt.run_gpt_prompt_decide_to_react(
          persona, target, retrieved)[0]))
    self.assertEqual({"Insight": [1, 2]}, invoke(
      "Insight (because of 1, 2)", lambda:
        run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
          object(), "statements", 1)[0]))

  def test_66_absent_call_time_context_is_not_invented_later(self):
    self.execute(FakeCompletionCompatAdapter(normalized_response()))
    event = get_telemetry()[0]
    with use_cost_ledger_context(CostLedgerContext(
        cognitive_category="LATE", actor_id="late-actor",
        simulation_id="late-simulation", simulation_step=9)):
      record = build_cost_ledger_records((event,))[0]
    self.assertEqual((None, None, None, None), (
      record.cognitive_category, record.actor_id,
      record.simulation_id, record.simulation_step))
    for changes in (
        {"actor_id": ""}, {"actor_id": "   "},
        {"simulation_id": 1}, {"cognitive_category": []},
        {"simulation_step": True}):
      with self.subTest(changes=changes):
        with self.assertRaises((TypeError, ValueError)):
          LLMReplayContext(**changes)

  def test_67_planning_thought_valid_output_matches_legacy_contract(self):
    persona = SimpleNamespace(scratch=SimpleNamespace(name="Maria Lopez"))
    legacy = FakeProvider()
    legacy.queue_completion_response("Remember the appointment")
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="exact prompt"), (
        use_provider(legacy)):
      historical = run_gpt_prompt.run_gpt_prompt_planning_thought_on_convo(
        persona, "conversation")

    adapter, unused_endpoint = modern_sdk_adapter(
      sdk_chat_response("Remember the appointment"))
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="exact prompt"), (
        use_modern_completion_runtime(
          build_modern_completion_runtime_config(), adapter)):
      modern = run_gpt_prompt.run_gpt_prompt_planning_thought_on_convo(
        persona, "conversation")

    self.assertEqual("Remember the appointment", historical[0])
    self.assertEqual(historical[0], modern[0])
    self.assertIsInstance(modern[0], str)
    self.assertEqual("...", modern[1][-1])

  def test_68_planning_length_output_is_accepted_only_in_completion_compat(self):
    partial = "Remember the appointment because it affects tomorrow"
    adapter, endpoint = modern_sdk_adapter(sdk_chat_response(
      partial, finish_reason="length", input_tokens=857, output_tokens=50))
    persona = SimpleNamespace(scratch=SimpleNamespace(name="Maria Lopez"))
    context = LLMReplayContext(
      actor_id="Maria Lopez", simulation_id="fixture", simulation_step=107)
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="exact prompt"), (
        use_llm_replay_context(context)), use_modern_completion_runtime(
          build_modern_completion_runtime_config(), adapter):
      output = run_gpt_prompt.run_gpt_prompt_planning_thought_on_convo(
        persona, "conversation")[0]

    self.assertEqual(partial, output)
    self.assertEqual({
      "model": COMPLETION_COMPAT_MODEL,
      "messages": [{"role": "user", "content": "exact prompt"}],
      "store": False,
      "temperature": 0,
      "max_tokens": 50,
      "top_p": 1,
      "frequency_penalty": 0,
      "presence_penalty": 0,
    }, endpoint.calls[0])
    event = get_telemetry()[0]
    self.assertEqual((
      COMPLETION_COMPAT, SUCCESS, "planning_thought_on_convo",
      "Maria Lopez", 107, "length", 857, 50), (
        event.operation, event.outcome, event.caller_id, event.actor_id,
        event.simulation_step, event.finish_reason,
        event.input_tokens, event.output_tokens))

  def test_69_native_chat_length_policy_remains_fail_closed(self):
    adapter, unused_endpoint = modern_sdk_adapter(
      sdk_chat_response("partial", finish_reason="length"))
    with self.assertRaises(LLMIncompleteResponseError):
      adapter.create_chat(model=COMPLETION_COMPAT_MODEL, messages=[])

  def test_70_completion_compat_does_not_accept_other_incomplete_states(self):
    fixtures = (
      sdk_chat_response("filtered", finish_reason="content_filter"),
      sdk_chat_response("partial", finish_reason="length",
                        status="incomplete"),
    )
    for response in fixtures:
      with self.subTest(response=response):
        clear_telemetry()
        adapter, unused_endpoint = modern_sdk_adapter(response)
        with self.assertRaises(LLMIncompleteResponseError):
          self.execute(adapter)

  def test_71_completion_compat_hard_errors_do_not_become_fail_safe(self):
    for error in (
        LLMAuthenticationError("auth"), LLMRefusalError("policy")):
      with self.subTest(error=type(error).__name__):
        clear_telemetry()
        with self.assertRaises(type(error)):
          self.execute(FakeCompletionCompatAdapter(error))

  def test_72_planning_stop_and_budget_match_historical_transport(self):
    adapter, endpoint = modern_sdk_adapter(sdk_chat_response())
    persona = SimpleNamespace(scratch=SimpleNamespace(name="Maria Lopez"))
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="prompt"), (
        use_modern_completion_runtime(
          build_modern_completion_runtime_config(), adapter)):
      run_gpt_prompt.run_gpt_prompt_planning_thought_on_convo(
        persona, "conversation")
    self.assertEqual(50, endpoint.calls[0]["max_tokens"])
    self.assertNotIn("stop", endpoint.calls[0])
    self.assertNotIn("stream", endpoint.calls[0])


class PostConversationEmbeddingAttributionTests(unittest.TestCase):
  """R1EMB-P1: the post-conversation embedding call reflect.py issues right
  after planning_thought_on_convo -> event_triple -> event_poignancy must
  reach the modern runtime with explicit, policy-authorized attribution
  instead of caller=None (live evidence: r1cli-a2-b-process-b3, tick 107)."""

  def setUp(self):
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_cache()
    clear_telemetry()
    self.sleep_patch = patch.object(gpt_structure, "temp_sleep")
    self.sleep_patch.start()

  def tearDown(self):
    self.sleep_patch.stop()
    reset_provider()
    reset_llm_provider_config()
    reset_embedding_cache()
    clear_telemetry()

  def _persona(self, name="Maria Lopez"):
    return SimpleNamespace(
      name=name,
      scratch=SimpleNamespace(
        name=name,
        get_str_iss=lambda: f"{name} is a resident of Smallville."))

  def _run_post_conversation_chain(self, persona, context=None,
                                   embedding_adapter=None):
    """Exercise the exact chain from the audit: generate_planning_thought_on_convo
    -> post-conversation memory construction -> event_triple -> event_poignancy
    -> embedding, using the real reflect.py/gpt_structure.py functions against
    fake transports.  Stops where reflect.py's post-conversation block does
    its first embedding write, matching the scope of the audited defect."""
    compat_adapter = FakeCompletionCompatAdapter(
      normalized_response("Remember the appointment"),
      normalized_response("planning, thinking)"))
    chat_adapter = FakeCompletionCompatAdapter(
      normalized_response('{"output": "7"}'))
    embedding_adapter = embedding_adapter or FakeEmbeddingAdapter()
    context = context or LLMReplayContext(
      cognitive_category="WORLD_TICK", actor_id=persona.name,
      simulation_id="r1emb-p1-offline", simulation_step=107)
    all_utt = f"{persona.name}: Hi\nKlaus Mueller: Hi\n"
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="exact prompt"), (
        use_llm_replay_context(context)), use_modern_completion_runtime(
        build_modern_completion_runtime_config(), compat_adapter), (
        use_modern_chat_runtime(
          build_modern_chat_runtime_config(), chat_adapter)), (
        use_embedding_runtime(
          build_modern_embedding_runtime_config(), embedding_adapter)):
      planning_thought = reflect_module.generate_planning_thought_on_convo(
        persona, all_utt)
      planning_thought = f"For {persona.scratch.name}'s planning: {planning_thought}"
      triple = reflect_module.generate_action_event_triple(
        planning_thought, persona)
      thought_poignancy = reflect_module.generate_poig_score(
        persona, "thought", planning_thought)
      vector = gpt_structure.get_embedding(
        planning_thought, caller_id="planning_thought_on_convo")
    return planning_thought, triple, thought_poignancy, vector

  def _embedding_event(self):
    matches = [event for event in get_telemetry()
              if event.operation == EMBEDDING]
    self.assertEqual(1, len(matches))
    return matches[0]

  def test_r1emb_p1_a_post_conversation_embedding_is_attributed(self):
    persona = self._persona()
    _, _, _, vector = self._run_post_conversation_chain(persona)

    self.assertEqual(1536, len(vector))
    event = self._embedding_event()
    self.assertIsNotNone(event.caller_id)
    self.assertEqual("Maria Lopez", event.actor_id)
    self.assertEqual(EMBEDDING, event.operation)
    self.assertEqual(SUCCESS, event.outcome)

  def test_r1emb_p1_b_semantic_owner_is_the_exact_caller(self):
    """The embedded text is planning_thought_on_convo's own output, so that
    is the correct semantic owner -- not the intervening event_triple or
    event_poignancy callers that merely ran afterward."""
    persona = self._persona()
    self._run_post_conversation_chain(persona)

    event = self._embedding_event()
    self.assertEqual("planning_thought_on_convo", event.caller_id)
    self.assertEqual(MEMORY_WRITE, event.cognitive_category)
    caller_sequence = [event.caller_id for event in get_telemetry()]
    self.assertEqual(
      ["planning_thought_on_convo", "event_triple", "event_poignancy",
       "planning_thought_on_convo"],
      caller_sequence)

  def test_r1emb_p1_c_anonymous_embedding_caller_is_rejected(self):
    with self.assertRaises(ModernChatCallerNotAllowedError):
      validate_modern_embedding_caller(None)

  def test_r1emb_p1_c2_anonymous_caller_fails_closed_through_the_real_seam(self):
    """caller=None must still fail once inside an active modern runtime --
    the guard being added here must not be bypassable at the call site."""
    with use_modern_chat_runtime(
        build_modern_chat_runtime_config(), FakeCompletionCompatAdapter()):
      with self.assertRaises(ModernChatCallerNotAllowedError):
        with gpt_structure.use_modern_embedding_caller_if_active(None):
          pass

  def test_r1emb_p1_d_unknown_embedding_caller_is_rejected(self):
    self.assertNotIn("unknown", M2_EMBEDDING_CALLER_ALLOWLIST)
    with self.assertRaises(ModernChatCallerNotAllowedError):
      validate_modern_embedding_caller("unknown")
    with use_modern_chat_runtime(
        build_modern_chat_runtime_config(), FakeCompletionCompatAdapter()):
      with self.assertRaises(ModernChatCallerNotAllowedError):
        with gpt_structure.use_modern_embedding_caller_if_active("unknown"):
          pass

  def test_r1emb_p1_e_unmigrated_embedding_call_sites_are_unaffected(self):
    """perceive.py/retrieve.py/plan.py/converse.py all call get_embedding
    without a caller_id; that is out of this wave's scope and must keep
    working exactly as before, even while the modern runtimes are active."""
    embedding_adapter = FakeEmbeddingAdapter()
    with use_modern_chat_runtime(
        build_modern_chat_runtime_config(), FakeCompletionCompatAdapter()):
      with use_embedding_runtime(
          build_modern_embedding_runtime_config(), embedding_adapter):
        vector = gpt_structure.get_embedding("unattributed perception text")
    self.assertEqual(1536, len(vector))
    event = self._embedding_event()
    self.assertIsNone(event.caller_id)
    self.assertEqual(SUCCESS, event.outcome)

  def test_r1emb_p1_f_actor_attribution_follows_the_ambient_replay_context(self):
    persona = self._persona("Klaus Mueller")
    context = LLMReplayContext(
      cognitive_category="WORLD_TICK", actor_id="Klaus Mueller",
      simulation_id="r1emb-p1-offline", simulation_step=42)
    self._run_post_conversation_chain(persona, context=context)

    event = self._embedding_event()
    self.assertEqual("Klaus Mueller", event.actor_id)
    self.assertEqual(42, event.simulation_step)
    self.assertEqual("r1emb-p1-offline", event.simulation_id)

  def test_r1emb_p1_g_cost_ledger_records_full_attribution(self):
    persona = self._persona()
    self._run_post_conversation_chain(persona)

    event = self._embedding_event()
    record = build_cost_ledger_records((event,))[0]
    self.assertEqual("planning_thought_on_convo", record.caller_id)
    self.assertEqual("Maria Lopez", record.actor_id)
    self.assertEqual(EMBEDDING, record.operation)
    self.assertEqual(MEMORY_WRITE, record.cognitive_category)

  def test_r1emb_p1_h_get_embedding_signature_stays_backward_compatible(self):
    """caller_id is additive and keyword-only; every existing positional
    call to get_embedding(text) or get_embedding(text, model) is unaffected."""
    import inspect
    parameters = inspect.signature(gpt_structure.get_embedding).parameters
    self.assertEqual(["text", "model", "caller_id"], list(parameters))
    self.assertIsNone(parameters["caller_id"].default)
    self.assertEqual(
      inspect.Parameter.KEYWORD_ONLY, parameters["caller_id"].kind)


if __name__ == "__main__":
  unittest.main()
