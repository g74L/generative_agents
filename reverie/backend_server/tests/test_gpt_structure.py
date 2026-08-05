import inspect
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template import gpt_structure
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION,
  EMBEDDING,
  ERROR,
  FakeProvider,
  clear_telemetry,
  get_telemetry,
  reset_embedding_cache,
  use_provider,
)


class GPTStructureProviderTests(unittest.TestCase):
  def setUp(self):
    clear_telemetry()
    reset_embedding_cache()
    self.fake = FakeProvider()
    self.provider_context = use_provider(self.fake)
    self.provider_context.__enter__()
    self.sleep_patch = patch.object(gpt_structure, "temp_sleep")
    self.sleep_patch.start()
    self.network_patches = [
      patch.object(gpt_structure.openai.ChatCompletion, "create",
                   side_effect=AssertionError("network chat transport used")),
      patch.object(gpt_structure.openai.Completion, "create",
                   side_effect=AssertionError("network completion transport used")),
      patch.object(gpt_structure.openai.Embedding, "create",
                   side_effect=AssertionError("network embedding transport used")),
    ]
    for network_patch in self.network_patches:
      network_patch.start()

  def tearDown(self):
    for network_patch in reversed(self.network_patches):
      network_patch.stop()
    self.sleep_patch.stop()
    self.provider_context.__exit__(None, None, None)
    clear_telemetry()
    reset_embedding_cache()

  def test_chatgpt_request_uses_injected_provider(self):
    self.fake.queue_chat_response("expected content")

    result = gpt_structure.ChatGPT_request("private prompt")

    self.assertEqual("expected content", result)
    self.assertEqual(1, len(self.fake.calls))
    self.assertEqual(CHAT, self.fake.calls[0].operation)
    self.assertEqual({
      "model": "gpt-3.5-turbo",
      "messages": [{"role": "user", "content": "private prompt"}],
    }, self.fake.calls[0].arguments)

  def test_gpt_request_forwards_every_legacy_parameter(self):
    self.fake.queue_completion_response(" completion text ")
    parameters = {
      "engine": "text-davinci-test",
      "temperature": 0.25,
      "max_tokens": 17,
      "top_p": 0.8,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.2,
      "stream": False,
      "stop": ["END"],
    }

    result = gpt_structure.GPT_request("exact prompt", parameters)

    self.assertEqual(" completion text ", result)
    self.assertEqual(COMPLETION, self.fake.calls[0].operation)
    self.assertEqual({
      "model": "text-davinci-test",
      "prompt": "exact prompt",
      "temperature": 0.25,
      "max_tokens": 17,
      "top_p": 0.8,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.2,
      "stream": False,
      "stop": ["END"],
    }, self.fake.calls[0].arguments)

  def test_get_embedding_preserves_baseline_normalization(self):
    self.fake.queue_embedding_response([0.1, 0.2])
    self.fake.queue_embedding_response([0.3])

    newline_result = gpt_structure.get_embedding("first\nsecond")
    blank_result = gpt_structure.get_embedding("")

    self.assertEqual([0.1, 0.2], newline_result)
    self.assertEqual([0.3], blank_result)
    self.assertEqual(["first second"], self.fake.calls[0].arguments["input"])
    self.assertEqual(["this is blank"], self.fake.calls[1].arguments["input"])
    self.assertEqual("text-embedding-ada-002",
                     self.fake.calls[0].arguments["model"])

  def test_existing_retry_is_one_logical_call_with_multiple_attempts(self):
    self.fake.queue_completion_response("invalid one")
    self.fake.queue_completion_response("invalid two")
    parameters = {
      "engine": "text-davinci-003",
      "temperature": 0,
      "max_tokens": 5,
      "top_p": 1,
      "frequency_penalty": 0,
      "presence_penalty": 0,
      "stream": False,
      "stop": None,
    }

    result = gpt_structure.safe_generate_response(
      "retry prompt", parameters, repeat=2, fail_safe_response="baseline fallback",
      func_validate=lambda response, prompt="": False,
      func_clean_up=lambda response, prompt="": response)

    self.assertEqual("baseline fallback", result)
    self.assertEqual([COMPLETION, COMPLETION],
                     [call.operation for call in self.fake.calls])
    self.assertEqual(self.fake.calls[0].arguments, self.fake.calls[1].arguments)
    events = get_telemetry()
    self.assertEqual(2, len(events))
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual([1, 2], [event.physical_attempt for event in events])
    self.assertEqual(events[0].input_fingerprint, events[1].input_fingerprint)

  def test_errors_keep_legacy_conversion_retry_and_fail_safe(self):
    self.fake.queue_error(COMPLETION, RuntimeError("first"))
    self.fake.queue_error(COMPLETION, ValueError("second"))
    parameters = {
      "engine": "text-davinci-003",
      "temperature": 0,
      "max_tokens": 5,
      "top_p": 1,
      "frequency_penalty": 0,
      "presence_penalty": 0,
      "stream": False,
      "stop": None,
    }

    result = gpt_structure.safe_generate_response(
      "retry prompt", parameters, repeat=2, fail_safe_response="baseline fallback",
      func_validate=lambda response, prompt="": False,
      func_clean_up=lambda response, prompt="": response)

    self.assertEqual("baseline fallback", result)
    events = get_telemetry()
    self.assertEqual([ERROR, ERROR], [event.outcome for event in events])
    self.assertEqual(["RuntimeError", "ValueError"],
                     [event.error_type for event in events])
    self.assertEqual(1, len({event.logical_call_id for event in events}))

  def test_chat_safe_generate_preserves_false_instead_of_declared_fail_safe(self):
    self.fake.queue_chat_response("not json")
    self.fake.queue_chat_response("still not json")

    result = gpt_structure.ChatGPT_safe_generate_response(
      "prompt", "example", "instruction", repeat=2,
      fail_safe_response="intentionally ignored",
      func_validate=lambda response, prompt="": True,
      func_clean_up=lambda response, prompt="": response)

    self.assertFalse(result)
    events = get_telemetry()
    self.assertEqual(2, len(events))
    self.assertEqual(1, len({event.logical_call_id for event in events}))

  def test_telemetry_contains_fingerprint_but_no_sensitive_input(self):
    secret_prompt = "sensitive conversation sk-test-secret"
    self.fake.queue_chat_response("ok")

    gpt_structure.ChatGPT_request(secret_prompt)

    event = get_telemetry()[0]
    event_text = repr(event)
    self.assertEqual(CHAT, event.operation)
    self.assertEqual(64, len(event.input_fingerprint))
    self.assertNotIn(secret_prompt, event_text)
    self.assertNotIn("sk-test-secret", event_text)
    self.assertFalse(hasattr(event, "prompt"))
    self.assertFalse(hasattr(event, "messages"))
    self.assertFalse(hasattr(event, "api_key"))

  def test_existing_public_function_signatures_remain_available(self):
    expected_parameters = {
      "ChatGPT_request": ["prompt"],
      "GPT4_request": ["prompt"],
      "GPT_request": ["prompt", "gpt_parameter", "caller_id"],
      "get_embedding": ["text", "model"],
      "safe_generate_response": ["prompt", "gpt_parameter", "repeat",
                                  "fail_safe_response", "func_validate",
                                  "func_clean_up", "verbose", "caller_id"],
      "ChatGPT_safe_generate_response": ["prompt", "example_output",
        "special_instruction", "repeat", "fail_safe_response",
        "func_validate", "func_clean_up", "verbose"],
      "GPT4_safe_generate_response": ["prompt", "example_output",
        "special_instruction", "repeat", "fail_safe_response",
        "func_validate", "func_clean_up", "verbose"],
      "ChatGPT_single_request": ["prompt"],
    }

    for function_name, parameter_names in expected_parameters.items():
      function = getattr(gpt_structure, function_name)
      self.assertEqual(parameter_names,
                       list(inspect.signature(function).parameters),
                       function_name)


if __name__ == "__main__":
  unittest.main()
