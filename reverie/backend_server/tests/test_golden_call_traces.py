import datetime
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from golden_trace import build_golden_trace
from persona.cognitive_modules import converse, retrieve
from persona.prompt_template import gpt_structure, run_gpt_prompt
from persona.prompt_template.llm_provider import (
  CHAT,
  COMPLETION,
  EMBEDDING,
  ERROR,
  HIT,
  MISS,
  CONVERSATION,
  RETRIEVAL,
  SUCCESS,
  FakeProvider,
  clear_telemetry,
  get_embedding_cache_stats,
  get_embedding_logical_events,
  get_telemetry,
  reset_embedding_cache,
  set_embedding_cache_enabled,
  use_provider,
)


def _scratch(name, action):
  return SimpleNamespace(
    name=name,
    act_description=action,
    curr_time=datetime.datetime(2023, 1, 2, 9, 0, 0),
    curr_tile=(1, 1),
    recency_decay=0.99,
    recency_w=1,
    relevance_w=1,
    importance_w=1,
    get_str_iss=lambda: f"Identity summary for {name}",
  )


def _node(node_id, description, embedding_key, last_accessed, vector):
  return SimpleNamespace(
    node_id=node_id,
    description=description,
    embedding_key=embedding_key,
    last_accessed=last_accessed,
    poignancy=5,
    vector=vector,
  )


def _persona(name, action, nodes):
  embeddings = {node.embedding_key: node.vector for node in nodes}
  memory = SimpleNamespace(
    seq_event=list(nodes),
    seq_thought=[],
    seq_chat=[],
    embeddings=embeddings,
    id_to_node={node.node_id: node for node in nodes},
  )
  return SimpleNamespace(name=name, scratch=_scratch(name, action), a_mem=memory)


class GoldenCallTraceTests(unittest.TestCase):
  def setUp(self):
    clear_telemetry()
    reset_embedding_cache()
    self.fake = FakeProvider()
    self.provider_context = use_provider(self.fake)
    self.provider_context.__enter__()
    self.previous_cwd = Path.cwd()
    os.chdir(BACKEND_SERVER)
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
    os.chdir(self.previous_cwd)
    self.provider_context.__exit__(None, None, None)
    clear_telemetry()
    reset_embedding_cache()

  def trace(self):
    return build_golden_trace(get_telemetry(), self.fake.calls)

  def test_golden_chat_wrapper_success(self):
    self.fake.queue_chat_response('{"output": "processed"}')

    output = gpt_structure.ChatGPT_safe_generate_response(
      "sensitive success prompt", "example", "return a string", repeat=3,
      fail_safe_response="fallback",
      func_validate=lambda response, prompt="": response == "processed",
      func_clean_up=lambda response, prompt="": response.upper())

    self.assertEqual("PROCESSED", output)
    trace = self.trace()
    self.assertEqual([{
      "logical_index": 1,
      "operation": CHAT,
      "physical_attempts": 1,
      "model": "gpt-3.5-turbo",
      "outcomes": [SUCCESS],
      "attempt_numbers": [1],
      "input_fingerprints": [trace[0]["input_fingerprints"][0]],
      "essential_arguments": {
        "message_count": 1,
        "message_roles": ["user"],
      },
    }], trace)
    self.assertEqual("user", self.fake.calls[0].arguments["messages"][0]["role"])
    self.assertIn("sensitive success prompt",
                  self.fake.calls[0].arguments["messages"][0]["content"])
    self.assertNotIn("sensitive success prompt", repr(trace))

  def test_golden_chat_wrapper_two_errors_then_success(self):
    self.fake.queue_error(CHAT, RuntimeError("first failure"))
    self.fake.queue_error(CHAT, ValueError("second failure"))
    self.fake.queue_chat_response('{"output": "valid"}')

    with redirect_stdout(io.StringIO()):
      output = gpt_structure.ChatGPT_safe_generate_response(
        "same retry prompt", "example", "return a string", repeat=3,
        fail_safe_response="fallback",
        func_validate=lambda response, prompt="": response == "valid",
        func_clean_up=lambda response, prompt="": response)

    self.assertEqual("valid", output)
    trace = self.trace()
    self.assertEqual(1, len(trace))
    self.assertEqual(CHAT, trace[0]["operation"])
    self.assertEqual(3, trace[0]["physical_attempts"])
    self.assertEqual([ERROR, ERROR, SUCCESS], trace[0]["outcomes"])
    self.assertEqual([1, 2, 3], trace[0]["attempt_numbers"])
    self.assertEqual(1, len(set(trace[0]["input_fingerprints"])))
    self.assertEqual(self.fake.calls[0].arguments,
                     self.fake.calls[1].arguments)
    self.assertEqual(self.fake.calls[1].arguments,
                     self.fake.calls[2].arguments)

  def test_golden_completion_wrapper_max_retry_then_legacy_fallback(self):
    for attempt in range(5):
      self.fake.queue_error(COMPLETION, RuntimeError(f"failure {attempt}"))
    parameters = {
      "engine": "text-davinci-003",
      "temperature": 0.5,
      "max_tokens": 30,
      "top_p": 0.9,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.2,
      "stream": False,
      "stop": ["\n"],
    }

    with redirect_stdout(io.StringIO()):
      output = gpt_structure.safe_generate_response(
        "completion retry prompt", parameters, repeat=5,
        fail_safe_response="legacy fallback",
        func_validate=lambda response, prompt="": False,
        func_clean_up=lambda response, prompt="": response)

    self.assertEqual("legacy fallback", output)
    trace = self.trace()
    self.assertEqual(1, len(trace))
    self.assertEqual(COMPLETION, trace[0]["operation"])
    self.assertEqual("text-davinci-003", trace[0]["model"])
    self.assertEqual(5, trace[0]["physical_attempts"])
    self.assertEqual([ERROR] * 5, trace[0]["outcomes"])
    self.assertEqual({
      "temperature": 0.5,
      "max_tokens": 30,
      "top_p": 0.9,
      "frequency_penalty": 0.1,
      "presence_penalty": 0.2,
      "stream": False,
      "stop": ["\n"],
    }, trace[0]["essential_arguments"])
    self.assertEqual(1, len(set(trace[0]["input_fingerprints"])))

  def test_golden_embedding_normalization_and_duplicate_physical_calls(self):
    for vector in ([1.0], [2.0], [3.0]):
      self.fake.queue_embedding_response(vector)

    outputs = [
      gpt_structure.get_embedding("same text"),
      gpt_structure.get_embedding("line one\nline two"),
      gpt_structure.get_embedding(""),
      gpt_structure.get_embedding("same text"),
    ]

    self.assertEqual([[1.0], [2.0], [3.0], [1.0]], outputs)
    self.assertEqual(["same text"], self.fake.calls[0].arguments["input"])
    self.assertEqual(["line one line two"], self.fake.calls[1].arguments["input"])
    self.assertEqual(["this is blank"], self.fake.calls[2].arguments["input"])
    trace = self.trace()
    self.assertEqual(3, len(trace))
    self.assertEqual([EMBEDDING] * 3,
                     [item["operation"] for item in trace])
    self.assertEqual([1] * 3,
                     [item["physical_attempts"] for item in trace])
    logical_events = get_embedding_logical_events()
    self.assertEqual([MISS, MISS, MISS, HIT],
                     [event.cache_outcome for event in logical_events])
    self.assertEqual(logical_events[0].cache_key_fingerprint,
                     logical_events[3].cache_key_fingerprint)
    stats = get_embedding_cache_stats()
    self.assertEqual(4, stats.logical_embedding_requests)
    self.assertEqual(3, stats.physical_embedding_attempts)
    self.assertEqual(1, stats.cache_hits)
    self.assertEqual(3, stats.cache_misses)

  def test_golden_importance_poignancy_parsing_and_legacy_failure(self):
    persona = SimpleNamespace(scratch=_scratch("Alice", "reading"))
    self.fake.queue_chat_response('{"output": "7"}')

    with redirect_stdout(io.StringIO()):
      output, metadata = run_gpt_prompt.run_gpt_prompt_event_poignancy(
        persona, "Alice reads a letter")

    self.assertEqual(7, output)
    self.assertEqual(4, metadata[-1])
    self.assertEqual(CHAT, self.trace()[0]["operation"])
    self.assertEqual("gpt-3.5-turbo", self.trace()[0]["model"])

    clear_telemetry()
    self.fake.calls.clear()
    for invalid in ("invalid", "still invalid", "not numeric"):
      self.fake.queue_chat_response(invalid)
    with redirect_stdout(io.StringIO()):
      failed_output = run_gpt_prompt.run_gpt_prompt_event_poignancy(
        persona, "Alice reads a letter")

    self.assertIsNone(failed_output)
    failure_trace = self.trace()
    self.assertEqual(1, len(failure_trace))
    self.assertEqual(3, failure_trace[0]["physical_attempts"])
    self.assertEqual([SUCCESS, SUCCESS, SUCCESS],
                     failure_trace[0]["outcomes"])

  def test_golden_event_triple_wrapper_parsing(self):
    persona = SimpleNamespace(name="Alice")
    self.fake.queue_completion_response("is reading, a book)")

    with redirect_stdout(io.StringIO()):
      output, metadata = run_gpt_prompt.run_gpt_prompt_event_triple(
        "quietly reading a book", persona)

    self.assertEqual(("Alice", "is reading", "a book"), output)
    self.assertEqual("text-davinci-003", metadata[2]["engine"])
    self.assertEqual(["Alice", "quietly reading a book", "Alice"], metadata[3])
    trace = self.trace()
    self.assertEqual(1, len(trace))
    self.assertEqual(COMPLETION, trace[0]["operation"])
    self.assertEqual("text-davinci-003", trace[0]["model"])
    self.assertEqual(1, trace[0]["physical_attempts"])
    self.assertIn("quietly reading a book",
                  self.fake.calls[0].arguments["prompt"])

  def test_golden_new_retrieve_embedding_order_ranking_and_last_accessed(self):
    old_time = datetime.datetime(2023, 1, 1, 8, 0, 0)
    nodes = [
      _node("node_1", "red memory", "red memory", old_time, [1.0, 0.0]),
      _node("node_2", "blue memory", "blue memory",
            old_time + datetime.timedelta(minutes=1), [0.0, 1.0]),
    ]
    persona = _persona("Alice", "reading", nodes)
    self.fake.queue_embedding_response([1.0, 0.0])
    self.fake.queue_embedding_response([0.0, 1.0])

    with redirect_stdout(io.StringIO()):
      result = retrieve.new_retrieve(persona, ["red focus", "blue focus"], 1)

    self.assertEqual("node_1", result["red focus"][0].node_id)
    self.assertEqual("node_2", result["blue focus"][0].node_id)
    self.assertEqual(persona.scratch.curr_time, nodes[0].last_accessed)
    self.assertEqual(persona.scratch.curr_time, nodes[1].last_accessed)
    self.assertEqual(["red focus"], self.fake.calls[0].arguments["input"])
    self.assertEqual(["blue focus"], self.fake.calls[1].arguments["input"])
    trace = self.trace()
    self.assertEqual([EMBEDDING, EMBEDDING],
                     [item["operation"] for item in trace])
    self.assertEqual([1, 1], [item["physical_attempts"] for item in trace])
    self.assertNotEqual(trace[0]["input_fingerprints"],
                        trace[1]["input_fingerprints"])
    self.assertEqual([RETRIEVAL, RETRIEVAL],
                     [event.category
                      for event in get_embedding_logical_events()])

  def test_golden_new_retrieve_repeated_focal_preserves_ranking_with_cache(self):
    old_time = datetime.datetime(2023, 1, 1, 8, 0, 0)

    baseline_node = _node(
      "node_1", "red memory", "red memory", old_time, [1.0, 0.0])
    baseline_persona = _persona("Alice", "reading", [baseline_node])
    set_embedding_cache_enabled(False)
    self.fake.queue_embedding_response([1.0, 0.0])
    self.fake.queue_embedding_response([1.0, 0.0])
    with redirect_stdout(io.StringIO()):
      baseline = retrieve.new_retrieve(
        baseline_persona, ["red focus", "red focus"], 1)
    baseline_physical = len(self.fake.calls)
    baseline_logical = get_embedding_cache_stats().logical_embedding_requests

    clear_telemetry()
    reset_embedding_cache()
    self.fake.calls.clear()
    cached_node = _node(
      "node_1", "red memory", "red memory", old_time, [1.0, 0.0])
    cached_persona = _persona("Alice", "reading", [cached_node])
    self.fake.queue_embedding_response([1.0, 0.0])
    with redirect_stdout(io.StringIO()):
      cached = retrieve.new_retrieve(
        cached_persona, ["red focus", "red focus"], 1)

    self.assertEqual("node_1", baseline["red focus"][0].node_id)
    self.assertEqual("node_1", cached["red focus"][0].node_id)
    self.assertEqual(baseline_persona.scratch.curr_time,
                     baseline_node.last_accessed)
    self.assertEqual(cached_persona.scratch.curr_time,
                     cached_node.last_accessed)
    self.assertEqual(2, baseline_logical)
    self.assertEqual(2, baseline_physical)
    cached_stats = get_embedding_cache_stats()
    self.assertEqual(2, cached_stats.logical_embedding_requests)
    self.assertEqual(1, cached_stats.physical_embedding_attempts)
    self.assertEqual(1, cached_stats.cache_hits)
    self.assertEqual([MISS, HIT],
                     [event.cache_outcome
                      for event in get_embedding_logical_events()])
    self.assertEqual([RETRIEVAL, RETRIEVAL],
                     [event.category
                      for event in get_embedding_logical_events()])

  def test_golden_agent_chat_v2_single_half_turn_sequence(self):
    old_time = datetime.datetime(2023, 1, 1, 8, 0, 0)
    init_node = _node("node_1", "Alice knows Bob", "Alice knows Bob",
                      old_time, [1.0, 0.0])
    init_persona = _persona("Alice", "reading", [init_node])
    target_persona = _persona("Bob", "making tea", [])
    maze = SimpleNamespace(access_tile=lambda tile: {
      "sector": "home", "arena": "kitchen"})
    self.fake.queue_embedding_response([1.0, 0.0])
    self.fake.queue_chat_response('{"output": "Alice trusts Bob"}')
    self.fake.queue_embedding_response([1.0, 0.0])
    self.fake.queue_embedding_response([1.0, 0.0])
    self.fake.queue_chat_response(
      '{"utterance": "Hello Bob", "end": true}')

    with redirect_stdout(io.StringIO()):
      chat = converse.agent_chat_v2(maze, init_persona, target_persona)

    self.assertEqual([["Alice", "Hello Bob"]], chat)
    trace = self.trace()
    self.assertEqual(
      [EMBEDDING, CHAT, EMBEDDING, EMBEDDING, CHAT],
      [item["operation"] for item in trace])
    self.assertEqual(
      ["text-embedding-ada-002", "gpt-3.5-turbo",
       "text-embedding-ada-002", "text-embedding-ada-002",
       "gpt-3.5-turbo"],
      [item["model"] for item in trace])
    self.assertEqual([1, 1, 1, 1, 1],
                     [item["physical_attempts"] for item in trace])
    self.assertEqual(["Bob"], self.fake.calls[0].arguments["input"])
    self.assertEqual(["Alice trusts Bob"],
                     self.fake.calls[2].arguments["input"])
    self.assertEqual(["Bob is making tea"],
                     self.fake.calls[3].arguments["input"])
    self.assertEqual(init_persona.scratch.curr_time,
                     init_node.last_accessed)
    self.assertNotIn("Alice trusts Bob", repr(trace))
    cache_stats = get_embedding_cache_stats()
    self.assertEqual(3, cache_stats.logical_embedding_requests)
    self.assertEqual(3, cache_stats.physical_embedding_attempts)
    self.assertEqual(0, cache_stats.cache_hits)
    self.assertEqual(3, cache_stats.cache_misses)
    self.assertEqual([CONVERSATION, CONVERSATION, CONVERSATION],
                     [event.category
                      for event in get_embedding_logical_events()])


if __name__ == "__main__":
  unittest.main()
