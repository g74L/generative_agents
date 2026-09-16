"""Offline contract tests for actor-local OCE social interpretation."""
import copy
from contextlib import ExitStack
from dataclasses import asdict, replace
import datetime
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

from controlled_replay import DeterministicReplayFakeAdapter
from persona.persona import Persona
from persona.cognitive_modules import retrieve
from persona.cognitive_modules.conversation_contract import (
  ConversationResult,
  ConversationTermination,
)
from persona.cognitive_modules.social_consequence import (
  commit_social_conversation_experience,
  experience_keyword,
)
from persona.cognitive_modules import social_interpretation as contract
from persona.prompt_template import modern_openai_provider as errors
from persona.prompt_template.chat_runtime import (
  build_modern_chat_runtime_config,
  use_modern_chat_runtime,
)
from persona.prompt_template.llm_provider import (
  REFLECTION,
  clear_telemetry,
  get_telemetry,
)


STORAGE = BACKEND.parents[1] / "environment/frontend_server/storage"
CONTROL = STORAGE / "base_the_ville_n25"
TOM, SAM = "Tom Moreno", "Sam Moore"
NOW = datetime.datetime(2023, 2, 13, 13, 5)
TRANSCRIPT = (
  (TOM, "Will you return next week with concrete ideas?"),
  (SAM, "Yes, I will return next week to discuss them."),
)
INTERPRETATION = "I should judge Sam by whether he actually returns with concrete ideas."
VALID_OUTPUT = '{"interpretation":' + repr(INTERPRETATION).replace("'", '"') + '}'


def _empty_memory(memory):
  memory.id_to_node.clear()
  memory.seq_event.clear()
  memory.seq_thought.clear()
  memory.seq_chat.clear()
  memory.kw_to_event.clear()
  memory.kw_to_thought.clear()
  memory.kw_to_chat.clear()
  memory.kw_strength_event.clear()
  memory.kw_strength_thought.clear()
  memory.embeddings.clear()


def _memory_snapshot(memory):
  def node_value(node):
    return (
      node.node_id, node.node_count, node.type_count, node.type, node.depth,
      node.created, node.expiration, node.last_accessed, node.subject,
      node.predicate, node.object, node.description, node.embedding_key,
      node.poignancy, tuple(sorted(node.keywords)), copy.deepcopy(node.filling),
    )
  return {
    "nodes": tuple(node_value(memory.id_to_node[key])
                   for key in sorted(memory.id_to_node)),
    "events": tuple(node.node_id for node in memory.seq_event),
    "thoughts": tuple(node.node_id for node in memory.seq_thought),
    "chats": tuple(node.node_id for node in memory.seq_chat),
    "embeddings": copy.deepcopy(memory.embeddings),
    "event_strength": copy.deepcopy(memory.kw_strength_event),
    "thought_strength": copy.deepcopy(memory.kw_strength_thought),
  }


class SocialInterpretationContractTests(unittest.TestCase):
  def setUp(self):
    self.stack = ExitStack()
    self.addCleanup(self.stack.close)
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    clear_telemetry()
    self.addCleanup(clear_telemetry)
    self.network = []

    def reject_network(*args, **kwargs):
      del args, kwargs
      self.network.append("attempt")
      raise AssertionError("network forbidden")

    for owner, name in ((socket, "create_connection"),
                        (socket.socket, "connect"),
                        (socket.socket, "connect_ex")):
      self.stack.enter_context(patch.object(owner, name, reject_network))
    self.live_guards = [self.stack.enter_context(patch.object(
      errors.ModernOpenAIClientAdapter, name,
      side_effect=AssertionError("live provider forbidden")))
      for name in ("create_chat", "create_embedding")]

    self.tom = Persona(TOM, str(CONTROL / "personas" / TOM))
    self.sam = Persona(SAM, str(CONTROL / "personas" / SAM))
    for actor in (self.tom, self.sam):
      _empty_memory(actor.a_mem)
      actor.scratch.curr_time = NOW
    conversation = ConversationResult(
      TRANSCRIPT, ConversationTermination.MODEL_END)
    factual = commit_social_conversation_experience(
      conversation, self.tom, self.sam, started_at=NOW,
      embedding_fn=lambda text: [0.1] * 1536)
    self.chat = factual.initiator_chat
    self.embedding_calls = []

  def tearDown(self):
    self.assertEqual([], self.network)
    for guard in self.live_guards:
      guard.assert_not_called()

  def embedding(self, text):
    self.embedding_calls.append(text)
    return [0.25] * 1536

  def execute(self, responses, actor=None, counterpart=SAM, chat=None):
    adapter = DeterministicReplayFakeAdapter(responses)
    with use_modern_chat_runtime(build_modern_chat_runtime_config(), adapter):
      result = contract.interpret_social_experience(
        actor or self.tom, counterpart, chat or self.chat,
        embedding_fn=self.embedding)
    return result, adapter

  def request(self):
    return contract.build_social_interpretation_request(
      self.tom, SAM, self.chat)

  def test_valid_oce_chat_builds_immutable_actor_local_request(self):
    request = self.request()
    self.assertEqual(TOM, request.actor_name)
    self.assertEqual(SAM, request.counterpart_name)
    self.assertEqual(NOW, request.experience_time)
    self.assertEqual(TRANSCRIPT, request.transcript)
    self.assertIn("Tom Moreno", request.actor_identity)
    self.assertEqual(self.chat.description, request.factual_description)
    with self.assertRaises(Exception):
      request.actor_name = "changed"

  def test_actor_identity_is_rendered_and_target_private_identity_is_excluded(self):
    sentinel = "PRIVATE_SAM_IDENTITY_SENTINEL_409"
    for field in ("innate", "learned", "currently"):
      setattr(self.sam.scratch, field, sentinel)
    with patch.object(self.sam.scratch, "get_str_iss",
                      side_effect=AssertionError("private target read")):
      request = self.request()
      prompt = contract.render_social_interpretation_prompt(request)
    self.assertIn(request.actor_identity, prompt)
    self.assertNotIn(sentinel, repr(asdict(request)))
    self.assertNotIn(sentinel, prompt)

  def test_prompt_marks_interpretation_as_belief_not_world_truth(self):
    prompt = contract.render_social_interpretation_prompt(self.request())
    for phrase in ("not authoritative world truth", "Do not invent events",
                   "turn uncertainty into fact", "relationship score"):
      self.assertIn(phrase, prompt)

  def test_strict_output_accepts_exact_single_nonempty_key(self):
    self.assertEqual(
      INTERPRETATION,
      contract.parse_social_interpretation_output(VALID_OUTPUT))

  def test_strict_output_rejects_wrappers_duplicates_and_extra_keys(self):
    invalid = (
      "not-json", "{}", "[]", "null",
      '{"interpretation":""}', '{"interpretation":"  "}',
      '{"interpretation":42}',
      '{"interpretation":"x","extra":true}',
      '{"interpretation":"x","interpretation":"y"}',
      "```json\n" + VALID_OUTPUT + "\n```", VALID_OUTPUT + " trailing",
      '{"interpretation":NaN}',
    )
    for value in invalid:
      with self.subTest(value=value):
        with self.assertRaises(contract.InvalidSocialInterpretationOutput):
          contract.parse_social_interpretation_output(value)

  def test_format_retry_keeps_identical_prompt_and_one_logical_call(self):
    result, adapter = self.execute(["invalid", VALID_OUTPUT])
    self.assertEqual(contract.SocialInterpretationCommitStatus.COMMITTED,
                     result.status)
    self.assertEqual(2, len(adapter.calls))
    self.assertEqual(adapter.calls[0][1], adapter.calls[1][1])
    events = get_telemetry()
    self.assertEqual([1, 2], [event.physical_attempt for event in events])
    self.assertEqual(1, len({event.logical_call_id for event in events}))
    self.assertEqual({"social_interpretation"},
                     {event.caller_id for event in events})
    self.assertEqual({REFLECTION},
                     {event.cognitive_category for event in events})

  def test_invalid_output_exhaustion_commits_nothing(self):
    result, adapter = self.execute(["invalid"] * 3)
    self.assertEqual(contract.SocialInterpretationStatus.INVALID_OUTPUT,
                     result.status)
    self.assertIsNone(result.interpretation)
    self.assertEqual(3, result.attempt_count)
    self.assertEqual(3, len(adapter.calls))
    self.assertEqual([], self.tom.a_mem.seq_thought)
    self.assertEqual([], self.embedding_calls)

  def test_provider_failure_taxonomy_has_no_fallback_or_commit(self):
    cases = (
      (errors.LLMRefusalError, contract.SocialInterpretationStatus.REFUSAL),
      (errors.LLMTimeoutError, contract.SocialInterpretationStatus.TIMEOUT),
      (errors.LLMConnectionError,
       contract.SocialInterpretationStatus.UNAVAILABLE),
      (errors.LLMRateLimitError,
       contract.SocialInterpretationStatus.UNAVAILABLE),
      (errors.LLMServerError,
       contract.SocialInterpretationStatus.UNAVAILABLE),
      (errors.LLMAuthenticationError,
       contract.SocialInterpretationStatus.PROVIDER_ERROR),
      (errors.LLMProviderError,
       contract.SocialInterpretationStatus.PROVIDER_ERROR),
    )
    for error_type, expected in cases:
      with self.subTest(error=error_type.__name__):
        result, adapter = self.execute([error_type("private provider text")])
        self.assertEqual(expected, result.status)
        self.assertIsNone(result.interpretation)
        self.assertNotIn("private provider text", repr(result))
        self.assertEqual(1, len(adapter.calls))
        self.assertEqual([], self.tom.a_mem.seq_thought)
        self.assertEqual([], self.embedding_calls)

  def test_inactive_runtime_is_provider_error(self):
    result = contract.generate_social_interpretation(self.request())
    self.assertEqual(contract.SocialInterpretationStatus.PROVIDER_ERROR,
                     result.status)
    self.assertEqual("ModernChatRuntimeInactiveError", result.reason)

  def test_invalid_context_is_reported_before_provider(self):
    request = self.request()
    invalid = (
      replace(request, actor_identity=""),
      replace(request, actor_name=""),
      replace(request, counterpart_name=""),
      replace(request, experience_id="bad"),
      replace(request, experience_time=None),
      replace(request, factual_description="invented fact"),
      replace(request, transcript=()),
    )
    for value in invalid:
      with self.subTest(value=value):
        adapter = DeterministicReplayFakeAdapter([VALID_OUTPUT])
        with use_modern_chat_runtime(build_modern_chat_runtime_config(),
                                     adapter):
          result = contract.generate_social_interpretation(value)
        self.assertEqual(contract.SocialInterpretationStatus.INVALID_CONTEXT,
                         result.status)
        self.assertEqual([], adapter.calls)

  def test_legacy_chat_without_marker_is_rejected_pre_provider(self):
    self.chat.keywords = {TOM, SAM}
    result, adapter = self.execute([VALID_OUTPUT])
    self.assertEqual(contract.SocialInterpretationStatus.INVALID_CONTEXT,
                     result.status)
    self.assertEqual("MISSING_EXPERIENCE_MARKER", result.reason)
    self.assertEqual([], adapter.calls)

  def test_wrong_actor_and_counterpart_are_rejected_pre_provider(self):
    for actor, counterpart in ((self.sam, SAM), (self.tom, "Wrong Person")):
      with self.subTest(actor=actor.name, counterpart=counterpart):
        result, adapter = self.execute(
          [VALID_OUTPUT], actor=actor, counterpart=counterpart)
        self.assertEqual(
          contract.SocialInterpretationStatus.INVALID_CONTEXT, result.status)
        self.assertEqual([], adapter.calls)

  def test_missing_identity_is_rejected_pre_provider(self):
    adapter = DeterministicReplayFakeAdapter([VALID_OUTPUT])
    with patch.object(self.tom.scratch, "get_str_iss", return_value=""), \
        use_modern_chat_runtime(build_modern_chat_runtime_config(), adapter):
      result = contract.interpret_social_experience(
        self.tom, SAM, self.chat, embedding_fn=self.embedding)
    self.assertEqual(contract.SocialInterpretationStatus.INVALID_CONTEXT,
                     result.status)
    self.assertEqual([], adapter.calls)

  def test_missing_transcript_is_rejected_pre_provider(self):
    self.chat.filling = []
    result, adapter = self.execute([VALID_OUTPUT])
    self.assertEqual(contract.SocialInterpretationStatus.INVALID_CONTEXT,
                     result.status)
    self.assertEqual([], adapter.calls)

  def test_chat_like_object_not_persisted_by_actor_is_rejected(self):
    fake = copy.copy(self.chat)
    result, adapter = self.execute([VALID_OUTPUT], chat=fake)
    self.assertEqual(contract.SocialInterpretationStatus.INVALID_CONTEXT,
                     result.status)
    self.assertEqual("UNPERSISTED_SOURCE_CHAT", result.reason)
    self.assertEqual([], adapter.calls)

  def test_multiple_and_malformed_experience_markers_are_rejected(self):
    original = set(self.chat.keywords)
    valid_other = (contract.EXPERIENCE_KEYWORD_PREFIX
                   + contract.SOCIAL_CONVERSATION_SCHEMA + ":" + "a" * 64)
    for keywords, reason in (
        (original | {valid_other}, "MULTIPLE_EXPERIENCE_MARKERS"),
        ({value for value in original
          if not str(value).startswith(contract.EXPERIENCE_KEYWORD_PREFIX)}
         | {contract.EXPERIENCE_KEYWORD_PREFIX + "bad"},
         "MALFORMED_EXPERIENCE_MARKER")):
      with self.subTest(reason=reason):
        self.chat.keywords = keywords
        result, adapter = self.execute([VALID_OUTPUT])
        self.assertEqual(contract.SocialInterpretationStatus.INVALID_CONTEXT,
                         result.status)
        self.assertEqual(reason, result.reason)
        self.assertEqual([], adapter.calls)

  def test_success_commits_exact_actor_local_thought_and_lineage_once(self):
    source_snapshot = copy.deepcopy(vars(self.chat))
    counterpart_snapshot = _memory_snapshot(self.sam.a_mem)
    scratch_snapshot = copy.deepcopy(vars(self.tom.scratch))
    result, adapter = self.execute([VALID_OUTPUT])
    self.assertEqual(contract.SocialInterpretationCommitStatus.COMMITTED,
                     result.status)
    thought = result.thought
    self.assertEqual((TOM, "interprets", SAM), thought.spo_summary())
    self.assertEqual(INTERPRETATION, thought.description)
    self.assertEqual([self.chat.node_id], thought.filling)
    self.assertEqual(0, thought.poignancy)
    self.assertIn(contract.interpretation_keyword(result.interpretation_id),
                  thought.keywords)
    self.assertEqual(source_snapshot, vars(self.chat))
    self.assertEqual(counterpart_snapshot, _memory_snapshot(self.sam.a_mem))
    self.assertEqual(scratch_snapshot, vars(self.tom.scratch))
    self.assertEqual([INTERPRETATION], self.embedding_calls)
    self.assertEqual(1, len(adapter.calls))

  def test_false_world_claim_remains_only_actor_thought(self):
    uncertain = "I suspect Sam may not follow through."
    output = '{"interpretation":"' + uncertain + '"}'
    maze = {"facts": ["Sam made a statement"]}
    maze_before = copy.deepcopy(maze)
    sam_before = _memory_snapshot(self.sam.a_mem)
    result, _ = self.execute([output])
    self.assertEqual(uncertain, result.thought.description)
    self.assertEqual(maze_before, maze)
    self.assertEqual(sam_before, _memory_snapshot(self.sam.a_mem))
    self.assertEqual([], self.sam.a_mem.seq_thought)

  def test_interpretation_id_is_deterministic_and_text_independent(self):
    request = self.request()
    first = contract.social_interpretation_id(request)
    second = contract.social_interpretation_id(request)
    self.assertEqual(first, second)
    self.assertTrue(first.startswith(
      contract.SOCIAL_INTERPRETATION_SCHEMA + ":"))
    self.assertEqual(first, contract.social_interpretation_id(request))
    changed = replace(request, counterpart_name="Different Person",
                      transcript=((TOM, "text"),
                                  ("Different Person", "reply")),
                      factual_description=(
                        'in a conversation; transcript: [["Tom Moreno","text"],'
                        '["Different Person","reply"]]'))
    self.assertNotEqual(first, contract.social_interpretation_id(changed))

  def test_repeat_before_save_returns_existing_without_provider_or_embedding(self):
    first, first_adapter = self.execute([VALID_OUTPUT])
    self.embedding_calls.clear()
    second, second_adapter = self.execute(
      [AssertionError("provider must not be called")])
    self.assertEqual(contract.SocialInterpretationCommitStatus.COMMITTED,
                     first.status)
    self.assertEqual(
      contract.SocialInterpretationCommitStatus.ALREADY_COMMITTED,
      second.status)
    self.assertIs(first.thought, second.thought)
    self.assertEqual(1, len(first_adapter.calls))
    self.assertEqual([], second_adapter.calls)
    self.assertEqual([], self.embedding_calls)
    self.assertEqual(1, len(self.tom.a_mem.seq_thought))

  def test_duplicate_marker_fails_closed_before_provider(self):
    first, _ = self.execute([VALID_OUTPUT])
    marker = contract.interpretation_keyword(first.interpretation_id)
    self.tom.a_mem.add_thought(
      NOW, None, TOM, "interprets", SAM, "duplicate", {TOM, SAM, marker},
      0, ("duplicate", [0.3] * 1536), [self.chat.node_id])
    self.embedding_calls.clear()
    adapter = DeterministicReplayFakeAdapter([VALID_OUTPUT])
    with use_modern_chat_runtime(build_modern_chat_runtime_config(), adapter):
      with self.assertRaises(contract.DuplicateSocialInterpretationError):
        contract.interpret_social_experience(
          self.tom, SAM, self.chat, embedding_fn=self.embedding)
    self.assertEqual([], adapter.calls)
    self.assertEqual([], self.embedding_calls)

  def test_marker_data_mismatch_fails_closed_before_provider(self):
    first, _ = self.execute([VALID_OUTPUT])
    first.thought.object = "Wrong Person"
    self.embedding_calls.clear()
    adapter = DeterministicReplayFakeAdapter([VALID_OUTPUT])
    with use_modern_chat_runtime(build_modern_chat_runtime_config(), adapter):
      with self.assertRaises(
          contract.SocialInterpretationDataMismatchError):
        contract.interpret_social_experience(
          self.tom, SAM, self.chat, embedding_fn=self.embedding)
    self.assertEqual([], adapter.calls)
    self.assertEqual([], self.embedding_calls)

  def test_commit_rejects_non_success_without_embedding(self):
    failure = contract.SocialInterpretationResult(
      contract.SocialInterpretationStatus.TIMEOUT, attempt_count=1,
      reason="LLMTimeoutError")
    with self.assertRaises(contract.SocialInterpretationCommitError):
      contract.commit_social_interpretation(
        self.tom, self.request(), self.chat, failure,
        embedding_fn=self.embedding)
    self.assertEqual([], self.embedding_calls)
    self.assertEqual([], self.tom.a_mem.seq_thought)

  def _save_reload_tom(self):
    target = Path(self.temporary.name) / TOM
    shutil.copytree(CONTROL / "personas" / TOM, target)
    self.tom.save(str(target / "bootstrap_memory"))
    return Persona(TOM, str(target))

  def test_save_reload_preserves_thought_embedding_marker_and_lineage(self):
    committed, _ = self.execute([VALID_OUTPUT])
    original = committed.thought
    self.tom = self._save_reload_tom()
    thought = self.tom.a_mem.seq_thought[0]
    chat = self.tom.a_mem.id_to_node[self.chat.node_id]
    self.chat = chat
    self.assertEqual("thought", thought.type)
    self.assertEqual(original.description, thought.description)
    self.assertEqual(original.spo_summary(), thought.spo_summary())
    self.assertEqual(original.created, thought.created)
    self.assertEqual(original.embedding_key, thought.embedding_key)
    self.assertEqual(original.filling, thought.filling)
    self.assertEqual(self.chat.node_id, thought.filling[0])
    self.assertIn(thought.embedding_key, self.tom.a_mem.embeddings)
    self.assertIn(
      contract.interpretation_keyword(committed.interpretation_id),
      thought.keywords)

  def test_repeat_after_reload_is_idempotent_without_calls_or_nodes(self):
    self.execute([VALID_OUTPUT])
    self.tom = self._save_reload_tom()
    self.chat = self.tom.a_mem.seq_chat[0]
    self.embedding_calls.clear()
    result, adapter = self.execute(
      [AssertionError("provider must not be called")])
    self.assertEqual(
      contract.SocialInterpretationCommitStatus.ALREADY_COMMITTED,
      result.status)
    self.assertEqual([], adapter.calls)
    self.assertEqual([], self.embedding_calls)
    self.assertEqual(1, len(self.tom.a_mem.seq_thought))

  def test_new_retrieve_can_include_interpretation_thought(self):
    committed, _ = self.execute([VALID_OUTPUT])
    with patch.object(retrieve, "get_embedding",
                      return_value=[0.25] * 1536):
      found = retrieve.new_retrieve(
        self.tom, ["Will Sam return with concrete ideas?"])
    self.assertIn(committed.thought,
                  found["Will Sam return with concrete ideas?"])
    self.assertNotIn(self.chat,
                     found["Will Sam return with concrete ideas?"])

  def test_operation_is_not_auto_wired_into_legacy_or_social_commit(self):
    sources = tuple((BACKEND / path).read_text(encoding="utf-8") for path in (
      "persona/cognitive_modules/reflect.py",
      "persona/cognitive_modules/plan.py",
      "persona/cognitive_modules/perceive.py",
      "persona/cognitive_modules/social_consequence.py",
    ))
    self.assertTrue(all("interpret_social_experience" not in source
                        for source in sources))


if __name__ == "__main__":
  unittest.main()
