"""Offline conversation boundary regressions through real runtime/fake adapter."""
import copy
from contextlib import ExitStack
from dataclasses import asdict, replace
import datetime
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

from controlled_replay import DeterministicReplayFakeAdapter
from persona.persona import Persona
from persona.memory_structures.scratch import Scratch
from persona.cognitive_modules import conversation_contract as contract, converse, plan
from persona.prompt_template import gpt_structure, modern_openai_provider as errors
from persona.prompt_template.chat_runtime import (
  build_modern_chat_runtime_config, use_modern_chat_runtime,
)
from persona.prompt_template.llm_provider import clear_telemetry, get_telemetry
from persona.prompt_template.replay_cost_guard import ReplayCostGuardError
STORAGE = BACKEND.parents[1] / 'environment/frontend_server/storage'
CONTROL = STORAGE / 'base_the_ville_n25'
TREATMENT = STORAGE / 'base_the_ville_n25_cpi0_tom'
TOM, SAM = 'Tom Moreno', 'Sam Moore'
NOW = datetime.datetime(2023, 2, 13, 13)

Status = contract.ConversationStatus
Termination = contract.ConversationTermination
REL = '{"relationship":"They know each other through the store."}'
UTT = '{"utterance":"I disagree with that.","end":false}'
END = '{"utterance":"We\'ll leave it there.","end":true}'


class ConversationContractTests(unittest.TestCase):
  def setUp(self):
    self.stack = ExitStack()
    self.addCleanup(self.stack.close)
    clear_telemetry()
    self.addCleanup(clear_telemetry)
    self.network = []
    def reject(*args, **kwargs):
      self.network.append('attempt')
      raise AssertionError('network forbidden')
    for owner, name in ((socket, 'create_connection'), (socket.socket, 'connect'),
                        (socket.socket, 'connect_ex')):
      self.stack.enter_context(patch.object(owner, name, reject))
    self.live_guards = [self.stack.enter_context(patch.object(
      errors.ModernOpenAIClientAdapter, name, side_effect=AssertionError('live forbidden')))
      for name in ('create_chat', 'create_embedding')]
    self.tom = Persona(TOM, str(CONTROL / 'personas' / TOM))
    self.sam = Persona(SAM, str(CONTROL / 'personas' / SAM))
    for actor in (self.tom, self.sam):
      actor.scratch.curr_time = NOW
      actor.scratch.curr_tile = (88, 44) if actor.name == TOM else (87, 44)
      actor.scratch.act_description = ('serving customers' if actor.name == TOM else
        'visiting the store and discussing local business concerns in relation to the upcoming mayoral election')
      self.assertEqual([], actor.a_mem.seq_event)
      self.assertEqual([], actor.a_mem.seq_thought)
    self.maze = SimpleNamespace(access_tile=lambda tile: {
      'sector': 'The Willows Market and Pharmacy', 'arena': 'store'})
    self.relationship = contract.ConversationRelationshipResult(Status.SUCCESS, 'They know each other.')

  def tearDown(self):
    self.assertEqual([], self.network)
    for guard in self.live_guards:
      guard.assert_not_called()

  def request(self, speaker=None, target=None, retrieved=None, transcript=()):
    return contract.build_utterance_request(self.maze, speaker or self.tom,
      target or self.sam, {} if retrieved is None else retrieved, transcript, self.relationship)

  def execute(self, responses, surface='utterance', request=None, retries=0):
    adapter = DeterministicReplayFakeAdapter(responses)
    with use_modern_chat_runtime(build_modern_chat_runtime_config(application_retry_count=retries), adapter):
      if surface == 'conversation':
        result = converse.agent_chat_v2(self.maze, self.sam, self.tom)
      elif surface == 'bridge':
        result = plan.generate_convo(self.maze, self.sam, self.tom)
      elif surface == 'relationship':
        result = contract.interpret_relationship(request or contract.build_relationship_request(self.tom, self.sam, {}))
      else:
        result = contract.generate_utterance(request or self.request())
    return result, adapter

  def snapshot(self):
    return copy.deepcopy((vars(self.sam.scratch), vars(self.tom.scratch),
                          vars(self.sam.a_mem), vars(self.tom.a_mem), vars(self.maze)))

  def test_valid_utterance_preserves_exact_text_and_boolean(self):
    result, adapter = self.execute([UTT])
    self.assertEqual(Status.SUCCESS, result.status)
    self.assertEqual('I disagree with that.', result.utterance)
    self.assertIs(False, result.end)
    self.assertEqual(1, result.attempt_count)
    self.assertEqual(1, len(adapter.calls))

  def test_model_end_appends_valid_turn_and_stops_immediately(self):
    before = self.snapshot()
    result, adapter = self.execute([REL, END], 'conversation')
    self.assertEqual(Termination.MODEL_END, result.termination)
    self.assertEqual(((SAM, "We'll leave it there."),), result.transcript)
    self.assertEqual(1, result.turn_count)
    self.assertEqual(2, len(adapter.calls))
    self.assertEqual(before, self.snapshot())

  def test_model_end_is_normally_consumable(self):
    result = contract.ConversationResult(
      ((SAM, 'A valid completed turn.'),), Termination.MODEL_END)
    self.assertIs(result, result.require_complete())
    self.assertEqual(list(result.transcript), list(result))

  def test_safety_ceiling_preserves_sixteen_valid_alternating_turns(self):
    result, adapter = self.execute([REL, UTT] * 16, 'conversation')
    self.assertEqual(Termination.SAFETY_CEILING, result.termination)
    self.assertIsNone(result.failure)
    self.assertEqual(16, result.turn_count)
    self.assertEqual([SAM, TOM] * 8, [row[0] for row in result.transcript])
    self.assertEqual(32, len(adapter.calls))

  def test_safety_ceiling_is_incomplete_but_diagnostically_available(self):
    transcript = ((SAM, 'A valid but unfinished turn.'),)
    result = contract.ConversationResult(transcript, Termination.SAFETY_CEILING)
    self.assertEqual(transcript, result.transcript)
    self.assertIsNone(result.failure)
    for consume in (result.require_complete, lambda: list(result)):
      with self.subTest(consume=consume):
        with self.assertRaises(contract.ConversationIncompleteError) as raised:
          consume()
        self.assertIs(result, raised.exception.result)
        self.assertEqual(Termination.SAFETY_CEILING,
                         raised.exception.termination)
        self.assertEqual(1, raised.exception.turn_count)

  def test_alternation_stops_on_target_model_end(self):
    result, adapter = self.execute([REL, UTT, REL, END], 'conversation')
    self.assertEqual(Termination.MODEL_END, result.termination)
    self.assertEqual([SAM, TOM], [row[0] for row in result.transcript])
    self.assertEqual(4, len(adapter.calls))
    events = get_telemetry()
    self.assertEqual([SAM, SAM, TOM, TOM], [e.actor_id for e in events])
    self.assertEqual(['CONVERSATION'] * 4, [e.cognitive_category for e in events])

  def test_relationship_requires_exact_nonempty_text_object(self):
    invalid = ['{}', '{"relationship":""}', '{"relationship":"  "}',
      '{"relationship":null}', '{"relationship":42}', '{"output":"legacy"}',
      '{"relationship":"valid","extra":true}',
      '{"relationship":"a","relationship":"b"}', '...']
    for text in invalid:
      with self.subTest(text=text):
        result, adapter = self.execute([text] * 3, 'relationship')
        self.assertEqual(Status.INVALID_OUTPUT, result.status)
        self.assertIsNone(result.relationship)
        self.assertEqual(3, len(adapter.calls))
    result, _ = self.execute([REL], 'relationship')
    self.assertEqual('They know each other through the store.', result.relationship)

  def test_invalid_booleans_do_not_become_end_decisions_or_speech(self):
    for end in ('false', 'continue', None, 0, 1, [], {}):
      with self.subTest(end=end):
        text = json.dumps({'utterance': 'text', 'end': end})
        result, adapter = self.execute([REL] + [text] * 3, 'conversation')
        self.assertEqual(Termination.FAILURE, result.termination)
        self.assertEqual(Status.INVALID_OUTPUT, result.failure.status)
        self.assertEqual((), result.transcript)
        self.assertEqual(4, len(adapter.calls))

  def test_invalid_utterance_schema_and_wrappers_are_rejected(self):
    invalid = ['null', '[]', '{}', 'non-JSON', '{"end":false}',
      '{"utterance":"","end":false}', '{"utterance":"  ","end":false}',
      '{"utterance":123,"end":false}', '{"utterance":null,"end":false}',
      '{"utterance":true,"end":false}', '{"utterance":[],"end":false}',
      '{"speaker":"text","other":false}', '{"other":false,"speaker":"text"}',
      '{"utterance":"a","end":false,"extra":0}',
      '{"utterance":"a","end":false,"end":true}',
      '```json\n' + UTT + '\n```', UTT + ' trailing',
      '{"utterance":"x","end":NaN}']
    for text in invalid:
      with self.subTest(text=text):
        result, adapter = self.execute([REL] + [text] * 3, 'conversation')
        self.assertEqual(Status.INVALID_OUTPUT, result.failure.status)
        self.assertEqual(Termination.FAILURE, result.termination)
        self.assertEqual((), result.transcript)
        self.assertEqual(3, result.failure.attempt_count)
        self.assertEqual(4, len(adapter.calls))
    # JSON field order carries no meaning; exact named fields are accepted.
    self.assertEqual(('ok', False), contract.parse_utterance_output('{"end":false,"utterance":"ok"}'))

  def test_legacy_fallback_path_is_never_called(self):
    with ExitStack() as stack:
      guards = [stack.enter_context(patch.object(owner, name,
        side_effect=AssertionError('legacy bypass'))) for owner, name in (
          (converse, 'run_gpt_generate_iterative_chat_utt'),
          (converse, 'run_gpt_prompt_agent_chat_summarize_relationship'),
          (gpt_structure, 'ChatGPT_safe_generate_response_OLD'),
          (gpt_structure, 'ChatGPT_safe_generate_response'))]
      result, _ = self.execute([REL] + ['not-json'] * 3, 'conversation')
      self.assertEqual(Termination.FAILURE, result.termination)
      self.assertEqual(Status.INVALID_OUTPUT, result.failure.status)
      self.assertEqual((), result.transcript)
      for guard in guards:
        guard.assert_not_called()

  def test_relationship_failure_prevents_utterance_and_second_retrieval(self):
    with patch.object(converse, 'new_retrieve', wraps=converse.new_retrieve) as retrieve, \
        patch.object(converse, 'generate_one_utterance') as utterance:
      result, adapter = self.execute(['malformed'] * 3, 'conversation')
    self.assertEqual('relationship', result.failure.surface)
    self.assertEqual(Status.INVALID_OUTPUT, result.failure.status)
    self.assertEqual(1, retrieve.call_count)
    utterance.assert_not_called()
    self.assertEqual(3, len(adapter.calls))

  def test_failed_later_turn_preserves_only_previous_success(self):
    result, adapter = self.execute([REL, UTT, REL] + ['bad'] * 3, 'conversation')
    self.assertEqual(((SAM, 'I disagree with that.'),), result.transcript)
    self.assertEqual(TOM, result.failure.speaker)
    self.assertEqual(6, len(adapter.calls))

  def test_provider_error_mapping_on_both_required_surfaces(self):
    for error_type, status in (
        (errors.LLMRefusalError, Status.REFUSAL), (errors.LLMTimeoutError, Status.TIMEOUT),
        (errors.LLMConnectionError, Status.UNAVAILABLE), (errors.LLMRateLimitError, Status.UNAVAILABLE),
        (errors.LLMServerError, Status.UNAVAILABLE), (errors.LLMAuthenticationError, Status.PROVIDER_ERROR),
        (errors.LLMAuthorizationError, Status.PROVIDER_ERROR), (errors.LLMModelNotFoundError, Status.PROVIDER_ERROR),
        (errors.LLMInvalidRequestError, Status.PROVIDER_ERROR), (errors.LLMUnsupportedOperationError, Status.PROVIDER_ERROR),
        (errors.ModernOpenAISdkUnavailableError, Status.PROVIDER_ERROR), (errors.LLMProviderError, Status.PROVIDER_ERROR)):
      for surface in ('relationship', 'utterance'):
        with self.subTest(error=error_type.__name__, surface=surface):
          sequence = [] if surface == 'relationship' else [REL]
          result, adapter = self.execute(sequence + [error_type('private provider text')], 'conversation')
          self.assertEqual(status, result.failure.status)
          self.assertEqual(surface, result.failure.surface)
          self.assertEqual(Termination.FAILURE, result.termination)
          self.assertEqual((), result.transcript)
          self.assertEqual(len(sequence) + 1, len(adapter.calls))
          self.assertNotIn('private provider text', repr(result))

  def test_empty_incomplete_malformed_responses_exhaust_as_invalid(self):
    for error_type in (errors.LLMEmptyOutputError, errors.LLMIncompleteResponseError,
                       errors.LLMMalformedResponseError):
      for surface in ('relationship', 'utterance'):
        with self.subTest(error=error_type.__name__, surface=surface):
          result, adapter = self.execute([error_type('private')] * 3, surface)
          self.assertEqual(Status.INVALID_OUTPUT, result.status)
          self.assertEqual(3, result.attempt_count)
          self.assertEqual(3, len(adapter.calls))
    result, adapter = self.execute([''] * 3)
    self.assertEqual(Status.INVALID_OUTPUT, result.status)
    self.assertEqual(3, len(adapter.calls))

  def test_inactive_runtime_is_provider_error_without_legacy_fallback(self):
    for result in (contract.generate_utterance(self.request()),
                   contract.interpret_relationship(contract.build_relationship_request(self.tom, self.sam, {}))):
      self.assertEqual(Status.PROVIDER_ERROR, result.status)
      self.assertEqual('ModernChatRuntimeInactiveError', result.reason)
    self.assertEqual((), get_telemetry())

  def test_programming_and_accounting_exceptions_propagate(self):
    for error in (TypeError('bug'), AssertionError('bug'), RuntimeError('bug'), ReplayCostGuardError('guard')):
      for surface in ('relationship', 'utterance'):
        with self.subTest(error=type(error).__name__, surface=surface):
          with self.assertRaises(type(error)):
            self.execute([error], surface)

  def test_format_retries_keep_one_logical_call_and_identical_prompt(self):
    result, adapter = self.execute(['invalid', UTT])
    self.assertEqual(Status.SUCCESS, result.status)
    self.assertEqual(2, result.attempt_count)
    self.assertEqual(adapter.calls[0][1], adapter.calls[1][1])
    events = get_telemetry()
    self.assertEqual([1, 2], [event.physical_attempt for event in events])
    self.assertEqual(1, len({event.logical_call_id for event in events}))

  def test_transport_retry_accounting_remains_owned_by_runtime(self):
    result, adapter = self.execute([errors.LLMTimeoutError('timeout'), UTT], retries=1)
    self.assertEqual(Status.SUCCESS, result.status)
    self.assertEqual(1, result.attempt_count)
    self.assertEqual(2, len(adapter.calls))
    self.assertEqual([1, 2], [event.physical_attempt for event in get_telemetry()])

  def test_tom_control_treatment_identity_available_in_actual_provider_input(self):
    path = TREATMENT / 'personas' / TOM / 'bootstrap_memory/scratch.json'
    treatment = SimpleNamespace(name=TOM, scratch=Scratch(str(path)), a_mem=self.tom.a_mem)
    treatment.scratch.curr_time = NOW
    treatment.scratch.curr_tile = self.tom.scratch.curr_tile
    treatment.scratch.act_description = self.tom.scratch.act_description
    requests = [self.request(speaker=s) for s in (self.tom, treatment)]
    self.assertEqual({'identity'}, {k for k in asdict(requests[0]) if asdict(requests[0])[k] != asdict(requests[1])[k]})
    for request, expected in zip(requests, ('rude, aggressive, energetic', 'rude, argumentative, easily provoked, energetic')):
      result, adapter = self.execute([UTT], request=request)
      prompt = adapter.calls[0][1]['messages'][0]['content']
      self.assertIn(expected, prompt)
      self.assertIn(request.identity, prompt)
    for phrase in ('problematic relationship with alcohol', 'physical fight',
                   'disrespected, challenged, or provoked'):
      self.assertIn(phrase, prompt)

  def test_target_identity_excluded_unless_explicitly_known_locally(self):
    marker = 'PRIVATE_TARGET_MARKER_127'
    for name in ('innate', 'learned', 'currently'):
      setattr(self.sam.scratch, name, marker)
    with patch.object(self.sam.scratch, 'get_str_iss', side_effect=AssertionError('private ISS read')):
      for retrieved, known in (({}, False), ({SAM: [SimpleNamespace(description=marker, embedding_key=marker)]}, True)):
        relationship = contract.build_relationship_request(self.tom, self.sam, retrieved)
        utterance = self.request(retrieved=retrieved)
        for prompt in (contract.render_relationship_prompt(relationship), contract.render_utterance_prompt(utterance)):
          self.assertEqual(known, marker in prompt)

  def test_relationship_context_transcript_and_observable_situation_are_supplied(self):
    request = self.request(transcript=((SAM, 'Already spoken'),))
    prompt = contract.render_utterance_prompt(request)
    for value in (request.relationship, 'Already spoken', request.target_activity, request.speaker_activity):
      self.assertIn(value, prompt)
    self.assertIn('Continue the conversation', prompt)
    self.assertNotIn('speaker initiates', prompt)

  def test_invalid_context_returns_before_provider(self):
    for field, value in (('identity', ''), ('relationship', ''), ('memories', None),
                          ('current_time', None), ('transcript', (('third actor', 'text'),)),
                          ('previous_chat', ''), ('target_activity', None)):
      with self.subTest(field=field):
        result, adapter = self.execute([UTT], request=replace(self.request(), **{field:value}))
        self.assertEqual(Status.INVALID_CONTEXT, result.status)
        self.assertEqual([], adapter.calls)
    result, adapter = self.execute([REL], 'relationship',
      replace(contract.build_relationship_request(self.tom, self.sam, {}), memories=None))
    self.assertEqual(Status.INVALID_CONTEXT, result.status)
    self.assertEqual([], adapter.calls)

  def test_bridge_retains_typed_success_and_failure_blocks_consequences(self):
    (result, duration), adapter = self.execute([REL, END], 'bridge')
    self.assertIsInstance(result, contract.ConversationResult)
    self.assertEqual(Termination.MODEL_END, result.termination)
    self.assertEqual(list(result.transcript), list(result))
    self.assertIsInstance(duration, int)
    with ExitStack() as stack:
      guards = [stack.enter_context(patch.object(plan, name, side_effect=AssertionError('consequence forbidden')))
                for name in ('_chat_react', '_create_react', 'generate_new_decomp_schedule', 'generate_convo_summary')]
      with self.assertRaises(contract.ConversationCognitionUnavailableError) as raised:
        self.execute([REL] + ['invalid'] * 3, 'bridge')
      self.assertEqual(Status.INVALID_OUTPUT, raised.exception.status)
      self.assertEqual((), raised.exception.result.transcript)
      for guard in guards:
        guard.assert_not_called()

  def test_safety_ceiling_bridge_blocks_completed_conversation_consumers(self):
    with ExitStack() as stack:
      guards = [stack.enter_context(patch.object(plan, name,
        side_effect=AssertionError('completed consequence forbidden')))
        for name in ('_chat_react', '_create_react',
                     'generate_new_decomp_schedule', 'generate_convo_summary')]
      with self.assertRaises(contract.ConversationIncompleteError) as raised:
        self.execute([REL, UTT] * 16, 'bridge')
      self.assertEqual(Termination.SAFETY_CEILING,
                       raised.exception.termination)
      self.assertEqual(16, raised.exception.turn_count)
      self.assertEqual(16, len(raised.exception.result.transcript))
      for guard in guards:
        guard.assert_not_called()

  def test_failure_remains_cognitive_failure_with_diagnostic_transcript(self):
    transcript = ((SAM, 'A prior valid turn.'),)
    failure = contract.ConversationFailure(
      'utterance', TOM, SAM, Status.TIMEOUT, 1, 'LLMTimeoutError')
    result = contract.ConversationResult(
      transcript, Termination.FAILURE, failure)
    self.assertEqual(transcript, result.transcript)
    for consume in (result.require_complete, lambda: list(result)):
      with self.subTest(consume=consume):
        with self.assertRaises(
            contract.ConversationCognitionUnavailableError) as raised:
          consume()
        self.assertIs(result, raised.exception.result)
        self.assertEqual(Status.TIMEOUT, raised.exception.status)
        self.assertEqual('utterance', raised.exception.surface)

  def test_non_success_cannot_carry_speech_relationship_or_end(self):
    for status in Status:
      if status == Status.SUCCESS:
        continue
      with self.assertRaises(ValueError):
        contract.ConversationUtteranceResult(status, '...', False, reason='failure')
      with self.assertRaises(ValueError):
        contract.ConversationRelationshipResult(status, '', reason='failure')
    for utterance, end in (('', False), ('ok', 'false'), (42, False), ('ok', None)):
      with self.assertRaises(ValueError):
        contract.ConversationUtteranceResult(Status.SUCCESS, utterance, end)

  def test_retrieval_error_fails_closed_without_cognition(self):
    with patch.object(converse, 'new_retrieve', side_effect=errors.LLMTimeoutError('private')):
      result, adapter = self.execute([], 'conversation')
    self.assertEqual(Status.TIMEOUT, result.failure.status)
    self.assertEqual('relationship_retrieval', result.failure.surface)
    self.assertEqual((), result.transcript)
    self.assertEqual([], adapter.calls)


if __name__ == '__main__':
  unittest.main()
