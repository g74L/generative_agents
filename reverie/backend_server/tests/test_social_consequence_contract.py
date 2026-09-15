"""Offline regressions for the factual bilateral social-consequence boundary."""
from contextlib import ExitStack
import datetime
from pathlib import Path
import shutil
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
  sys.path.insert(0, str(BACKEND))

from persona.persona import Persona
from persona.cognitive_modules import perceive as perceive_module
from persona.cognitive_modules import plan
from persona.cognitive_modules.conversation_contract import (
  ConversationFailure,
  ConversationResult,
  ConversationStatus,
  ConversationTermination,
  build_utterance_request,
)
from persona.cognitive_modules.social_consequence import (
  FatalPartialSocialConsequenceCommitError,
  PartialSocialConsequenceStateError,
  SocialConsequenceCommitStatus,
  SocialConsequenceInvariantError,
  UNINTERPRETED_CHAT_POIGNANCY,
  build_social_conversation_experience,
  commit_social_conversation_experience,
  experience_keyword,
  factual_chat_description,
)
from persona.prompt_template.llm_provider import clear_telemetry, get_telemetry


STORAGE = BACKEND.parents[1] / "environment/frontend_server/storage"
CONTROL = STORAGE / "base_the_ville_n25"
TOM, SAM = "Tom Moreno", "Sam Moore"
NOW = datetime.datetime(2023, 2, 13, 13, 5)
TRANSCRIPT = (
  (TOM, "The neighborhood meeting starts at six."),
  (SAM, "I will bring the written proposal."),
)


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


class _OneTileMaze:
  def __init__(self, event):
    self.event = event

  def get_nearby_tiles(self, tile, radius):
    del tile, radius
    return [(0, 0)]

  def access_tile(self, tile):
    del tile
    return {
      "world": "the Ville", "sector": "Oak Hill", "arena": "store",
      "game_object": None, "events": {self.event},
    }

  def get_tile_path(self, tile, level):
    del tile, level
    return "the Ville:Oak Hill:store"


class SocialConsequenceContractTests(unittest.TestCase):
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
    self.tom = Persona(TOM, str(CONTROL / "personas" / TOM))
    self.sam = Persona(SAM, str(CONTROL / "personas" / SAM))
    for actor in (self.tom, self.sam):
      _empty_memory(actor.a_mem)
      actor.scratch.curr_time = NOW
      actor.scratch.curr_tile = (0, 0)
    self.conversation = ConversationResult(
      TRANSCRIPT, ConversationTermination.MODEL_END)
    self.embedding_calls = []

  def tearDown(self):
    self.assertEqual([], self.network)
    self.assertEqual((), get_telemetry())

  def embedding(self, text):
    self.embedding_calls.append(text)
    return [0.25] * 1536

  def commit(self, conversation=None, tom=None, sam=None, started_at=NOW):
    return commit_social_conversation_experience(
      conversation or self.conversation, tom or self.tom, sam or self.sam,
      started_at=started_at, embedding_fn=self.embedding)

  def _save_reload(self):
    reloaded = []
    root = Path(self.temporary.name)
    for actor in (self.tom, self.sam):
      target = root / actor.name
      shutil.copytree(CONTROL / "personas" / actor.name, target)
      actor.save(str(target / "bootstrap_memory"))
      reloaded.append(Persona(actor.name, str(target)))
    return tuple(reloaded)

  def _install_scratch_chat(self, actor, other, result, with_marker=True):
    actor.scratch.curr_time = NOW + datetime.timedelta(minutes=1)
    actor.scratch.curr_tile = (0, 0)
    actor.scratch.act_event = (actor.name, "chat with", other.name)
    actor.scratch.act_description = "legacy downstream summary"
    actor.scratch.chat = [list(row) for row in TRANSCRIPT]
    actor.scratch.chat_experience_id = (
      result.experience.experience_id if with_marker else None)
    return _OneTileMaze(
      (actor.name, "chat with", other.name, "talking together"))

  def test_model_end_commits_bilateral_independent_factual_nodes(self):
    result = self.commit()
    self.assertEqual(SocialConsequenceCommitStatus.COMMITTED, result.status)
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(1, len(self.sam.a_mem.seq_chat))
    a, b = result.initiator_chat, result.target_chat
    self.assertIsNot(a, b)
    self.assertEqual((TOM, "chat with", SAM), a.spo_summary())
    self.assertEqual((SAM, "chat with", TOM), b.spo_summary())
    self.assertEqual(TRANSCRIPT, a.filling)
    self.assertEqual(TRANSCRIPT, b.filling)
    self.assertEqual(NOW, a.created)
    self.assertEqual(NOW, b.created)
    self.assertEqual(UNINTERPRETED_CHAT_POIGNANCY, a.poignancy)
    self.assertEqual([a.description], self.embedding_calls)
    marker = experience_keyword(result.experience.experience_id)
    self.assertIn(marker, a.keywords)
    self.assertIn(marker, b.keywords)
    self.assertNotIn(marker.lower(), self.tom.a_mem.kw_to_event)
    self.assertNotIn(marker.lower(), self.tom.a_mem.kw_to_thought)

  def test_experience_id_is_canonical_deterministic_and_sensitive(self):
    first = build_social_conversation_experience(
      self.conversation, TOM, SAM, NOW)
    same = build_social_conversation_experience(
      self.conversation, TOM, SAM, NOW)
    later = build_social_conversation_experience(
      self.conversation, TOM, SAM, NOW + datetime.timedelta(seconds=1))
    changed = build_social_conversation_experience(ConversationResult(
      TRANSCRIPT[:-1] + ((SAM, "A different reply."),),
      ConversationTermination.MODEL_END), TOM, SAM, NOW)
    self.assertEqual(first.experience_id, same.experience_id)
    self.assertNotEqual(first.experience_id, later.experience_id)
    self.assertNotEqual(first.experience_id, changed.experience_id)
    for invalid_time in (
        NOW.replace(microsecond=1),
        NOW.replace(tzinfo=datetime.timezone.utc)):
      with self.subTest(invalid_time=invalid_time):
        with self.assertRaisesRegex(ValueError, "second precision"):
          build_social_conversation_experience(
            self.conversation, TOM, SAM, invalid_time)

  def test_factual_description_preserves_exact_utterances_without_summary(self):
    experience = build_social_conversation_experience(
      self.conversation, TOM, SAM, NOW)
    description = factual_chat_description(experience)
    self.assertTrue(description.startswith("in a conversation; transcript: "))
    for speaker, utterance in TRANSCRIPT:
      self.assertIn(speaker, description)
      self.assertIn(utterance, description)
    self.assertNotIn("legacy downstream summary", description)

  def test_safety_ceiling_and_failure_never_commit(self):
    failure = ConversationFailure(
      "utterance", TOM, SAM, ConversationStatus.TIMEOUT, 1,
      "LLMTimeoutError")
    conversations = (
      ConversationResult(TRANSCRIPT, ConversationTermination.SAFETY_CEILING),
      ConversationResult(TRANSCRIPT, ConversationTermination.FAILURE, failure),
    )
    for conversation in conversations:
      with self.subTest(termination=conversation.termination):
        with self.assertRaises(RuntimeError):
          self.commit(conversation=conversation)
        self.assertEqual([], self.tom.a_mem.seq_chat)
        self.assertEqual([], self.sam.a_mem.seq_chat)
    self.assertEqual([], self.embedding_calls)

  def test_same_experience_is_idempotent_without_second_embedding(self):
    first = self.commit()
    second = self.commit()
    self.assertEqual(SocialConsequenceCommitStatus.COMMITTED, first.status)
    self.assertEqual(
      SocialConsequenceCommitStatus.ALREADY_COMMITTED, second.status)
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(1, len(self.sam.a_mem.seq_chat))
    self.assertEqual(1, len(self.embedding_calls))

  def test_save_reload_preserves_marker_and_idempotency(self):
    first = self.commit()
    self.tom, self.sam = self._save_reload()
    self.embedding_calls.clear()
    second = self.commit()
    self.assertEqual(
      SocialConsequenceCommitStatus.ALREADY_COMMITTED, second.status)
    self.assertEqual(first.experience.experience_id,
                     second.experience.experience_id)
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(1, len(self.sam.a_mem.seq_chat))
    self.assertEqual([], self.embedding_calls)

  def test_one_sided_marker_fails_closed_without_repair(self):
    experience = build_social_conversation_experience(
      self.conversation, TOM, SAM, NOW)
    description = factual_chat_description(experience)
    self.tom.a_mem.add_chat(
      NOW, None, TOM, "chat with", SAM, description,
      {TOM, SAM, experience_keyword(experience.experience_id)}, 0,
      (description, [0.25] * 1536), TRANSCRIPT)
    with self.assertRaises(PartialSocialConsequenceStateError):
      self.commit()
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(0, len(self.sam.a_mem.seq_chat))
    self.assertEqual([], self.embedding_calls)

  def test_marker_data_mismatch_fails_closed(self):
    result = self.commit()
    result.initiator_chat.description = "invented incompatible narrative"
    with self.assertRaises(SocialConsequenceInvariantError):
      self.commit()
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(1, len(self.sam.a_mem.seq_chat))

  def test_second_write_failure_is_explicit_fatal_partial_commit(self):
    with patch.object(self.sam.a_mem, "add_chat",
                      side_effect=RuntimeError("write failed")):
      with self.assertRaises(FatalPartialSocialConsequenceCommitError):
        self.commit()
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(0, len(self.sam.a_mem.seq_chat))

  def test_direct_commit_self_perception_skips_chat_but_keeps_event(self):
    result = self.commit()
    maze = self._install_scratch_chat(self.tom, self.sam, result)
    before_chat = len(self.tom.a_mem.seq_chat)
    before_event = len(self.tom.a_mem.seq_event)
    with patch.object(perceive_module, "get_embedding",
                      return_value=[0.1, 0.2, 0.3]), \
        patch.object(perceive_module, "generate_poig_score", return_value=4):
      perceived = perceive_module.perceive(self.tom, maze)
    self.assertEqual(before_chat, len(self.tom.a_mem.seq_chat))
    self.assertEqual(before_event + 1, len(self.tom.a_mem.seq_event))
    self.assertEqual(1, len(perceived))
    self.assertEqual([result.initiator_chat.node_id], perceived[0].filling)

  def test_self_perception_marker_without_commit_fails_closed(self):
    fake = SimpleNamespace(experience=SimpleNamespace(
      experience_id="missing-experience"))
    maze = self._install_scratch_chat(self.tom, self.sam, fake)
    with patch.object(perceive_module, "get_embedding",
                      return_value=[0.1, 0.2, 0.3]), \
        patch.object(perceive_module, "generate_poig_score", return_value=4):
      with self.assertRaises(SocialConsequenceInvariantError):
        perceive_module.perceive(self.tom, maze)
    self.assertEqual([], self.tom.a_mem.seq_chat)
    self.assertEqual([], self.tom.a_mem.seq_event)

  def test_legacy_self_perception_still_creates_chat_and_event(self):
    fake = SimpleNamespace(experience=SimpleNamespace(experience_id=None))
    maze = self._install_scratch_chat(
      self.tom, self.sam, fake, with_marker=False)
    with patch.object(perceive_module, "get_embedding",
                      return_value=[0.1, 0.2, 0.3]), \
        patch.object(perceive_module, "generate_poig_score", return_value=4):
      perceived = perceive_module.perceive(self.tom, maze)
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(1, len(self.tom.a_mem.seq_event))
    self.assertEqual([self.tom.a_mem.seq_chat[0].node_id],
                     perceived[0].filling)
    self.assertEqual("legacy downstream summary",
                     self.tom.a_mem.seq_chat[0].description)

  def test_reload_direct_last_chat_and_utterance_context_are_meaningful(self):
    self.commit()
    self.tom, self.sam = self._save_reload()
    a = self.tom.a_mem.get_last_chat(SAM)
    b = self.sam.a_mem.get_last_chat(TOM)
    self.assertEqual(TRANSCRIPT, tuple(tuple(row) for row in a.filling))
    self.assertEqual(TRANSCRIPT, tuple(tuple(row) for row in b.filling))
    self.tom.scratch.curr_time = NOW + datetime.timedelta(minutes=10)
    self.tom.scratch.curr_tile = (0, 0)
    self.tom.scratch.act_description = "planning the meeting"
    self.sam.scratch.act_description = "reviewing the proposal"
    maze = SimpleNamespace(access_tile=lambda tile: {
      "arena": "store", "sector": "Oak Hill"})
    relationship = SimpleNamespace(
      status=ConversationStatus.SUCCESS, relationship="They are neighbors.")
    request = build_utterance_request(
      maze, self.tom, self.sam, {}, (), relationship)
    self.assertIsNotNone(request.previous_chat)
    for _, utterance in TRANSCRIPT:
      self.assertIn(utterance, request.previous_chat)

  def test_summary_failure_occurs_after_durable_in_memory_consequence(self):
    def actual_commit(*args, **kwargs):
      return commit_social_conversation_experience(
        *args, **kwargs, embedding_fn=self.embedding)

    with patch.object(plan, "generate_convo",
                      return_value=(self.conversation, 1)), \
        patch.object(plan, "commit_social_conversation_experience",
                     side_effect=actual_commit) as commit, \
        patch.object(plan, "generate_convo_summary",
                     side_effect=RuntimeError("summary failed")) as summary, \
        patch.object(plan, "_create_react") as create:
      with self.assertRaisesRegex(RuntimeError, "summary failed"):
        plan._chat_react(None, self.tom, None, f"chat with {SAM}",
                         {TOM: self.tom, SAM: self.sam})
    commit.assert_called_once()
    summary.assert_called_once()
    create.assert_not_called()
    self.assertEqual(1, len(self.tom.a_mem.seq_chat))
    self.assertEqual(1, len(self.sam.a_mem.seq_chat))

  def test_commit_failure_prevents_summary_and_schedule_calls(self):
    with patch.object(plan, "generate_convo",
                      return_value=(self.conversation, 1)), \
        patch.object(plan, "commit_social_conversation_experience",
                     side_effect=PartialSocialConsequenceStateError(
                       "partial")) as commit, \
        patch.object(plan, "generate_convo_summary") as summary, \
        patch.object(plan, "_create_react") as create:
      with self.assertRaises(PartialSocialConsequenceStateError):
        plan._chat_react(None, self.tom, None, f"chat with {SAM}",
                         {TOM: self.tom, SAM: self.sam})
    commit.assert_called_once()
    summary.assert_not_called()
    create.assert_not_called()

  def test_chat_react_passes_experience_marker_to_both_actions(self):
    def actual_commit(*args, **kwargs):
      return commit_social_conversation_experience(
        *args, **kwargs, embedding_fn=self.embedding)

    with patch.object(plan, "generate_convo",
                      return_value=(self.conversation, 1)), \
        patch.object(plan, "commit_social_conversation_experience",
                     side_effect=actual_commit), \
        patch.object(plan, "generate_convo_summary",
                     return_value="legacy summary"), \
        patch.object(plan, "_create_react") as create:
      plan._chat_react(None, self.tom, None, f"chat with {SAM}",
                       {TOM: self.tom, SAM: self.sam})
    self.assertEqual(2, create.call_count)
    markers = [call.args[-1] for call in create.call_args_list]
    self.assertEqual(1, len(set(markers)))
    self.assertEqual(
      experience_keyword(markers[0]),
      next(keyword for keyword in self.tom.a_mem.seq_chat[0].keywords
           if keyword.startswith("__oce_social_experience__:")))


if __name__ == "__main__":
  unittest.main()
