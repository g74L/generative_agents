import contextlib
import io
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template import gpt_structure, run_gpt_prompt


parse = run_gpt_prompt._parse_task_decomp_response


def prompt(total):
  return ("In 5 min increments (total duration in minutes: "
          f"{total}):")


class TaskDecompParserTests(unittest.TestCase):
  def assert_invalid(self, response, total=10):
    with self.assertRaises(ValueError) as raised:
      parse(response, prompt(total))
    self.assertNotIsInstance(raised.exception, IndexError)
    if isinstance(response, str) and response.strip():
      self.assertNotIn(response, str(raised.exception))

  def test_01_canonical_response_preserves_list_contract(self):
    response = (
      "1) wash up (duration in minutes: 10, minutes left: 20)\n"
      "2) prepare breakfast (duration in minutes: 20, minutes left: 0)")
    self.assertEqual(
      [["wash up", 10], ["prepare breakfast", 20]],
      parse(response, prompt(30)))

  def test_02_blank_lines_crlf_and_trailing_newline_are_ignored(self):
    response = (
      "  1) wash up (duration in minutes: 10)\r\n"
      "   \r\n2) prepare breakfast (duration in minutes: 20)\r\n")
    self.assertEqual(
      [["wash up", 10], ["prepare breakfast", 20]],
      parse(response, prompt(30)))

  def test_03_empty_responses_fail_without_index_error(self):
    for response in ("", " ", "\n", "\n\n"):
      with self.subTest(response=repr(response)):
        self.assert_invalid(response)

  def test_04_empty_task_and_missing_marker_are_rejected(self):
    for response in (
        "(duration in minutes: 10)",
        "wash up for ten minutes",
        "wash up\n(duration in minutes: 10)"):
      with self.subTest(response=response):
        self.assert_invalid(response)

  def test_05_invalid_durations_are_rejected(self):
    for value in ("", "ten", "0", "-5", "1.5"):
      with self.subTest(value=value):
        self.assert_invalid(
          f"wash up (duration in minutes: {value})")

  def test_06_supported_list_prefixes_are_explicit(self):
    response = "\n".join((
      "1) wash up (duration in minutes: 5)",
      "2. shower (duration in minutes: 5)",
      "- dress (duration in minutes: 5)",
      "* breakfast (duration in minutes: 5)",
    ))
    self.assertEqual(
      [["wash up", 5], ["shower", 5], ["dress", 5], ["breakfast", 5]],
      parse(response, prompt(20)))

  def test_07_short_description_and_final_period_are_preserved_safely(self):
    self.assertEqual(
      [["shower", 10]],
      parse("shower (duration in minutes: 10)", prompt(10)))
    self.assertEqual(
      [["wash up", 10]],
      parse("wash up. (duration in minutes: 10)", prompt(10)))

  def test_08_total_duration_alignment_preserves_legacy_intent(self):
    cases = (
      ("wash (duration in minutes: 10)\n"
       "breakfast (duration in minutes: 20)",
       [["wash", 10], ["breakfast", 20]]),
      ("wash (duration in minutes: 10)\n"
       "breakfast (duration in minutes: 10)",
       [["wash", 10], ["breakfast", 20]]),
      ("wash (duration in minutes: 20)\n"
       "breakfast (duration in minutes: 20)",
       [["wash", 20], ["breakfast", 10]]),
    )
    for response, expected in cases:
      with self.subTest(response=response):
        result = parse(response, prompt(30))
        self.assertEqual(expected, result)
        self.assertEqual(30, sum(duration for _, duration in result))

  def test_09_prompt_total_is_parsed_from_last_canonical_marker(self):
    full_prompt = (
      "example (total duration in minutes: 180):\n"
      "request (total duration in minutes 10):")
    self.assertEqual(
      [["wash", 10]],
      parse("wash (duration in minutes: 10)", full_prompt))
    with self.assertRaisesRegex(ValueError,
                                "TASK_DECOMP_PROMPT_TOTAL_INVALID"):
      parse("wash (duration in minutes: 10)", "missing total")

  def test_10_parser_never_prints_raw_response(self):
    response = "private response (duration in minutes: 10)"
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
      self.assertEqual([["private response", 10]],
                       parse(response, prompt(10)))
    self.assertEqual("", output.getvalue())


class TaskDecompSafeGenerateIntegrationTests(unittest.TestCase):
  def invoke(self, responses, repeat, fail_safe):
    def validate(response, prompt=""):
      try:
        parse(response, prompt)
      except ValueError:
        return False
      return True

    network_calls = []
    def block_network(*args, **kwargs):
      del args, kwargs
      network_calls.append("blocked")
      raise AssertionError("network forbidden in task_decomp parser tests")

    with patch.object(gpt_structure, "GPT_request",
                      side_effect=list(responses)) as fake_provider, \
        patch("socket.create_connection", side_effect=block_network), \
        patch.object(socket.socket, "connect", block_network):
      result = gpt_structure.safe_generate_response(
        prompt(10), {}, repeat, fail_safe, validate, parse,
        caller_id="task_decomp")
    self.assertEqual([], network_calls)
    return result, fake_provider.call_count

  def test_11_valid_response_cleans_up_on_first_attempt(self):
    result, calls = self.invoke(
      ["wash (duration in minutes: 10)"], 2, [["working", 10]])
    self.assertEqual([["wash", 10]], result)
    self.assertEqual(1, calls)

  def test_12_malformed_then_valid_response_retries_cleanly(self):
    result, calls = self.invoke(
      ["malformed", "wash (duration in minutes: 10)"],
      2, [["working", 10]])
    self.assertEqual([["wash", 10]], result)
    self.assertEqual(2, calls)

  def test_13_all_malformed_responses_return_structured_fail_safe(self):
    fail_safe = [["working", 10]]
    result, calls = self.invoke(["malformed", "still malformed"],
                                2, fail_safe)
    self.assertEqual(fail_safe, result)
    self.assertEqual(2, calls)


if __name__ == "__main__":
  unittest.main()
