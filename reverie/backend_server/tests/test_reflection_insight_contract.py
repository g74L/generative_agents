from pathlib import Path
from types import SimpleNamespace
import datetime
import json
import sys
import unittest
from unittest.mock import Mock, patch


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.cognitive_modules import reflect as reflect_module
from persona.prompt_template import gpt_structure
from persona.prompt_template import run_gpt_prompt


class ReflectionInsightContractTests(unittest.TestCase):
  def nodes(self):
    return [
      SimpleNamespace(node_id="node_A", embedding_key="alpha"),
      SimpleNamespace(node_id="node_B", embedding_key="beta"),
      SimpleNamespace(node_id="node_C", embedding_key="gamma"),
    ]

  def generate(self, evidence):
    captured = {}

    def fake_prompt(persona, statements, n):
      captured["statements"] = statements
      return {"valid insight": evidence}, []

    with patch.object(
        reflect_module, "run_gpt_prompt_insight_and_guidance", fake_prompt):
      result = reflect_module.generate_insights_and_evidence(
        object(), self.nodes(), 1)
    return result, captured["statements"]

  def test_01_maps_basic_1_based_evidence_and_numbers_statements(self):
    result, statements = self.generate([1, 3])
    self.assertEqual({"valid insight": ["node_A", "node_C"]}, result)
    self.assertEqual("1. alpha\n2. beta\n3. gamma\n", statements)

  def test_02_maps_first_evidence_item(self):
    self.assertEqual(["node_A"], self.generate([1])[0]["valid insight"])

  def test_03_maps_last_evidence_item(self):
    self.assertEqual(["node_C"], self.generate([3])[0]["valid insight"])

  def test_04_preserves_multiple_evidence_order(self):
    self.assertEqual(
      ["node_C", "node_A", "node_B"],
      self.generate([3, 1, 2])[0]["valid insight"])

  def test_05_rejects_out_of_range_high_evidence(self):
    with self.assertRaises(
        reflect_module.ReflectionInsightContractError) as raised:
      self.generate([4])
    self.assertEqual("EVIDENCE_OUT_OF_RANGE", raised.exception.category)

  def test_06_rejects_zero_evidence(self):
    with self.assertRaises(reflect_module.ReflectionInsightContractError):
      self.generate([0])

  def test_07_rejects_negative_evidence(self):
    with self.assertRaises(reflect_module.ReflectionInsightContractError):
      self.generate([-1])

  def test_08_rejects_empty_evidence(self):
    with self.assertRaises(reflect_module.ReflectionInsightContractError):
      self.generate([])

  def test_09_retry_exhaustion_returns_type_safe_empty_mapping(self):
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="prompt"), patch.object(
        gpt_structure, "GPT_request", return_value="malformed" ) as request:
      output, details = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. alpha", 5)
    self.assertEqual({}, output)
    self.assertEqual({}, details[-1])
    self.assertEqual(5, request.call_count)

  def test_10_invalid_generation_never_writes_artificial_thought(self):
    memory = SimpleNamespace(add_thought=Mock())
    persona = SimpleNamespace(a_mem=memory)
    nodes = self.nodes()
    with patch.object(
        reflect_module, "generate_focal_points", return_value=["focus"]), (
        patch.object(reflect_module, "new_retrieve",
                     return_value={"focus": nodes})), patch.object(
        reflect_module, "run_gpt_prompt_insight_and_guidance",
        return_value=({}, [])):
      with self.assertRaises(reflect_module.ReflectionInsightContractError):
        reflect_module.run_reflect(persona)
    memory.add_thought.assert_not_called()

  def test_11_valid_lineage_reaches_add_thought_as_node_id_list(self):
    memory = SimpleNamespace(add_thought=Mock())
    persona = SimpleNamespace(
      a_mem=memory,
      scratch=SimpleNamespace(curr_time=datetime.datetime(2026, 1, 1)))
    nodes = self.nodes()
    with patch.object(
        reflect_module, "generate_focal_points", return_value=["focus"]), (
        patch.object(reflect_module, "new_retrieve",
                     return_value={"focus": nodes})), patch.object(
        reflect_module, "run_gpt_prompt_insight_and_guidance",
        return_value=({"valid insight": [1, 3]}, [])), patch.object(
        reflect_module, "generate_action_event_triple",
        return_value=("actor", "reflects", "insight")), patch.object(
        reflect_module, "generate_poig_score", return_value=7), patch.object(
        reflect_module, "get_embedding", return_value=[0.1, 0.2]):
      reflect_module.run_reflect(persona)
    self.assertEqual(["node_A", "node_C"], memory.add_thought.call_args.args[-1])
    self.assertNotIn("this is blank", memory.add_thought.call_args.args)
    self.assertNotIn("node_1", memory.add_thought.call_args.args)

  def test_12_insight_prompt_uses_bounded_300_token_budget(self):
    captured = {}

    def fake_safe(prompt, parameters, repeat, fail_safe, validate, clean_up,
                  *, caller_id):
      captured.update(parameters)
      self.assertEqual("insight_and_guidance", caller_id)
      return {"valid insight": [1]}

    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="prompt"), patch.object(
        run_gpt_prompt, "safe_generate_response", fake_safe):
      output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. alpha", 1)[0]
    self.assertEqual({"valid insight": [1]}, output)
    self.assertEqual(300, captured["max_tokens"])

  def test_13_parser_rejects_non_positive_evidence_numbers(self):
    for response in ("Insight (because of 0)",
                     "Insight (because of -1)"):
      with self.subTest(response=response), patch.object(
          run_gpt_prompt, "debug", False), patch.object(
          run_gpt_prompt, "generate_prompt", return_value="prompt"), (
          patch.object(gpt_structure, "GPT_request", return_value=response)):
        output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
          object(), "1. alpha", 1)[0]
      self.assertEqual({}, output)

  def test_14_prompt_template_documents_1_based_requested_count_contract(self):
    template_path = (
      BACKEND_SERVER / "persona" / "prompt_template" / "v2" /
      "insight_and_evidence_v1.txt")
    template = template_path.read_text()
    generated = gpt_structure.generate_prompt(
      ["1. alpha\n2. beta\n3. gamma", "3"], str(template_path))
    self.assertIn("1-based numbered list", template)
    self.assertIn("number of requested high-level insights", template)
    self.assertIn("Infer exactly 3 high-level insights", generated)
    self.assertIn("Return only the requested insight lines", generated)
    self.assertFalse(generated.endswith("1."))
    self.assertNotIn('target persona name or "the conversation"', template)

  def run_prompt_with_response(self, response):
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="PRIVATE-PROMPT"), (
        patch.object(gpt_structure, "GPT_request", return_value=response)):
      output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. private statement", 5)[0]
    return output, run_gpt_prompt.get_last_insight_validation_diagnostics()

  def test_15_valid_output_reports_valid_without_changing_result(self):
    response = "1. Useful insight (because of 1, 3)"
    output, diagnostics = self.run_prompt_with_response(response)
    self.assertEqual({"Useful insight": [1, 3]}, output)
    self.assertEqual(1, len(diagnostics))
    self.assertEqual(("PARSED", "VALID", "VALID", 1), (
      diagnostics[0]["parser_status"],
      diagnostics[0]["validation_status"],
      diagnostics[0]["failure_category"],
      diagnostics[0]["parsed_insight_count"]))

  def test_16_parser_failure_reports_parse_failure_for_all_retries(self):
    output, diagnostics = self.run_prompt_with_response("markdown is invalid")
    self.assertEqual({}, output)
    self.assertEqual(5, len(diagnostics))
    self.assertEqual(
      ["PARSE_FAILURE"] * 5,
      [item["failure_category"] for item in diagnostics])
    self.assertTrue(all(
      item["parser_status"] == "PARSE_FAILED"
      and item["validation_status"] == "INVALID"
      for item in diagnostics))

  def test_17_empty_response_reports_empty_response(self):
    output, diagnostics = self.run_prompt_with_response("   ")
    self.assertEqual({}, output)
    self.assertEqual(
      ["EMPTY_RESPONSE"] * 5,
      [item["failure_category"] for item in diagnostics])
    self.assertTrue(all(not item["response_present"]
                        for item in diagnostics))

  def test_18_non_positive_evidence_reports_specific_category(self):
    for response in ("Insight (because of 0)",
                     "Insight (because of -1)"):
      with self.subTest(response=response):
        output, diagnostics = self.run_prompt_with_response(response)
        self.assertEqual({}, output)
        self.assertEqual(
          ["EVIDENCE_NON_POSITIVE"] * 5,
          [item["failure_category"] for item in diagnostics])

  def test_19_mapping_categories_cover_empty_blank_and_out_of_range(self):
    cases = (
      ({}, "EMPTY_MAPPING"),
      ({" ": [1]}, "BLANK_INSIGHT"),
      ({"insight": [4]}, "EVIDENCE_OUT_OF_RANGE"),
    )
    for result, category in cases:
      with self.subTest(category=category), patch.object(
          reflect_module, "run_gpt_prompt_insight_and_guidance",
          return_value=(result, [])):
        with self.assertRaises(
            reflect_module.ReflectionInsightContractError) as raised:
          reflect_module.generate_insights_and_evidence(
            object(), self.nodes(), 1)
        self.assertEqual(category, raised.exception.category)

  def test_20_diagnostics_never_contain_raw_prompt_or_response(self):
    sentinel = "PRIVATE-RAW-RESPONSE-SENTINEL"
    output, diagnostics = self.run_prompt_with_response(sentinel)
    self.assertEqual({}, output)
    encoded = json.dumps(diagnostics, sort_keys=True)
    self.assertNotIn(sentinel, encoded)
    self.assertNotIn("PRIVATE-PROMPT", encoded)
    self.assertEqual({
      "attempt_number", "requested_insight_count", "provider_outcome",
      "provider_error_type", "finish_reason", "output_token_count",
      "response_present", "response_length_chars", "parser_status",
      "parsed_insight_count", "validation_status", "failure_category", "shape",
    }, set(diagnostics[0]))
    self.assertEqual({
      "response_line_count", "nonblank_line_count", "blank_line_count",
      "numbered_line_count", "unnumbered_nonblank_line_count",
      "markdown_bullet_line_count", "code_fence_present",
      "header_like_first_line", "trailing_noncanonical_line_count",
      "canonical_line_match_count", "canonical_citation_match_count",
      "citation_candidate_count", "lines_missing_citation_count",
      "citation_line_count",
      "citation_payload_canonical_positive_integer_list_count",
      "citation_payload_with_alpha_count",
      "citation_payload_with_noncomma_punctuation_count",
      "citation_payload_with_semicolon_count",
      "citation_payload_with_colon_count",
      "citation_payload_with_hash_marker_count",
      "citation_payload_with_dash_count",
      "citation_payload_with_slash_count",
      "citation_payload_with_square_bracket_count",
      "citation_payload_digit_run_count_total",
      "citation_payload_comma_count_total",
      "citation_payload_leading_non_digit_count",
      "citation_payload_trailing_non_digit_count",
      "citation_payload_only_digits_commas_whitespace_count",
      "citation_line_with_trailing_nonwhitespace_after_parenthesis_count",
      "citation_line_with_post_parenthesis_suffix_count",
      "post_parenthesis_suffix_single_char_count",
      "post_parenthesis_suffix_multi_char_count",
      "post_parenthesis_suffix_period_count",
      "post_parenthesis_suffix_comma_count",
      "post_parenthesis_suffix_semicolon_count",
      "post_parenthesis_suffix_colon_count",
      "post_parenthesis_suffix_exclamation_count",
      "post_parenthesis_suffix_question_mark_count",
      "post_parenthesis_suffix_dash_count",
      "post_parenthesis_suffix_alpha_count",
      "post_parenthesis_suffix_digit_count",
      "post_parenthesis_suffix_other_punctuation_count",
      "post_parenthesis_suffix_whitespace_only_count",
      "parenthesis_pair_count", "because_of_literal_count",
      "json_like_wrapper_present", "yaml_like_wrapper_present",
    }, set(diagnostics[0]["shape"]))

  def test_21_retry_count_and_invalid_outcome_remain_unchanged(self):
    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="prompt"), patch.object(
        gpt_structure, "GPT_request", return_value="invalid") as request:
      output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. alpha", 5)[0]
    self.assertEqual({}, output)
    self.assertEqual(5, request.call_count)
    self.assertEqual(
      list(range(1, 6)),
      [item["attempt_number"] for item in
       run_gpt_prompt.get_last_insight_validation_diagnostics()])

  def test_22_available_length_metadata_classifies_truncated_failure(self):
    events = []

    def fake_request(*args, **kwargs):
      events.append(SimpleNamespace(
        caller_id="insight_and_guidance", finish_reason="length",
        output_tokens=300))
      return "truncated invalid output"

    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="prompt"), patch.object(
        run_gpt_prompt, "get_telemetry", side_effect=lambda: list(events)), (
        patch.object(gpt_structure, "GPT_request", side_effect=fake_request)):
      output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. alpha", 5)[0]
    diagnostics = run_gpt_prompt.get_last_insight_validation_diagnostics()
    self.assertEqual({}, output)
    self.assertEqual(["TRUNCATED_OUTPUT"] * 5,
                     [item["failure_category"] for item in diagnostics])
    self.assertTrue(all(item["finish_reason"] == "length"
                        and item["output_token_count"] == 300
                        for item in diagnostics))

  def test_23_insight_count_mismatch_is_observed_but_not_rejected(self):
    output, diagnostics = self.run_prompt_with_response(
      "Only one insight (because of 1)")
    self.assertEqual({"Only one insight": [1]}, output)
    self.assertEqual(1, diagnostics[0]["parsed_insight_count"])
    self.assertEqual(5, diagnostics[0]["requested_insight_count"])
    self.assertEqual("VALID", diagnostics[0]["failure_category"])

  def test_24_multiple_numbered_lines_parse_but_markdown_bullets_do_not(self):
    parsed = run_gpt_prompt._parse_insight_and_guidance_response(
      "1. First (because of 1)\n2. Second (because of 2, 3)")
    self.assertEqual({"First": [1], "Second": [2, 3]}, parsed)
    with self.assertRaises(
        run_gpt_prompt.InsightResponseContractError) as raised:
      run_gpt_prompt._parse_insight_and_guidance_response(
        "- First (because of 1)\n- Second (because of 2)")
    self.assertEqual("PARSE_FAILURE", raised.exception.category)

  def test_25_provider_error_is_not_mislabeled_as_model_parse_failure(self):
    events = []

    def fake_request(*args, **kwargs):
      events.append(SimpleNamespace(
        caller_id="insight_and_guidance", finish_reason=None,
        output_tokens=None, outcome="ERROR",
        error_type="ModernChatResponseValidationError"))
      return "TOKEN LIMIT EXCEEDED"

    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="prompt"), patch.object(
        run_gpt_prompt, "get_telemetry", side_effect=lambda: list(events)), (
        patch.object(gpt_structure, "GPT_request", side_effect=fake_request)):
      output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. alpha", 5)[0]
    diagnostics = run_gpt_prompt.get_last_insight_validation_diagnostics()
    self.assertEqual({}, output)
    self.assertEqual(["PROVIDER_FAILURE"] * 5,
                     [item["failure_category"] for item in diagnostics])
    self.assertTrue(all(
      item["provider_outcome"] == "ERROR"
      and item["provider_error_type"] ==
      "ModernChatResponseValidationError" for item in diagnostics))

  def test_26_prompt_requires_canonical_numbered_citation_grammar(self):
    template = (
      BACKEND_SERVER / "persona" / "prompt_template" / "v2" /
      "insight_and_evidence_v1.txt").read_text()
    canonical_example = (
      "1. <insight> (because of 1, 2, 3)\n"
      "2. <insight> (because of 4, 5)\n"
      "3. <insight> (because of 6)")
    self.assertIn(canonical_example, template)
    self.assertIn("Number every line", template)
    self.assertIn('literal lowercase phrase "because of"', template)
    self.assertIn("at least one positive 1-based statement number", template)
    self.assertIn("Separate multiple evidence numbers with commas", template)

  def test_27_prompt_forbids_noncanonical_wrappers_and_trailing_prose(self):
    template = (
      BACKEND_SERVER / "persona" / "prompt_template" / "v2" /
      "insight_and_evidence_v1.txt").read_text()
    for required_prohibition in (
        'header such as "Insights:"', "section title",
        "introductory sentence", "closing sentence",
        "explanation outside the numbered insights", "markdown code fences",
        'markdown bullets such as "- insight"', "JSON", "YAML", "tables",
        "Do not insert blank lines between insight lines",
        "Do not write anything before the first insight",
        "or after the last insight", "alternative citation syntax"):
      with self.subTest(required_prohibition=required_prohibition):
        self.assertIn(required_prohibition, template)
    for forbidden_citation in (
        "because of: 1, 2", "[1, 2]", "(1, 2)", "based on 1, 2",
        "because of statements 1, 2", "because of #1, #2"):
      with self.subTest(forbidden_citation=forbidden_citation):
        self.assertIn(forbidden_citation, template)

  def test_28_canonical_prompt_format_passes_actual_parser_and_validator(self):
    response = (
      "1. First insight (because of 1)\n"
      "2. Second insight (because of 2, 3)\n"
      "3. Third insight (because of 4)")
    expected = {
      "First insight": [1],
      "Second insight": [2, 3],
      "Third insight": [4],
    }
    self.assertEqual(
      expected,
      run_gpt_prompt._parse_insight_and_guidance_response(response))

    with patch.object(run_gpt_prompt, "debug", False), patch.object(
        run_gpt_prompt, "generate_prompt", return_value="PRIVATE-PROMPT"), (
        patch.object(gpt_structure, "GPT_request", return_value=response)):
      output = run_gpt_prompt.run_gpt_prompt_insight_and_guidance(
        object(), "1. private statement", 3)[0]
    diagnostics = run_gpt_prompt.get_last_insight_validation_diagnostics()
    self.assertEqual(expected, output)
    self.assertEqual(1, len(diagnostics))
    self.assertEqual(("PARSED", "VALID", "VALID", 3, 3), (
      diagnostics[0]["parser_status"],
      diagnostics[0]["validation_status"],
      diagnostics[0]["failure_category"],
      diagnostics[0]["parsed_insight_count"],
      diagnostics[0]["requested_insight_count"]))

  def test_29_shape_classifier_reports_perfect_canonical_response(self):
    response = (
      "1. Insight A (because of 1, 2)\n"
      "2. Insight B (because of 3)")
    shape = run_gpt_prompt._classify_insight_response_shape(response)
    self.assertEqual({
      "response_line_count": 2,
      "nonblank_line_count": 2,
      "blank_line_count": 0,
      "numbered_line_count": 2,
      "unnumbered_nonblank_line_count": 0,
      "markdown_bullet_line_count": 0,
      "code_fence_present": False,
      "header_like_first_line": False,
      "trailing_noncanonical_line_count": 0,
      "canonical_line_match_count": 2,
      "canonical_citation_match_count": 2,
      "citation_candidate_count": 2,
      "lines_missing_citation_count": 0,
      "citation_line_count": 2,
      "citation_payload_canonical_positive_integer_list_count": 2,
      "citation_payload_with_alpha_count": 0,
      "citation_payload_with_noncomma_punctuation_count": 0,
      "citation_payload_with_semicolon_count": 0,
      "citation_payload_with_colon_count": 0,
      "citation_payload_with_hash_marker_count": 0,
      "citation_payload_with_dash_count": 0,
      "citation_payload_with_slash_count": 0,
      "citation_payload_with_square_bracket_count": 0,
      "citation_payload_digit_run_count_total": 3,
      "citation_payload_comma_count_total": 1,
      "citation_payload_leading_non_digit_count": 0,
      "citation_payload_trailing_non_digit_count": 0,
      "citation_payload_only_digits_commas_whitespace_count": 2,
      "citation_line_with_trailing_nonwhitespace_after_parenthesis_count": 0,
      "citation_line_with_post_parenthesis_suffix_count": 0,
      "post_parenthesis_suffix_single_char_count": 0,
      "post_parenthesis_suffix_multi_char_count": 0,
      "post_parenthesis_suffix_period_count": 0,
      "post_parenthesis_suffix_comma_count": 0,
      "post_parenthesis_suffix_semicolon_count": 0,
      "post_parenthesis_suffix_colon_count": 0,
      "post_parenthesis_suffix_exclamation_count": 0,
      "post_parenthesis_suffix_question_mark_count": 0,
      "post_parenthesis_suffix_dash_count": 0,
      "post_parenthesis_suffix_alpha_count": 0,
      "post_parenthesis_suffix_digit_count": 0,
      "post_parenthesis_suffix_other_punctuation_count": 0,
      "post_parenthesis_suffix_whitespace_only_count": 2,
      "parenthesis_pair_count": 2,
      "because_of_literal_count": 2,
      "json_like_wrapper_present": False,
      "yaml_like_wrapper_present": False,
    }, shape)
    self.assertEqual(
      {"Insight A": [1, 2], "Insight B": [3]},
      run_gpt_prompt._parse_insight_and_guidance_response(response))

  def test_30_shape_classifier_distinguishes_noncanonical_structures(self):
    cases = (
      (
        "header",
        "Insights:\n1. Insight A (because of 1)\n"
        "2. Insight B (because of 2)",
        {"unnumbered_nonblank_line_count": 1,
         "canonical_line_match_count": 2,
         "header_like_first_line": True}),
      (
        "blank separator",
        "1. Insight A (because of 1)\n\n2. Insight B (because of 2)",
        {"blank_line_count": 1, "canonical_line_match_count": 2}),
      (
        "markdown bullets",
        "- Insight A (because of 1)\n* Insight B (because of 2)",
        {"markdown_bullet_line_count": 2,
         "unnumbered_nonblank_line_count": 2}),
      (
        "markdown fence",
        "```text\n1. Insight A (because of 1)\n"
        "2. Insight B (because of 2)\n```",
        {"code_fence_present": True,
         "canonical_line_match_count": 2,
         "trailing_noncanonical_line_count": 1}),
      (
        "missing citation",
        "1. Insight A\n2. Insight B (because of 2)",
        {"numbered_line_count": 2,
         "canonical_line_match_count": 1,
         "lines_missing_citation_count": 1}),
      (
        "alternative citations",
        "1. Insight A [because of 1, 2]\n"
        "2. Insight B (based on 1, 2)\n"
        "3. Insight C because of: 1, 2",
        {"canonical_line_match_count": 0,
         "canonical_citation_match_count": 0,
         "citation_candidate_count": 3,
         "lines_missing_citation_count": 3}),
      (
        "trailing explanation",
        "1. Insight A (because of 1)\n"
        "2. Insight B (because of 2)\n"
        "Here is why these insights matter.",
        {"unnumbered_nonblank_line_count": 1,
         "canonical_line_match_count": 2,
         "trailing_noncanonical_line_count": 1}),
    )
    for name, response, expected in cases:
      with self.subTest(name=name):
        shape = run_gpt_prompt._classify_insight_response_shape(response)
        for field, value in expected.items():
          self.assertEqual(value, shape[field])
        with self.assertRaises(run_gpt_prompt.InsightResponseContractError):
          run_gpt_prompt._parse_insight_and_guidance_response(response)

  def test_31_shape_diagnostics_never_retain_semantic_payload(self):
    response = (
      "Insights:\n1. Insight A (because of 1)\n"
      "2. Insight B (because of 2)\nPRIVATE-RAW-RESPONSE")
    output, diagnostics = self.run_prompt_with_response(response)
    self.assertEqual({}, output)
    encoded = json.dumps(diagnostics, sort_keys=True)
    for forbidden in (
        "Insight A", "Insight B", "PRIVATE-RAW-RESPONSE", "PRIVATE-PROMPT",
        "raw response", "raw prompt"):
      with self.subTest(forbidden=forbidden):
        self.assertNotIn(forbidden, encoded)

  def test_32_shape_observation_does_not_change_parser_behavior(self):
    def parser_outcome(response, observe_shape):
      original = response
      if observe_shape:
        run_gpt_prompt._classify_insight_response_shape(response)
      self.assertIs(original, response)
      try:
        return (
          "VALID",
          run_gpt_prompt._parse_insight_and_guidance_response(response))
      except run_gpt_prompt.InsightResponseContractError as error:
        return ("INVALID", error.category, error.parsed_insight_count)

    responses = (
      "1. Insight A (because of 1)\n2. Insight B (because of 2)",
      "Insights:\n1. Insight A (because of 1)",
      "1. Insight A (because of 1)\n\n2. Insight B (because of 2)",
      "- Insight A (because of 1)",
      "1. Insight A [because of 1]",
      "1. Insight A (because of 1)\nTrailing explanation.",
    )
    for response in responses:
      with self.subTest(response_shape=len(response.splitlines())):
        self.assertEqual(
          parser_outcome(response, False), parser_outcome(response, True))

  def test_33_citation_token_classifier_accepts_parser_spacing_and_signs(self):
    response = (
      "1. Example A (because of 1)\n"
      "2. Example B (because of +2,3)\n"
      "3. Example C (because of 4 ,  5)\n"
      "4. Example D (because of\t6,\t+7)")
    shape = run_gpt_prompt._classify_insight_response_shape(response)
    self.assertEqual(4, shape["citation_line_count"])
    self.assertEqual(
      4,
      shape["citation_payload_canonical_positive_integer_list_count"])
    self.assertEqual(4, shape["canonical_citation_match_count"])
    self.assertEqual(4, shape["canonical_line_match_count"])
    self.assertEqual(7, shape["citation_payload_digit_run_count_total"])
    self.assertEqual(3, shape["citation_payload_comma_count_total"])
    self.assertEqual(
      2, shape["citation_payload_only_digits_commas_whitespace_count"])
    self.assertEqual(
      {"Example A": [1], "Example B": [2, 3],
       "Example C": [4, 5], "Example D": [6, 7]},
      run_gpt_prompt._parse_insight_and_guidance_response(response))

  def test_34_citation_token_classifier_distinguishes_bounded_token_classes(self):
    cases = (
      ("alpha", "1. Example (because of statements 1, 2)",
       {"citation_payload_with_alpha_count": 1}),
      ("semicolon", "1. Example (because of 1; 2; 3)",
       {"citation_payload_with_semicolon_count": 1}),
      ("colon", "1. Example (because of 1: 2)",
       {"citation_payload_with_colon_count": 1}),
      ("hash", "1. Example (because of #1, #2)",
       {"citation_payload_with_hash_marker_count": 1,
        "citation_payload_leading_non_digit_count": 1}),
      ("dash", "1. Example (because of 1-3)",
       {"citation_payload_with_dash_count": 1}),
      ("slash", "1. Example (because of 1/2)",
       {"citation_payload_with_slash_count": 1}),
      ("square bracket", "1. Example (because of [1, 2])",
       {"citation_payload_with_square_bracket_count": 1,
        "citation_payload_leading_non_digit_count": 1,
        "citation_payload_trailing_non_digit_count": 1}),
    )
    for name, response, expected in cases:
      with self.subTest(name=name):
        shape = run_gpt_prompt._classify_insight_response_shape(response)
        self.assertEqual(1, shape["citation_line_count"])
        self.assertEqual(
          0,
          shape["citation_payload_canonical_positive_integer_list_count"])
        self.assertEqual(
          1, shape["citation_payload_with_noncomma_punctuation_count"]
          if name != "alpha" else shape["citation_payload_with_alpha_count"])
        for field, value in expected.items():
          self.assertEqual(value, shape[field])
        with self.assertRaises(run_gpt_prompt.InsightResponseContractError):
          run_gpt_prompt._parse_insight_and_guidance_response(response)

  def test_35_citation_token_classifier_detects_post_parenthesis_structure(self):
    response = "1. Example (because of 1, 2)."
    shape = run_gpt_prompt._classify_insight_response_shape(response)
    self.assertEqual(
      1, shape["citation_payload_canonical_positive_integer_list_count"])
    self.assertEqual(
      1,
      shape[
        "citation_line_with_trailing_nonwhitespace_after_parenthesis_count"])
    self.assertEqual(
      1, shape["citation_line_with_post_parenthesis_suffix_count"])
    self.assertEqual(1, shape["post_parenthesis_suffix_single_char_count"])
    self.assertEqual(1, shape["post_parenthesis_suffix_period_count"])
    self.assertEqual(
      {"Example": [1, 2]},
      run_gpt_prompt._parse_insight_and_guidance_response(response))

  def test_36_citation_token_diagnostics_do_not_retain_payload_or_numbers(self):
    response = "1. PRIVATE-INSIGHT (because of PRIVATE-WORD 701, #809)"
    shape = run_gpt_prompt._classify_insight_response_shape(response)
    encoded = json.dumps(shape, sort_keys=True)
    for forbidden in (
        "PRIVATE-INSIGHT", "PRIVATE-WORD", "701", "809",
        "raw response", "raw prompt", "citation substring"):
      with self.subTest(forbidden=forbidden):
        self.assertNotIn(forbidden, encoded)

  def test_37_citation_token_observation_does_not_change_parser_behavior(self):
    def parser_outcome(response, observe_tokens):
      original = response
      if observe_tokens:
        run_gpt_prompt._classify_insight_response_shape(response)
      self.assertIs(original, response)
      try:
        return (
          "VALID",
          run_gpt_prompt._parse_insight_and_guidance_response(response))
      except run_gpt_prompt.InsightResponseContractError as error:
        return ("INVALID", error.category, error.parsed_insight_count)

    for response in (
        "1. Example (because of 1, 2)",
        "1. Example (because of statements 1, 2)",
        "1. Example (because of 1; 2)",
        "1. Example (because of #1, #2)",
        "1. Example (because of [1, 2])",
        "1. Example (because of 1, 2)."):
      with self.subTest(response_length=len(response)):
        self.assertEqual(
          parser_outcome(response, False), parser_outcome(response, True))

  def test_38_post_parenthesis_suffix_classifier_covers_bounded_classes(self):
    cases = (
      ("none", "1. Example (because of 1, 2)",
       {"citation_line_with_post_parenthesis_suffix_count": 0,
        "post_parenthesis_suffix_whitespace_only_count": 1}),
      ("period", "1. Example (because of 1, 2).",
       {"post_parenthesis_suffix_single_char_count": 1,
        "post_parenthesis_suffix_period_count": 1}),
      ("comma", "1. Example (because of 1, 2),",
       {"post_parenthesis_suffix_comma_count": 1}),
      ("semicolon", "1. Example (because of 1, 2);",
       {"post_parenthesis_suffix_semicolon_count": 1}),
      ("colon", "1. Example (because of 1, 2):",
       {"post_parenthesis_suffix_colon_count": 1}),
      ("exclamation", "1. Example (because of 1, 2)!",
       {"post_parenthesis_suffix_exclamation_count": 1}),
      ("question", "1. Example (because of 1, 2)?",
       {"post_parenthesis_suffix_question_mark_count": 1}),
      ("dash", "1. Example (because of 1, 2)-",
       {"post_parenthesis_suffix_dash_count": 1}),
      ("alpha", "1. Example (because of 1, 2) PRIVATE-SUFFIX",
       {"post_parenthesis_suffix_alpha_count": 1,
        "post_parenthesis_suffix_multi_char_count": 1}),
      ("digit", "1. Example (because of 1, 2) 701",
       {"post_parenthesis_suffix_digit_count": 1,
        "post_parenthesis_suffix_multi_char_count": 1}),
      ("other punctuation", "1. Example (because of 1, 2)#",
       {"post_parenthesis_suffix_other_punctuation_count": 1}),
      ("multiple", "1. Example (because of 1, 2).!",
       {"post_parenthesis_suffix_multi_char_count": 1,
        "post_parenthesis_suffix_period_count": 1,
        "post_parenthesis_suffix_exclamation_count": 1}),
    )
    for name, response, expected in cases:
      with self.subTest(name=name):
        shape = run_gpt_prompt._classify_insight_response_shape(response)
        for field, value in expected.items():
          self.assertEqual(value, shape[field])
        if name in ("none", "period"):
          self.assertEqual(
            {"Example": [1, 2]},
            run_gpt_prompt._parse_insight_and_guidance_response(response))
        else:
          self.assertEqual(
            1, shape["citation_line_with_post_parenthesis_suffix_count"])
          with self.assertRaises(run_gpt_prompt.InsightResponseContractError):
            run_gpt_prompt._parse_insight_and_guidance_response(response)

  def test_39_post_parenthesis_suffix_diagnostics_are_privacy_safe(self):
    response = "1. PRIVATE-INSIGHT (because of 701) PRIVATE-SUFFIX"
    shape = run_gpt_prompt._classify_insight_response_shape(response)
    encoded = json.dumps(shape, sort_keys=True)
    for forbidden in (
        "PRIVATE-INSIGHT", "PRIVATE-SUFFIX", "701", "raw prompt",
        "raw response", "suffix substring", "citation payload"):
      with self.subTest(forbidden=forbidden):
        self.assertNotIn(forbidden, encoded)

  def test_40_post_parenthesis_suffix_observation_preserves_behavior(self):
    def parser_outcome(response, observe_suffix):
      original = response
      if observe_suffix:
        run_gpt_prompt._classify_insight_response_shape(response)
      self.assertIs(original, response)
      try:
        return (
          "VALID",
          run_gpt_prompt._parse_insight_and_guidance_response(response))
      except run_gpt_prompt.InsightResponseContractError as error:
        return ("INVALID", error.category, error.parsed_insight_count)

    for response in (
        "1. Example (because of 1, 2)",
        "1. Example (because of 1, 2).",
        "1. Example (because of 1, 2),",
        "1. Example (because of 1, 2) PRIVATE-SUFFIX",
        "1. Example (because of 1, 2) 701",
        "1. Example (because of 1, 2).!"):
      with self.subTest(response_length=len(response)):
        self.assertEqual(
          parser_outcome(response, False), parser_outcome(response, True))

  def test_41_optional_period_preserves_parsed_semantics(self):
    without_period = "1. Insight A (because of 1, 2)"
    with_period = "1. Insight A (because of 1, 2)."
    expected = {"Insight A": [1, 2]}
    self.assertEqual(
      expected,
      run_gpt_prompt._parse_insight_and_guidance_response(without_period))
    self.assertEqual(
      expected,
      run_gpt_prompt._parse_insight_and_guidance_response(with_period))

  def test_42_period_compatibility_is_optional_per_line(self):
    all_periods = (
      "1. Insight A (because of 1, 2).\n"
      "2. Insight B (because of 3).\n"
      "3. Insight C (because of 4, 5).")
    mixed = (
      "1. Insight A (because of 1).\n"
      "2. Insight B (because of 2)\n"
      "3. Insight C (because of 3).")
    self.assertEqual({
      "Insight A": [1, 2], "Insight B": [3], "Insight C": [4, 5]},
      run_gpt_prompt._parse_insight_and_guidance_response(all_periods))
    self.assertEqual({
      "Insight A": [1], "Insight B": [2], "Insight C": [3]},
      run_gpt_prompt._parse_insight_and_guidance_response(mixed))

  def test_43_period_is_the_only_newly_accepted_suffix(self):
    accepted = (
      "1. Insight A (because of 1)",
      "1. Insight A (because of 1).",
      "1. Insight A (because of 1).   ",
    )
    rejected = (
      "1. Insight A (because of 1)..",
      "1. Insight A (because of 1)!",
      "1. Insight A (because of 1)?",
      "1. Insight A (because of 1),",
      "1. Insight A (because of 1);",
      "1. Insight A (because of 1):",
      "1. Insight A (because of 1)-",
      "1. Insight A (because of 1) note",
      "1. Insight A (because of 1)2",
      "1. Insight A (because of 1).!",
    )
    for response in accepted:
      with self.subTest(accepted_length=len(response)):
        self.assertEqual(
          {"Insight A": [1]},
          run_gpt_prompt._parse_insight_and_guidance_response(response))
    for response in rejected:
      with self.subTest(rejected_length=len(response)):
        with self.assertRaises(run_gpt_prompt.InsightResponseContractError):
          run_gpt_prompt._parse_insight_and_guidance_response(response)

  def test_44_period_compatibility_preserves_evidence_validation(self):
    for response, category in (
        ("1. Insight (because of 0).", "EVIDENCE_NON_POSITIVE"),
        ("1. Insight (because of -1).", "EVIDENCE_NON_POSITIVE"),
        ("1. Insight (because of ).", "PARSE_FAILURE"),
        ("1. Insight (because of 1; 2).", "PARSE_FAILURE")):
      with self.subTest(category=category):
        with self.assertRaises(
            run_gpt_prompt.InsightResponseContractError) as raised:
          run_gpt_prompt._parse_insight_and_guidance_response(response)
        self.assertEqual(category, raised.exception.category)

    parsed = run_gpt_prompt._parse_insight_and_guidance_response(
      "1. Insight (because of 4).")
    with patch.object(
        reflect_module, "run_gpt_prompt_insight_and_guidance",
        return_value=(parsed, [])):
      with self.assertRaises(
          reflect_module.ReflectionInsightContractError) as raised:
        reflect_module.generate_insights_and_evidence(
          object(), self.nodes(), 1)
    self.assertEqual("EVIDENCE_OUT_OF_RANGE", raised.exception.category)

  def test_45_period_form_is_valid_and_period_remains_observable(self):
    response = "1. Insight A (because of 1, 2)."
    output, diagnostics = self.run_prompt_with_response(response)
    self.assertEqual({"Insight A": [1, 2]}, output)
    self.assertEqual(1, len(diagnostics))
    self.assertEqual(("PARSED", "VALID", "VALID"), (
      diagnostics[0]["parser_status"],
      diagnostics[0]["validation_status"],
      diagnostics[0]["failure_category"]))
    shape = diagnostics[0]["shape"]
    self.assertEqual(1, shape["canonical_line_match_count"])
    self.assertEqual(1, shape["canonical_citation_match_count"])
    self.assertFalse(shape["header_like_first_line"])
    self.assertEqual(0, shape["lines_missing_citation_count"])
    self.assertEqual(1, shape["post_parenthesis_suffix_period_count"])

  def test_46_period_form_preserves_node_id_lineage_to_add_thought(self):
    memory = SimpleNamespace(add_thought=Mock())
    persona = SimpleNamespace(
      a_mem=memory,
      scratch=SimpleNamespace(curr_time=datetime.datetime(2026, 1, 1)))
    nodes = self.nodes()

    def period_prompt(*args, **kwargs):
      parsed = run_gpt_prompt._parse_insight_and_guidance_response(
        "1. Period insight (because of 1, 3).")
      return parsed, []

    with patch.object(
        reflect_module, "generate_focal_points", return_value=["focus"]), (
        patch.object(reflect_module, "new_retrieve",
                     return_value={"focus": nodes})), patch.object(
        reflect_module, "run_gpt_prompt_insight_and_guidance",
        side_effect=period_prompt), patch.object(
        reflect_module, "generate_action_event_triple",
        return_value=("actor", "reflects", "insight")), patch.object(
        reflect_module, "generate_poig_score", return_value=7), patch.object(
        reflect_module, "get_embedding", return_value=[0.1, 0.2]):
      reflect_module.run_reflect(persona)
    self.assertEqual(["node_A", "node_C"], memory.add_thought.call_args.args[-1])

  def test_47_invalid_suffix_preserves_no_write_invariant(self):
    memory = SimpleNamespace(add_thought=Mock())
    persona = SimpleNamespace(a_mem=memory)
    nodes = self.nodes()

    def invalid_suffix_prompt(*args, **kwargs):
      with self.assertRaises(run_gpt_prompt.InsightResponseContractError):
        run_gpt_prompt._parse_insight_and_guidance_response(
          "1. Invalid suffix (because of 1)!")
      return {}, []

    with patch.object(
        reflect_module, "generate_focal_points", return_value=["focus"]), (
        patch.object(reflect_module, "new_retrieve",
                     return_value={"focus": nodes})), patch.object(
        reflect_module, "run_gpt_prompt_insight_and_guidance",
        side_effect=invalid_suffix_prompt):
      with self.assertRaises(reflect_module.ReflectionInsightContractError):
        reflect_module.run_reflect(persona)
    memory.add_thought.assert_not_called()


if __name__ == "__main__":
  unittest.main()
