"""Offline contract tests against the real modern OpenAI Python SDK.

Run only in ``.venv-modern-contract``.  Every SDK request is intercepted by
``httpx.MockTransport`` before DNS or socket access is possible.
"""
import base64
from importlib.metadata import version
import json
from pathlib import Path
import socket
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import openai
from openai import OpenAI


BACKEND_SERVER = Path(__file__).resolve().parents[1]
if str(BACKEND_SERVER) not in sys.path:
  sys.path.insert(0, str(BACKEND_SERVER))

from persona.prompt_template.llm_provider_config import modern_openai_config
from persona.prompt_template.modern_openai_provider import (
  LLMAuthenticationError,
  LLMAuthorizationError,
  LLMConnectionError,
  LLMEmptyOutputError,
  LLMIncompleteResponseError,
  LLMInvalidRequestError,
  LLMModelNotFoundError,
  LLMRateLimitError,
  LLMRefusalError,
  LLMServerError,
  LLMTimeoutError,
  LLMUnsupportedParameterError,
  ModernOpenAIClientAdapter,
  ModernOpenAIProvider,
)


CONTRACT_OPENAI_VERSION = "2.53.0"
CONTRACT_HTTPX_VERSION = "0.28.1"
DUMMY_API_KEY = "sk-test-modern-contract-validation"
CONTRACT_HOST = "contract.invalid"
BASE_URL = f"https://{CONTRACT_HOST}/v1"


def _chat_response(content="contract output", finish_reason="stop",
                   refusal=None, usage_marker="standard"):
  response = {
    "id": "chatcmpl-contract",
    "object": "chat.completion",
    "created": 1720000000,
    "model": "gpt-3.5-turbo",
    "choices": [{
      "index": 0,
      "message": {
        "role": "assistant",
        "content": content,
        "refusal": refusal,
      },
      "finish_reason": finish_reason,
      "logprobs": None,
    }],
  }
  if usage_marker == "standard":
    response["usage"] = {
      "prompt_tokens": 11,
      "completion_tokens": 7,
      "total_tokens": 18,
      "prompt_tokens_details": {
        "cached_tokens": 3,
        "audio_tokens": 0,
      },
      "completion_tokens_details": {
        "reasoning_tokens": 2,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
      },
    }
  elif usage_marker != "absent":
    response["usage"] = usage_marker
  return response


def _embedding_response(vector=(0.25, -0.5, 1.0), usage_marker="standard"):
  encoded = base64.b64encode(
    struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
  response = {
    "object": "list",
    "data": [{
      "object": "embedding",
      "index": 0,
      "embedding": encoded,
    }],
    "model": "text-embedding-ada-002",
  }
  if usage_marker == "standard":
    response["usage"] = {"prompt_tokens": 4, "total_tokens": 4}
  elif usage_marker != "absent":
    response["usage"] = usage_marker
  return response


def _error_response(status, code=None):
  return httpx.Response(status, json={
    "error": {
      "message": "synthetic contract error",
      "type": "invalid_request_error",
      "param": None,
      "code": code,
    },
  })


class ModernOpenAISDKContractTests(unittest.TestCase):
  def setUp(self):
    if version("openai") != CONTRACT_OPENAI_VERSION:
      self.fail(
        f"Contract suite requires openai=={CONTRACT_OPENAI_VERSION}")
    if version("httpx") != CONTRACT_HTTPX_VERSION:
      self.fail(f"Contract suite requires httpx=={CONTRACT_HTTPX_VERSION}")
    self.requests = []
    self.clients = []
    self.socket_patch = patch.object(
      socket, "create_connection",
      side_effect=AssertionError("real socket access attempted"))
    self.dns_patch = patch.object(
      socket, "getaddrinfo",
      side_effect=AssertionError("real DNS access attempted"))
    self.socket_patch.start()
    self.dns_patch.start()

  def tearDown(self):
    for client in reversed(self.clients):
      client.close()
    self.dns_patch.stop()
    self.socket_patch.stop()

  def make_client(self, handler, timeout=17.5):
    def guarded_handler(request):
      self.assertEqual(CONTRACT_HOST, request.url.host)
      body = json.loads(request.content) if request.content else None
      self.requests.append({
        "method": request.method,
        "path": request.url.path,
        "body": body,
      })
      return handler(request)

    http_client = httpx.Client(
      transport=httpx.MockTransport(guarded_handler))
    client = OpenAI(
      api_key=DUMMY_API_KEY,
      base_url=BASE_URL,
      max_retries=0,
      timeout=timeout,
      http_client=http_client,
    )
    self.clients.append(client)
    return client

  def adapter(self, handler, timeout=17.5):
    config = modern_openai_config(request_timeout_seconds=timeout)
    return ModernOpenAIClientAdapter(
      config=config, client=self.make_client(handler, timeout))

  def test_real_sdk_and_httpx_versions_are_exact(self):
    self.assertEqual(CONTRACT_OPENAI_VERSION, openai.__version__)
    self.assertEqual(CONTRACT_HTTPX_VERSION, httpx.__version__)

  def test_client_construction_accepts_contract_options_without_request(self):
    client = self.make_client(
      lambda request: self.fail("construction performed an HTTP request"))
    self.assertEqual(0, client.max_retries)
    self.assertEqual(17.5, client.timeout)
    self.assertEqual([], self.requests)

  def test_chat_request_and_real_response_object_match_adapter(self):
    def handler(request):
      return httpx.Response(
        200,
        headers={"x-request-id": "req-chat-contract"},
        json=_chat_response())

    adapter = self.adapter(handler)
    messages = [{"role": "user", "content": "contract prompt"}]
    result = adapter.create_chat(model="gpt-3.5-turbo", messages=messages)

    self.assertEqual([{
      "method": "POST",
      "path": "/v1/chat/completions",
      "body": {
        "messages": messages,
        "model": "gpt-3.5-turbo",
        "store": False,
      },
    }], self.requests)
    self.assertEqual("contract output", result.text)
    self.assertEqual("gpt-3.5-turbo", result.model)
    self.assertEqual("req-chat-contract", result.request_id)
    self.assertEqual("stop", result.finish_reason)
    self.assertEqual((11, 7, 3, 2), (
      result.usage.input_tokens,
      result.usage.output_tokens,
      result.usage.cached_input_tokens,
      result.usage.reasoning_tokens,
    ))

  def test_provider_preserves_legacy_chat_shape_with_real_sdk(self):
    adapter = self.adapter(lambda request: httpx.Response(
      200, headers={"x-request-id": "req-provider"},
      json=_chat_response()))
    provider = ModernOpenAIProvider(
      modern_openai_config(request_timeout_seconds=17.5), adapter)

    result = provider.chat_completion(
      model="gpt-3.5-turbo",
      messages=[{"role": "user", "content": "contract prompt"}])

    self.assertEqual(
      "contract output", result["choices"][0]["message"]["content"])
    self.assertEqual("req-provider",
                     provider.consume_response_metadata().request_id)

  def test_request_id_is_none_when_header_is_absent(self):
    adapter = self.adapter(
      lambda request: httpx.Response(200, json=_chat_response()))
    result = adapter.create_chat(model="gpt-3.5-turbo", messages=[])
    self.assertIsNone(result.request_id)

  def test_real_usage_objects_cover_optional_shapes(self):
    usage_shapes = (
      ({
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
      }, (4, 2, None, None)),
      ({
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
      }, (4, 2, None, None)),
      (None, (None, None, None, None)),
      ("absent", (None, None, None, None)),
    )
    for usage_marker, expected in usage_shapes:
      with self.subTest(usage_marker=usage_marker):
        self.requests.clear()
        adapter = self.adapter(lambda request, marker=usage_marker:
          httpx.Response(200, json=_chat_response(usage_marker=marker)))
        usage = adapter.create_chat(
          model="gpt-3.5-turbo", messages=[]).usage
        self.assertEqual(expected, (
          usage.input_tokens,
          usage.output_tokens,
          usage.cached_input_tokens,
          usage.reasoning_tokens,
        ))

  def test_real_sdk_builds_typed_chat_and_usage_objects(self):
    client = self.make_client(lambda request: httpx.Response(
      200,
      headers={"x-request-id": "req-typed-contract"},
      json=_chat_response()))

    response = client.chat.completions.create(
      model="gpt-3.5-turbo", messages=[])

    self.assertEqual("ChatCompletion", type(response).__name__)
    self.assertEqual("CompletionUsage", type(response.usage).__name__)
    self.assertEqual(
      "PromptTokensDetails",
      type(response.usage.prompt_tokens_details).__name__)
    self.assertEqual(
      "CompletionTokensDetails",
      type(response.usage.completion_tokens_details).__name__)
    self.assertEqual("req-typed-contract", response._request_id)

  def test_real_sdk_response_states_map_to_distinct_errors(self):
    cases = (
      (_chat_response(refusal="refused"), LLMRefusalError),
      (_chat_response(content=None), LLMEmptyOutputError),
      (_chat_response(content=""), LLMEmptyOutputError),
      (_chat_response(finish_reason="length"),
       LLMIncompleteResponseError),
      (_chat_response(finish_reason="content_filter"),
       LLMIncompleteResponseError),
    )
    for response, expected_error in cases:
      with self.subTest(expected_error=expected_error.__name__):
        self.requests.clear()
        adapter = self.adapter(
          lambda request, payload=response: httpx.Response(
            200, json=payload))
        with self.assertRaises(expected_error):
          adapter.create_chat(model="gpt-3.5-turbo", messages=[])

  def test_embedding_request_and_base64_response_match_adapter(self):
    def handler(request):
      return httpx.Response(
        200,
        headers={"x-request-id": "req-embedding-contract"},
        json=_embedding_response())

    adapter = self.adapter(handler)
    result = adapter.create_embedding(
      model="text-embedding-ada-002", input=["normalized input"])

    self.assertEqual("/v1/embeddings", self.requests[0]["path"])
    self.assertEqual("text-embedding-ada-002",
                     self.requests[0]["body"]["model"])
    self.assertEqual(["normalized input"], self.requests[0]["body"]["input"])
    self.assertNotIn("dimensions", self.requests[0]["body"])
    self.assertEqual("base64", self.requests[0]["body"]["encoding_format"])
    self.assertEqual((0.25, -0.5, 1.0), result.vector)
    self.assertEqual("text-embedding-ada-002", result.model)
    self.assertEqual("req-embedding-contract", result.request_id)
    self.assertEqual(4, result.usage.input_tokens)

  def test_provider_preserves_legacy_embedding_shape_with_real_sdk(self):
    adapter = self.adapter(lambda request: httpx.Response(
      200, headers={"x-request-id": "req-provider-embedding"},
      json=_embedding_response()))
    provider = ModernOpenAIProvider(
      modern_openai_config(request_timeout_seconds=17.5), adapter)
    result = provider.embedding(
      input=["normalized input"], model="text-embedding-ada-002")
    self.assertEqual([0.25, -0.5, 1.0], result["data"][0]["embedding"])

  def test_real_sdk_http_error_classes(self):
    cases = (
      (400, openai.BadRequestError),
      (401, openai.AuthenticationError),
      (403, openai.PermissionDeniedError),
      (404, openai.NotFoundError),
      (422, openai.UnprocessableEntityError),
      (429, openai.RateLimitError),
      (500, openai.InternalServerError),
      (503, openai.InternalServerError),
    )
    for status, expected_error in cases:
      with self.subTest(status=status):
        self.requests.clear()
        client = self.make_client(
          lambda request, value=status: _error_response(value))
        with self.assertRaises(expected_error):
          client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[])
        self.assertEqual(1, len(self.requests))

  def test_real_sdk_errors_map_to_application_hierarchy(self):
    cases = (
      (400, None, LLMInvalidRequestError),
      (400, "unsupported_parameter", LLMUnsupportedParameterError),
      (422, None, LLMInvalidRequestError),
      (422, "unsupported_value", LLMUnsupportedParameterError),
      (401, None, LLMAuthenticationError),
      (403, None, LLMAuthorizationError),
      (404, None, LLMModelNotFoundError),
      (429, None, LLMRateLimitError),
      (500, None, LLMServerError),
      (503, None, LLMServerError),
    )
    for status, code, expected_error in cases:
      with self.subTest(status=status, code=code):
        self.requests.clear()
        adapter = self.adapter(
          lambda request, value=status, error_code=code:
            _error_response(value, error_code))
        with self.assertRaises(expected_error):
          adapter.create_chat(model="gpt-3.5-turbo", messages=[])
        self.assertEqual(1, len(self.requests))

  def test_retryable_http_errors_are_not_retried_by_real_sdk(self):
    for status in (429, 500):
      with self.subTest(status=status):
        self.requests.clear()
        adapter = self.adapter(
          lambda request, value=status: _error_response(value))
        with self.assertRaises((LLMRateLimitError, LLMServerError)):
          adapter.create_chat(model="gpt-3.5-turbo", messages=[])
        self.assertEqual(1, len(self.requests))

  def test_connection_failure_is_single_attempt_and_normalized(self):
    def handler(request):
      raise httpx.ConnectError("synthetic connection failure", request=request)

    adapter = self.adapter(handler)
    with self.assertRaises(LLMConnectionError):
      adapter.create_chat(model="gpt-3.5-turbo", messages=[])
    self.assertEqual(1, len(self.requests))

  def test_timeout_is_single_attempt_and_normalized(self):
    def handler(request):
      raise httpx.ReadTimeout("synthetic timeout", request=request)

    adapter = self.adapter(handler, timeout=9.25)
    self.assertEqual(9.25, adapter.client.timeout)
    with self.assertRaises(LLMTimeoutError):
      adapter.create_chat(model="gpt-3.5-turbo", messages=[])
    self.assertEqual(1, len(self.requests))

  def test_all_requests_use_mock_transport_and_no_responses_endpoint(self):
    adapter = self.adapter(lambda request: httpx.Response(
      200, json=_chat_response()))
    adapter.create_chat(model="gpt-3.5-turbo", messages=[])
    self.assertEqual(["/v1/chat/completions"], [
      request["path"] for request in self.requests])
    self.assertFalse(any("responses" in request["path"]
                         for request in self.requests))


if __name__ == "__main__":
  unittest.main()
