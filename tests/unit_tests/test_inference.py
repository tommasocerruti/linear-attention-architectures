# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
import unittest.mock

import numpy as np
import pytest
import torch

pytest.importorskip("flask")
pytest.importorskip("flask_restful")

from megatron.core.inference.text_generation_server import MegatronServer
from megatron.core.tokenizers import MegatronTokenizer
from tests.unit_tests.inference.engines.test_static_engine import StaticInferenceEngineTestHarness
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="module")
def gpt2_tiktoken_tokenizer():
    return MegatronTokenizer.from_pretrained(
        tokenizer_path="/opt/data/tokenizers/tiktoken/tiktoken.vocab.json",
        vocab_size=131072,
        num_special_tokens=1000,
        pattern="v1",
    )


@pytest.fixture(scope="module")
def static_inference_engine(gpt2_tiktoken_tokenizer):
    engine_wrapper = StaticInferenceEngineTestHarness()
    engine_wrapper.setup_engine(vocab_size=gpt2_tiktoken_tokenizer.vocab_size, legacy=True)

    controller = engine_wrapper.static_engine.controller
    controller.tokenizer = gpt2_tiktoken_tokenizer

    def mock_forward(*args, **kwargs):
        tokens = args[0]
        B, L = tokens.shape
        assert B == 1, "Test assumes batch_size == 1"
        V = gpt2_tiktoken_tokenizer.vocab_size
        next_token_idxs = tokens[0, 1:]
        logits = torch.zeros(1, L, V, dtype=torch.float32, device=tokens.device)
        logits[0, torch.arange(L - 1), next_token_idxs] = 100
        logits[0, -1, gpt2_tiktoken_tokenizer.eos] = 100
        return logits

    controller.inference_wrapped_model.model.forward = mock_forward
    yield engine_wrapper.static_engine


@pytest.fixture(scope="module")
def app(static_inference_engine):
    return MegatronServer(static_inference_engine).app


@pytest.fixture()
def client(app):
    return app.test_client()


@unittest.mock.patch(
    'megatron.core.inference.text_generation_server.text_generation_server.send_do_generate'
)
def test_generations_endpoint(mock_send_do_generate, client, gpt2_tiktoken_tokenizer):
    Utils.initialize_distributed()

    prompts = ["twinkle twinkle little star, how I wonder what you are"]
    request_data = {"prompts": prompts, "tokens_to_generate": 10, "logprobs": True}

    response = client.put('/api', json=request_data)

    assert response.status_code == 200
    assert response.is_json
    json_data = response.get_json()
    assert 'text' in json_data
    assert 'logprobs' in json_data
    assert len(json_data['text']) == len(prompts)
    assert len(json_data['logprobs']) == len(prompts)

    # Verify that beam search does not work
    request_data["beam_width"] = 1
    response = client.put('/api', json=request_data)
    assert response.status_code == 400  # Bad Request

    mock_send_do_generate.assert_called_once()


@unittest.mock.patch(
    "megatron.core.inference.text_generation_server.endpoints.completions.send_do_generate"
)
def test_completions_endpoint(mock_send_do_generate, client, gpt2_tiktoken_tokenizer):
    Utils.initialize_distributed()

    twinkle = ("twinkle twinkle little star,", " how I wonder what you are")
    request_data = {"prompt": twinkle[0] + twinkle[1], "max_tokens": 0, "logprobs": 5, "echo": True}

    response = client.post('/completions', json=request_data)

    assert response.status_code == 200
    assert response.is_json

    json_data = response.get_json()
    assert 'choices' in json_data
    assert len(json_data['choices']) > 0
    assert 'text' in json_data['choices'][0]
    assert 'logprobs' in json_data['choices'][0]

    # whats up with the reconstruction of the prompt?
    # we are replicating what lm-eval-harness::TemplateLM::_encode_pair does
    # it encodes prompt, then prompt+suffix, and then infers the suffix tokens
    # from the combined encoding.
    logprobs = json_data["choices"][0]["logprobs"]
    num_reconstructed_prompt_tokens = np.searchsorted(logprobs["text_offset"], len(twinkle[0]))
    assert num_reconstructed_prompt_tokens == len(gpt2_tiktoken_tokenizer.tokenize(twinkle[0]))
    suffix_logprob = logprobs["token_logprobs"][num_reconstructed_prompt_tokens:]

    # we mock logits to be 0 everywhere, and 100 at gt tokens, so logprob should be 0 for gt tokens
    assert sum(suffix_logprob) == 0, f"{suffix_logprob} != [0, .... 0]"

    # Test for unsupported HTTP methods
    response = client.put('/completions', json=request_data)
    assert response.status_code == 405  # Method Not Allowed

    mock_send_do_generate.assert_called_once()


@unittest.mock.patch(
    "megatron.core.inference.text_generation_server.endpoints.completions.send_do_generate"
)
def test_completions_endpoint_supports_lm_eval_loglikelihood(
    mock_send_do_generate, client, gpt2_tiktoken_tokenizer
):
    Utils.initialize_distributed()

    twinkle = ("twinkle twinkle little star,", " how I wonder what you are")
    request_data = {"prompt": twinkle[0] + twinkle[1], "max_tokens": 1, "logprobs": 1, "echo": True}

    response = client.post('/completions', json=request_data)

    assert response.status_code == 200
    assert response.is_json

    json_data = response.get_json()
    logprobs = json_data["choices"][0]["logprobs"]
    num_reconstructed_prompt_tokens = np.searchsorted(logprobs["text_offset"], len(twinkle[0]))

    # lm-eval sums logprobs over the continuation tokens and drops the final generated token.
    suffix_logprob = logprobs["token_logprobs"][num_reconstructed_prompt_tokens:-1]
    assert sum(suffix_logprob) == 0, f"{suffix_logprob} != [0, .... 0]"

    mock_send_do_generate.assert_called_once()


@unittest.mock.patch(
    "megatron.core.inference.text_generation_server.endpoints.completions.send_do_generate"
)
def test_completions_endpoint_lm_eval_slice_excludes_context(
    mock_send_do_generate, client, static_inference_engine, gpt2_tiktoken_tokenizer
):
    Utils.initialize_distributed()

    context = "twinkle twinkle little star,"
    continuation = " how I wonder what you are"
    request_data = {"prompt": context + continuation, "max_tokens": 1, "logprobs": 1, "echo": True}

    original_forward = static_inference_engine.controller.inference_wrapped_model.model.forward
    context_token_count = len(gpt2_tiktoken_tokenizer.tokenize(context))
    full_token_count = len(gpt2_tiktoken_tokenizer.tokenize(context + continuation))
    continuation_token_count = full_token_count - context_token_count

    def mock_forward(*args, **kwargs):
        tokens = args[0]
        batch_size, sequence_length = tokens.shape
        assert batch_size == 1, "Test assumes batch_size == 1"
        vocab_size = gpt2_tiktoken_tokenizer.vocab_size
        logits = torch.zeros(
            1, sequence_length, vocab_size, dtype=torch.float32, device=tokens.device
        )

        # Make context-token logprobs bad and continuation-token logprobs good.
        for position, next_token in enumerate(tokens[0, 1:].tolist()):
            if context_token_count - 1 <= position < full_token_count - 1:
                logits[0, position, next_token] = 100
        logits[0, -1, gpt2_tiktoken_tokenizer.eos] = 100
        return logits

    static_inference_engine.controller.inference_wrapped_model.model.forward = mock_forward
    try:
        response = client.post('/completions', json=request_data)
    finally:
        static_inference_engine.controller.inference_wrapped_model.model.forward = original_forward

    assert response.status_code == 200
    assert response.is_json

    json_data = response.get_json()
    logprobs = json_data["choices"][0]["logprobs"]
    reconstructed_context_tokens = np.searchsorted(logprobs["text_offset"], len(context))

    assert reconstructed_context_tokens == context_token_count
    assert logprobs["token_logprobs"][context_token_count - 1] < -1

    suffix_logprob = logprobs["token_logprobs"][reconstructed_context_tokens:-1]
    assert len(suffix_logprob) == continuation_token_count
    assert sum(suffix_logprob) == 0, f"{suffix_logprob} != [0, .... 0]"

    mock_send_do_generate.assert_called_once()
