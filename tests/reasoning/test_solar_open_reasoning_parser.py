# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.solar_open_reasoning_parser import SolarOpenReasoningParser

BEGIN = 20
END = 21
THINK = 22
CONTENT = 23
TOOL_CALLS = 30
# An arbitrary id for ordinary text tokens in synthetic token sequences.
TEXT = 1000

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<|begin|>": BEGIN,
        "<|end|>": END,
        "<|think|>": THINK,
        "<|content|>": CONTENT,
        "<|tool_calls|>": TOOL_CALLS,
    }
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return SolarOpenReasoningParser(mock_tokenizer)


def stream_chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def run_streaming(
    parser: SolarOpenReasoningParser, chunks: list[str]
) -> list[DeltaMessage]:
    deltas: list[DeltaMessage] = []
    previous_text = ""
    for chunk in chunks:
        current_text = previous_text + chunk
        delta = parser.extract_reasoning_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        if delta is not None:
            deltas.append(delta)
        previous_text = current_text
    return deltas


def reconstruct(deltas: list[DeltaMessage]) -> tuple[str, str]:
    """Reassemble streamed reasoning and content.

    Unlike ``tests.reasoning.utils.StreamingReasoningReconstructor`` this
    allows a single delta to carry both reasoning and content: the serving
    layer explicitly supports boundary deltas carrying the tail of the
    reasoning block together with the beginning of the content (see
    ``DelegatingParser.parse_delta``).
    """
    reasoning = ""
    content = ""
    for delta in deltas:
        if delta.reasoning:
            reasoning += delta.reasoning
        if delta.content:
            content += delta.content
    return reasoning, content


# ---------------------------------------------------------------------------
# Non-streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractReasoning:
    def test_reasoning_and_content(self, parser):
        model_output = (
            "<|think|>Let me think.<|end|>"
            "<|begin|>assistant<|content|>The answer is 42."
        )
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning == "Let me think."
        assert content == "The answer is 42."

    def test_reasoning_and_tool_calls(self, parser):
        model_output = (
            "<|think|>I should call a tool.<|end|>"
            "<|begin|>assistant<|tool_calls|>"
            "<|tool_call:begin|>a1b2c3d4e5<|tool_call:name|>get_weather"
            '<|tool_call:args|>{"city": "Seoul"}<|tool_call:end|>'
        )
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning == "I should call a tool."
        # The tool-calls tag is kept so the tool parser can detect the
        # section.
        assert content.startswith("<|tool_calls|>")
        assert "get_weather" in content

    def test_content_only(self, parser):
        # reasoning_effort low/minimal: the chat template prefilled an empty
        # think turn, so the model emits the content turn directly.
        model_output = "<|content|>Hello!"
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning is None
        assert content == "Hello!"

    def test_tool_calls_only(self, parser):
        model_output = (
            "<|tool_calls|><|tool_call:begin|>a1b2c3d4e5"
            "<|tool_call:name|>get_time<|tool_call:args|>{}<|tool_call:end|>"
        )
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning is None
        assert content == model_output

    def test_truncated_reasoning(self, parser):
        # max_tokens reached before ``<|end|>``: surface the partial
        # reasoning.
        model_output = "<|think|>Still thinking"
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning == "Still thinking"
        assert content is None

    def test_reasoning_with_empty_content(self, parser):
        model_output = "<|think|>Done.<|end|><|begin|>assistant<|content|>"
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning == "Done."
        assert content is None

    def test_plain_text_fallback(self, parser):
        # No recognized structure: treat the whole output as content.
        model_output = "Just plain text."
        reasoning, content = parser.extract_reasoning(model_output, request=None)

        assert reasoning is None
        assert content == model_output

    def test_empty_output(self, parser):
        reasoning, content = parser.extract_reasoning("", request=None)

        assert reasoning is None
        assert content is None

    def test_bare_content_marker(self, parser):
        # The raw marker must not leak as content when the body is empty.
        reasoning, content = parser.extract_reasoning("<|content|>", request=None)

        assert reasoning is None
        assert content is None


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractReasoningStreaming:
    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_reasoning_and_content(self, parser, chunk_size):
        text = (
            "<|think|>Let me think.<|end|>"
            "<|begin|>assistant<|content|>The answer is 42."
        )
        deltas = run_streaming(parser, stream_chunks(text, chunk_size))
        reasoning, content = reconstruct(deltas)

        assert reasoning == "Let me think."
        assert content == "The answer is 42."

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_reasoning_and_tool_calls(self, parser, chunk_size):
        text = (
            "<|think|>I should call a tool.<|end|>"
            "<|begin|>assistant<|tool_calls|><|tool_call:begin|>a1b2c3d4e5"
            '<|tool_call:name|>get_weather<|tool_call:args|>{"city": "Seoul"}'
            "<|tool_call:end|>"
        )
        deltas = run_streaming(parser, stream_chunks(text, chunk_size))
        reasoning, content = reconstruct(deltas)

        assert reasoning == "I should call a tool."
        assert content.startswith("<|tool_calls|>")

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_content_only(self, parser, chunk_size):
        text = "<|content|>Hello!"
        deltas = run_streaming(parser, stream_chunks(text, chunk_size))
        reasoning, content = reconstruct(deltas)

        assert reasoning == ""
        assert content == "Hello!"

    def test_streaming_no_template_tokens_leak(self, parser):
        text = (
            "<|think|>Some thought with < and <| inside.<|end|>"
            "<|begin|>assistant<|content|>Answer with < bracket."
        )
        deltas = run_streaming(parser, stream_chunks(text, 1))
        reasoning, content = reconstruct(deltas)

        assert reasoning == "Some thought with < and <| inside."
        assert content == "Answer with < bracket."
        assert "<|begin|>" not in content
        assert "<|end|>" not in reasoning

    def test_streaming_truncated_reasoning(self, parser):
        # Stream ends before ``<|end|>`` arrives: the reasoning emitted so
        # far (minus held-back marker prefix bytes) was streamed already.
        text = "<|think|>Still thinking"
        deltas = run_streaming(parser, stream_chunks(text, 4))
        reasoning, content = reconstruct(deltas)

        assert reasoning == "Still thinking"
        assert content == ""

    def test_streaming_state_reset_between_requests(self, parser):
        text = "<|think|>One.<|end|><|begin|>assistant<|content|>Two."
        deltas_first = run_streaming(parser, stream_chunks(text, 3))
        deltas_second = run_streaming(parser, stream_chunks(text, 3))

        assert reconstruct(deltas_first) == reconstruct(deltas_second)

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_unexpected_transition_text_no_marker_leak(
        self, parser, chunk_size
    ):
        # Off-format: stray text between <|end|> and the body marker. The
        # stray text is surfaced as content, but the body marker must still
        # be recognized and stripped instead of leaking to the client.
        text = "<|think|>R<|end|>stray<|begin|>assistant<|content|>Hi"
        deltas = run_streaming(parser, stream_chunks(text, chunk_size))
        reasoning, content = reconstruct(deltas)

        assert reasoning == "R"
        assert "<|content|>" not in content
        assert "<|begin|>" not in content
        assert content == "strayHi"


# ---------------------------------------------------------------------------
# is_reasoning_end / extract_content_ids tests
# ---------------------------------------------------------------------------


class TestIsReasoningEnd:
    def test_high_effort_prompt_not_ended(self, parser):
        # Prompt ends with the generation header ``<|begin|>assistant``.
        prompt_ids = [TEXT, CONTENT, TEXT, END, BEGIN, TEXT]
        assert parser.is_reasoning_end(prompt_ids) is False

    def test_low_effort_prompt_not_ended(self, parser):
        # The empty think pair sits before the final generation header; the
        # streaming state machine handles the content tag of the new turn.
        prompt_ids = [TEXT, BEGIN, TEXT, THINK, END, BEGIN, TEXT]
        assert parser.is_reasoning_end(prompt_ids) is False

    def test_multi_turn_prompt_not_ended(self, parser):
        # ``<|content|>`` tags from earlier turns must not flip the check.
        prompt_ids = [
            BEGIN, TEXT, CONTENT, TEXT, END,
            BEGIN, TEXT, THINK, TEXT, END,
            BEGIN, TEXT,
        ]  # fmt: skip
        assert parser.is_reasoning_end(prompt_ids) is False

    def test_generated_content_ended(self, parser):
        generated_ids = [THINK, TEXT, END, BEGIN, TEXT, CONTENT, TEXT]
        assert parser.is_reasoning_end(generated_ids) is True

    def test_generated_tool_calls_ended(self, parser):
        generated_ids = [THINK, TEXT, END, BEGIN, TEXT, TOOL_CALLS, TEXT]
        assert parser.is_reasoning_end(generated_ids) is True

    def test_generated_reasoning_not_ended(self, parser):
        generated_ids = [THINK, TEXT, TEXT]
        assert parser.is_reasoning_end(generated_ids) is False

    def test_generated_low_effort_content_ended(self, parser):
        generated_ids = [CONTENT, TEXT]
        assert parser.is_reasoning_end(generated_ids) is True

    def test_streaming_aware_flip(self, parser):
        # During streaming the flip must wait for the parser's own state
        # machine, even if the token ids already contain the content tag.
        deltas = run_streaming(parser, ["<|think|>thought<|end|>", "<|begi"])
        assert parser.is_reasoning_end([THINK, TEXT, END, BEGIN]) is False

        parser.extract_reasoning_streaming(
            previous_text="<|think|>thought<|end|><|begi",
            current_text="<|think|>thought<|end|><|begin|>assistant<|content|>Hi",
            delta_text="n|>assistant<|content|>Hi",
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
        )
        assert parser.is_reasoning_end([THINK, TEXT, END, BEGIN, TEXT, CONTENT]) is True
        assert deltas  # the reasoning was streamed before the flip


class TestExtractContentIds:
    def test_content_ids_after_content_tag(self, parser):
        ids = [THINK, TEXT, END, BEGIN, CONTENT, TEXT, TEXT]
        assert parser.extract_content_ids(ids) == [TEXT, TEXT]

    def test_content_ids_from_tool_calls_tag(self, parser):
        ids = [THINK, TEXT, END, BEGIN, TOOL_CALLS, TEXT]
        assert parser.extract_content_ids(ids) == [TOOL_CALLS, TEXT]

    def test_content_ids_no_body(self, parser):
        ids = [THINK, TEXT, TEXT]
        assert parser.extract_content_ids(ids) == []
