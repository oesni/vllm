# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers.solar_open_tool_parser import SolarOpenToolParser

TOOL_CALL_ID_RE = re.compile(r"^[a-z0-9]{10}$")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {
        "<|begin|>": 20,
        "<|end|>": 21,
        "<|think|>": 22,
        "<|content|>": 23,
        "<|flush|>": 24,
        "<|calls|>": 25,
        "<|tool_calls|>": 30,
        "<|tool_call:begin|>": 31,
        "<|tool_call:end|>": 32,
        "<|tool_call:name|>": 33,
        "<|tool_call:args|>": 34,
    }
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return SolarOpenToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


def make_tool_call_block(
    tool_call_id: str = "a1b2c3d4e5",
    name: str = "get_weather",
    args: str = '{"city": "Seoul"}',
) -> str:
    return (
        f"<|tool_call:begin|>{tool_call_id}"
        f"<|tool_call:name|>{name}"
        f"<|tool_call:args|>{args}<|tool_call:end|>"
    )


def stream_chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def run_streaming(
    parser: SolarOpenToolParser,
    request: ChatCompletionRequest,
    chunks: list[str],
) -> list[DeltaMessage]:
    """Feed chunks through the streaming API, collecting delta messages."""
    deltas: list[DeltaMessage] = []
    previous_text = ""
    for chunk in chunks:
        current_text = previous_text + chunk
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=chunk,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        if delta is not None:
            deltas.append(delta)
        previous_text = current_text
    return deltas


def reconstruct(deltas: list[DeltaMessage]) -> tuple[str, list[dict]]:
    """Reassemble streamed content and tool calls from delta messages."""
    content = ""
    tool_calls: list[dict] = []
    for delta in deltas:
        if delta.content:
            content += delta.content
        for tool_call in delta.tool_calls:
            while len(tool_calls) <= tool_call.index:
                tool_calls.append({"id": None, "name": "", "arguments": ""})
            entry = tool_calls[tool_call.index]
            if tool_call.id:
                entry["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    entry["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    entry["arguments"] += tool_call.function.arguments
    return content, tool_calls


# ---------------------------------------------------------------------------
# Non-streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = "<|tool_calls|>" + make_tool_call_block()
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call.id == "a1b2c3d4e5"
        assert tool_call.function.name == "get_weather"
        assert json.loads(tool_call.function.arguments) == {"city": "Seoul"}
        assert result.content is None

    def test_parallel_tool_calls(self, parser, mock_request):
        model_output = (
            "<|tool_calls|>"
            + make_tool_call_block("a1b2c3d4e5", "get_weather", '{"city": "Seoul"}')
            + make_tool_call_block("f6g7h8i9j0", "get_time", '{"tz": "UTC"}')
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"
        assert result.tool_calls[1].id == "f6g7h8i9j0"

    def test_tool_call_without_section_wrapper(self, parser, mock_request):
        # The reasoning parser may hand over the body without the
        # ``<|tool_calls|>`` wrapper.
        model_output = make_tool_call_block()
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.content is None

    def test_empty_args(self, parser, mock_request):
        model_output = "<|tool_calls|>" + make_tool_call_block(args="{}")
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert json.loads(result.tool_calls[0].function.arguments) == {}

    def test_python_literal_args_normalized(self, parser, mock_request):
        model_output = "<|tool_calls|>" + make_tool_call_block(
            args="{'city': 'Seoul', 'days': 3}"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "Seoul", "days": 3}

    def test_malformed_args_round_trip(self, parser, mock_request):
        raw_args = '{"city": "Seoul", broken'
        model_output = "<|tool_calls|>" + make_tool_call_block(args=raw_args)
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.arguments == raw_args

    def test_invalid_id_regenerated(self, parser, mock_request):
        model_output = "<|tool_calls|>" + make_tool_call_block(tool_call_id="NOT-VALID")
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert TOOL_CALL_ID_RE.match(result.tool_calls[0].id)
        assert result.tool_calls[0].id != "NOT-VALID"

    def test_duplicate_ids_deduplicated(self, parser, mock_request):
        # Greedy sampling can collapse the model's "random" ids to the same
        # sequence for parallel calls; ids must stay unique per response.
        model_output = (
            "<|tool_calls|>"
            + make_tool_call_block("0000000000", "get_weather", '{"city": "Seoul"}')
            + make_tool_call_block("0000000000", "get_time", '{"tz": "UTC"}')
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        ids = [tc.id for tc in result.tool_calls]
        assert len(ids) == 2
        assert len(set(ids)) == 2
        assert all(TOOL_CALL_ID_RE.match(i) for i in ids)

    def test_content_before_tool_calls(self, parser, mock_request):
        # Mixed output: a content turn followed by a tool-calls turn.
        model_output = (
            "Let me check.<|end|><|begin|>assistant<|tool_calls|>"
            + make_tool_call_block()
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check."
        assert len(result.tool_calls) == 1

    def test_deeply_nested_args_do_not_crash(self, parser, mock_request):
        # json.loads raises RecursionError (not JSONDecodeError) on deeply
        # nested input; the parser must degrade gracefully, not crash.
        model_output = "<|tool_calls|>" + make_tool_call_block(args="[" * 100000)
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.arguments == "[" * 100000

    def test_unicode_args(self, parser, mock_request):
        model_output = "<|tool_calls|>" + make_tool_call_block(args='{"city": "서울"}')
        result = parser.extract_tool_calls(model_output, mock_request)

        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"city": "서울"}
        # ensure_ascii=False keeps the original characters.
        assert "서울" in result.tool_calls[0].function.arguments


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCallsStreaming:
    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_content_only(self, parser, mock_request, chunk_size):
        text = "Hello, how can I help you today?"
        deltas = run_streaming(parser, mock_request, stream_chunks(text, chunk_size))
        content, tool_calls = reconstruct(deltas)

        assert content == text
        assert tool_calls == []

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_single_tool_call(self, parser, mock_request, chunk_size):
        text = "<|tool_calls|>" + make_tool_call_block()
        deltas = run_streaming(parser, mock_request, stream_chunks(text, chunk_size))
        content, tool_calls = reconstruct(deltas)

        assert content == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "a1b2c3d4e5"
        assert tool_calls[0]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["arguments"]) == {"city": "Seoul"}

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1000])
    def test_streaming_parallel_tool_calls(self, parser, mock_request, chunk_size):
        text = (
            "<|tool_calls|>"
            + make_tool_call_block("a1b2c3d4e5", "get_weather", '{"city": "Seoul"}')
            + make_tool_call_block("f6g7h8i9j0", "get_time", '{"tz": "UTC"}')
        )
        deltas = run_streaming(parser, mock_request, stream_chunks(text, chunk_size))
        content, tool_calls = reconstruct(deltas)

        assert content == ""
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["arguments"]) == {"city": "Seoul"}
        assert tool_calls[1]["id"] == "f6g7h8i9j0"
        assert tool_calls[1]["name"] == "get_time"
        assert json.loads(tool_calls[1]["arguments"]) == {"tz": "UTC"}

    @pytest.mark.parametrize("chunk_size", [1, 3, 7])
    def test_streaming_content_then_tool_call(self, parser, mock_request, chunk_size):
        text = (
            "Let me check.<|end|><|begin|>assistant<|tool_calls|>"
            + make_tool_call_block()
        )
        deltas = run_streaming(parser, mock_request, stream_chunks(text, chunk_size))
        content, tool_calls = reconstruct(deltas)

        assert content == "Let me check."
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "get_weather"

    def test_streaming_markers_never_leak(self, parser, mock_request):
        text = "<|tool_calls|>" + make_tool_call_block()
        deltas = run_streaming(parser, mock_request, stream_chunks(text, 1))
        content, tool_calls = reconstruct(deltas)

        assert "<|" not in content
        for tool_call in tool_calls:
            assert "<|" not in tool_call["arguments"]

    def test_streaming_name_emitted_before_args(self, parser, mock_request):
        text = "<|tool_calls|>" + make_tool_call_block()
        deltas = run_streaming(parser, mock_request, stream_chunks(text, 5))

        # The first tool-call delta must carry the id, type and name.
        first_tool_delta = next(d for d in deltas if d.tool_calls)
        tool_call = first_tool_delta.tool_calls[0]
        assert tool_call.id == "a1b2c3d4e5"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert tool_call.function.arguments == ""

    def test_streaming_args_held_until_unambiguous(self, parser, mock_request):
        # While inside the args of a call, bytes that could be the start of
        # ``<|tool_call:end|>`` must not be emitted as arguments.
        text = "<|tool_calls|>" + make_tool_call_block(args='{"a": "<b>"}')
        deltas = run_streaming(parser, mock_request, stream_chunks(text, 2))
        _, tool_calls = reconstruct(deltas)

        assert json.loads(tool_calls[0]["arguments"]) == {"a": "<b>"}

    def test_streaming_invalid_id_regenerated(self, parser, mock_request):
        text = "<|tool_calls|>" + make_tool_call_block(tool_call_id="")
        deltas = run_streaming(parser, mock_request, stream_chunks(text, 4))
        _, tool_calls = reconstruct(deltas)

        assert TOOL_CALL_ID_RE.match(tool_calls[0]["id"])

    def test_streaming_duplicate_ids_deduplicated(self, parser, mock_request):
        text = (
            "<|tool_calls|>"
            + make_tool_call_block("0000000000", "get_weather", '{"city": "Seoul"}')
            + make_tool_call_block("0000000000", "get_time", '{"tz": "UTC"}')
        )
        deltas = run_streaming(parser, mock_request, stream_chunks(text, 5))
        _, tool_calls = reconstruct(deltas)

        assert len(tool_calls) == 2
        assert tool_calls[0]["id"] != tool_calls[1]["id"]

    def test_streaming_state_reset_between_requests(self, parser, mock_request):
        text = "<|tool_calls|>" + make_tool_call_block()
        deltas_first = run_streaming(parser, mock_request, stream_chunks(text, 3))
        deltas_second = run_streaming(parser, mock_request, stream_chunks(text, 3))
        _, tool_calls_first = reconstruct(deltas_first)
        _, tool_calls_second = reconstruct(deltas_second)

        assert len(tool_calls_first) == 1
        # The second stream must start over at index 0.
        assert len(tool_calls_second) == 1
        assert tool_calls_second[0]["name"] == "get_weather"

    def test_streaming_unstreamed_args_bookkeeping(self, parser, mock_request):
        # ``get_remaining_unstreamed_args`` must be empty once the call's
        # args were fully streamed.
        text = "<|tool_calls|>" + make_tool_call_block()
        run_streaming(parser, mock_request, stream_chunks(text, 3))

        assert parser.get_remaining_unstreamed_args() == ""
