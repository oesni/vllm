# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class SolarOpenReasoningParser(ReasoningParser):
    """Reasoning parser for Solar Open models (e.g. upstage/Solar-Open-100B).

    Solar Open renders each assistant turn as ``<|begin|>assistant`` followed
    by a body. The generation prompt ends with ``<|begin|>assistant``, so the
    model output for one request looks like one of:

    - ``<|think|>{reasoning}<|end|><|begin|>assistant<|content|>{content}``
    - ``<|think|>{reasoning}<|end|><|begin|>assistant<|tool_calls|>{calls}``
    - ``<|content|>{content}`` (reasoning suppressed, e.g. the chat template
      prefilled an empty ``<|begin|>assistant<|think|><|end|>`` turn for
      ``reasoning_effort`` of ``low`` or ``minimal``)
    - ``<|tool_calls|>{calls}``

    where generation terminates with ``<|flush|>`` or ``<|calls|>``, both of
    which are EOS tokens and thus never appear in the decoded text. All the
    markers above are single tokens with ``special=False``, so they survive
    detokenization with default sampling parameters.

    This parser splits the reasoning block from the response body:

    - ``reasoning`` is the text between ``<|think|>`` and ``<|end|>``.
    - ``content`` is everything after ``<|content|>`` (the tag is stripped),
      or everything starting at ``<|tool_calls|>`` (the tag is kept so a tool
      parser can detect the tool-call section).

    Streaming uses a small state machine with a suffix hold-back: trailing
    bytes that could be the prefix of an upcoming marker are kept in the
    buffer, because vLLM's incremental detokenizer can split a single marker
    token's text across delta boundaries.

    ``is_reasoning_end`` must answer "has the reasoning for the *current*
    assistant turn ended?". A plain containment check is wrong for prompts:
    the chat template renders every prior turn with ``<|content|>`` tags, so
    any multi-turn prompt contains them. The check is therefore scoped to the
    tokens after the last ``<|begin|>`` token. For the streaming case the
    parser defers the flip to its own text-level state so that bytes held
    back in the buffer are not leaked to the tool parser. This relies on the
    parser instance lifecycle: the serving frontend creates a fresh instance
    per request (and calls ``is_reasoning_end`` on the prompt before any
    streaming), while engine-side users such as structured output never call
    ``extract_reasoning_streaming`` and thus always take the token-id path.
    """

    THINK = "<|think|>"
    END = "<|end|>"
    BEGIN = "<|begin|>"
    CONTENT = "<|content|>"
    TOOL_CALLS = "<|tool_calls|>"

    # The transition between the reasoning turn and the response turn,
    # emitted by the model right after ``<|end|>``.
    _TRANSITION = "<|begin|>assistant"

    # Streaming state machine states.
    _STATE_START = "start"
    _STATE_REASONING = "reasoning"
    _STATE_TRANSITION = "transition"
    _STATE_CONTENT = "content"

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        self.begin_token_id = self.vocab.get(self.BEGIN)
        self.content_token_id = self.vocab.get(self.CONTENT)
        self.tool_calls_token_id = self.vocab.get(self.TOOL_CALLS)
        if None in (
            self.begin_token_id,
            self.content_token_id,
            self.tool_calls_token_id,
        ):
            raise RuntimeError(
                "Solar Open reasoning parser could not locate the "
                f"{self.BEGIN}, {self.CONTENT} or {self.TOOL_CALLS} tokens "
                "in the tokenizer!"
            )
        self._reset_stream()

    @property
    def reasoning_start_str(self) -> str:
        return self.THINK

    @property
    def reasoning_end_str(self) -> str:
        return self.END

    def _reset_stream(self) -> None:
        """Reset streaming-only state. Called on init and at the start of
        every new stream (detected via empty ``previous_text``)."""
        self._stream_buffer: str = ""
        self._stream_state: str = self._STATE_START
        # ``True`` once we've handled at least one streaming delta in this
        # request — used to gate the streaming-aware ``is_reasoning_end``
        # behavior so non-streaming and prompt-side callers fall back to
        # the canonical token-id check.
        self._stream_active: bool = False

    @staticmethod
    def _holdback_suffix(buf: str, sentinels: tuple[str, ...]) -> int:
        """Return the number of trailing bytes of ``buf`` that are a proper,
        non-empty prefix of any sentinel in ``sentinels`` — these bytes
        might complete into the sentinel once more data arrives and must
        stay in the buffer.

        Anchor on the last ``<`` (all Solar Open markers start with ``<|``)
        and only consider genuine *tail* prefixes; a letter that happens to
        appear inside a marker is NOT held back.
        """
        last_lt = buf.rfind("<")
        if last_lt == -1:
            return 0
        tail = buf[last_lt:]
        for s in sentinels:
            if len(tail) < len(s) and s.startswith(tail):
                return len(tail)
        return 0

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Streaming-aware path: defer the flip until the text-level state
        # machine has entered the content phase. This prevents the serving
        # layer from bypassing this parser while partial bytes of a marker
        # are still held back in the buffer, which would leak them to the
        # tool parser as a spurious content delta.
        if self._stream_active:
            return self._stream_state == self._STATE_CONTENT

        # Canonical token-id check, scoped to the current assistant turn:
        # only the tokens after the last ``<|begin|>`` matter. The chat
        # template renders every prior turn with ``<|content|>`` tags, so an
        # unscoped containment check would fire on any multi-turn prompt.
        ids = list(input_ids)
        start = 0
        for i in range(len(ids) - 1, -1, -1):
            if ids[i] == self.begin_token_id:
                start = i + 1
                break
        tail = ids[start:]
        return self.content_token_id in tail or self.tool_calls_token_id in tail

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        # Return the token ids of the response body, mirroring the text
        # contract: after ``<|content|>`` (tag excluded), or from
        # ``<|tool_calls|>`` (tag included).
        if self.content_token_id in input_ids:
            idx = input_ids.index(self.content_token_id)
            return input_ids[idx + 1 :]
        if self.tool_calls_token_id in input_ids:
            idx = input_ids.index(self.tool_calls_token_id)
            return input_ids[idx:]
        return []

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        think_idx = model_output.find(self.THINK)
        if think_idx != -1:
            after_think = model_output[think_idx + len(self.THINK) :]
            end_idx = after_think.find(self.END)
            if end_idx == -1:
                # Truncated reasoning (e.g. max_tokens reached before
                # ``<|end|>``): surface what was generated as reasoning.
                return after_think or None, None
            reasoning = after_think[:end_idx] or None
            rest = after_think[end_idx + len(self.END) :]
        else:
            reasoning = None
            rest = model_output

        content = self._extract_body(rest)
        if reasoning is None and content is None:
            # No recognized turn structure (e.g. a raw completion): treat
            # the whole output as content.
            return None, model_output or None
        return reasoning, content or None

    def _extract_body(self, text: str) -> str | None:
        """Extract the response body from the text after the reasoning
        block: the text after ``<|content|>``, or from ``<|tool_calls|>``
        (inclusive), whichever appears first. Returns an empty string for
        a present-but-empty body and ``None`` when no body marker exists.
        """
        content_idx = text.find(self.CONTENT)
        tool_calls_idx = text.find(self.TOOL_CALLS)
        if content_idx != -1 and (tool_calls_idx == -1 or content_idx < tool_calls_idx):
            return text[content_idx + len(self.CONTENT) :]
        if tool_calls_idx != -1:
            return text[tool_calls_idx:]
        return None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        # Fresh stream — the serving frontend creates a fresh parser per
        # request, but reset defensively in case an instance is reused.
        if not previous_text:
            self._reset_stream()
        self._stream_active = True

        if delta_text:
            self._stream_buffer += delta_text

        reasoning_out = ""
        content_out = ""

        while True:
            buf = self._stream_buffer

            if self._stream_state == self._STATE_START:
                # Look for the first structural marker.
                think_idx = buf.find(self.THINK)
                body_state = self._find_body_start(buf)
                if think_idx != -1 and (
                    body_state is None or think_idx < body_state[0]
                ):
                    # Anything before ``<|think|>`` is unexpected; surface
                    # it as content rather than dropping it.
                    content_out += buf[:think_idx]
                    self._stream_buffer = buf[think_idx + len(self.THINK) :]
                    self._stream_state = self._STATE_REASONING
                    continue
                if body_state is not None:
                    idx, keep_tag, tag = body_state
                    content_out += buf[:idx]
                    body_start = idx if keep_tag else idx + len(tag)
                    content_out += buf[body_start:]
                    self._stream_buffer = ""
                    self._stream_state = self._STATE_CONTENT
                    continue
                # No marker yet: flush everything that cannot be the start
                # of one as content (defensive; the model normally opens
                # with a marker token).
                hb = self._holdback_suffix(
                    buf, (self.THINK, self.CONTENT, self.TOOL_CALLS)
                )
                flush_end = len(buf) - hb
                if flush_end > 0:
                    content_out += buf[:flush_end]
                    self._stream_buffer = buf[flush_end:]
                break

            if self._stream_state == self._STATE_REASONING:
                end_idx = buf.find(self.END)
                if end_idx != -1:
                    reasoning_out += buf[:end_idx]
                    self._stream_buffer = buf[end_idx + len(self.END) :]
                    self._stream_state = self._STATE_TRANSITION
                    continue
                hb = self._holdback_suffix(buf, (self.END,))
                flush_end = len(buf) - hb
                if flush_end > 0:
                    reasoning_out += buf[:flush_end]
                    self._stream_buffer = buf[flush_end:]
                break

            if self._stream_state == self._STATE_TRANSITION:
                # Between ``<|end|>`` and the response body the model emits
                # exactly ``<|begin|>assistant``. Swallow it, then enter the
                # content phase at ``<|content|>`` or ``<|tool_calls|>``.
                body_state = self._find_body_start(buf)
                trans_idx = buf.find(self._TRANSITION)
                if trans_idx != -1 and (
                    body_state is None or trans_idx < body_state[0]
                ):
                    # Surface any unexpected text before the turn header as
                    # content, swallow the header itself.
                    content_out += buf[:trans_idx]
                    self._stream_buffer = buf[trans_idx + len(self._TRANSITION) :]
                    continue
                if body_state is not None:
                    idx, keep_tag, tag = body_state
                    body_start = idx if keep_tag else idx + len(tag)
                    # Surface unexpected text before the body marker as
                    # content (it is empty for well-formed output).
                    content_out += buf[:idx].replace(self._TRANSITION, "")
                    content_out += buf[body_start:]
                    self._stream_buffer = ""
                    self._stream_state = self._STATE_CONTENT
                    continue
                hb = self._holdback_suffix(
                    buf, (self.CONTENT, self.TOOL_CALLS, self._TRANSITION)
                )
                if hb == len(buf):
                    break
                # Unexpected text after the reasoning block: surface it as
                # content rather than swallowing it silently, but stay in
                # the transition state so a later body marker is still
                # recognized (and stripped) instead of leaking through.
                flush_end = len(buf) - hb
                content_out += buf[:flush_end]
                self._stream_buffer = buf[flush_end:]
                break

            if self._stream_state == self._STATE_CONTENT:
                content_out += buf
                self._stream_buffer = ""
                break

            break

        if not reasoning_out and not content_out:
            return None
        # A boundary delta may carry both the tail of the reasoning block
        # and the beginning of the content; emit both on one DeltaMessage.
        return DeltaMessage(
            reasoning=reasoning_out or None,
            content=content_out or None,
        )

    def _find_body_start(self, buf: str) -> tuple[int, bool, str] | None:
        """Locate the response-body marker in ``buf``.

        Returns ``(index, keep_tag, tag)`` for the earliest of
        ``<|content|>`` (tag stripped from the content) and
        ``<|tool_calls|>`` (tag kept for the tool parser), or ``None``
        if neither marker is present.
        """
        content_idx = buf.find(self.CONTENT)
        tool_calls_idx = buf.find(self.TOOL_CALLS)
        if content_idx != -1 and (tool_calls_idx == -1 or content_idx < tool_calls_idx):
            return content_idx, False, self.CONTENT
        if tool_calls_idx != -1:
            return tool_calls_idx, True, self.TOOL_CALLS
        return None
