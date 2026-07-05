from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.services.llm_client as llm_client_module
from app.config import Settings
from app.schemas.llm_io import LLMMessage, LLMResponse, Role, StopReason, ToolCall, Usage
from app.services.llm_client import (
    AnthropicClient,
    LLMError,
    OpenAICompatibleClient,
    build_llm_client,
)

CONFIG_ENV_VARS = (
    "REPOPILOT_LLM_PROVIDER",
    "REPOPILOT_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search code.",
            "parameters": {"type": "object"},
        },
    },
]


@pytest.fixture(autouse=True)
def clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in CONFIG_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


class _FakeCompletions:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise RuntimeError("Fake response is missing.")
        return self.response


class _FakeClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.completions = _FakeCompletions(response=response, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeAnthropicMessages:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.response is None:
            raise RuntimeError("Fake response is missing.")
        return self.response


class _FakeAnthropicClient:
    def __init__(self, response: object | None = None, error: Exception | None = None) -> None:
        self.messages = _FakeAnthropicMessages(response=response, error=error)


def _completion(
    *,
    finish_reason: str = "stop",
    content: str | None = "Done.",
    tool_calls: list[object] | None = None,
    tokens_in: int = 100,
    tokens_out: int = 50,
    model: str = "deepseek-v4-pro",
) -> object:
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def _anthropic_response(
    *,
    stop_reason: str = "end_turn",
    content: list[object] | None = None,
    tokens_in: int = 100,
    tokens_out: int = 50,
    model: str = "claude-opus-4-8",
) -> object:
    if content is None:
        content = [_anthropic_text("Done.")]
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        content=content,
        usage=SimpleNamespace(input_tokens=tokens_in, output_tokens=tokens_out),
    )


def _anthropic_text(text: str) -> object:
    return {"type": "text", "text": text}


def _anthropic_tool_use(
    *,
    call_id: str = "toolu-1",
    name: str = "read_file",
    arguments: dict[str, object] | None = None,
) -> object:
    if arguments is None:
        arguments = {}
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def _tool_call(
    *,
    call_id: str = "call-1",
    name: str = "read_file",
    arguments: str = "{}",
) -> object:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_openai_compatible_client__normalizes_text_completion_usage_and_cost() -> None:
    fake_client = _FakeClient(
        response=_completion(content="Ready.", tokens_in=1000, tokens_out=500)
    )
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Hello")])

    assert response.message == LLMMessage(role=Role.assistant, content="Ready.")
    assert response.stop_reason is StopReason.end_turn
    assert response.usage.tokens_in == 1000
    assert response.usage.tokens_out == 500
    assert response.usage.cost_usd == pytest.approx(0.00087)
    assert response.model == "deepseek-v4-pro"
    assert response.raw_finish_reason == "stop"


def test_openai_compatible_client__normalizes_tool_calls_and_raw_finish_reason() -> None:
    fake_client = _FakeClient(
        response=_completion(
            finish_reason="tool_calls",
            content=None,
            tool_calls=[
                _tool_call(
                    call_id="call-123",
                    name="search_code",
                    arguments='{"query": "LLMClient", "limit": 5}',
                )
            ],
        )
    )
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Find it")])

    assert response.stop_reason is StopReason.tool_use
    assert response.raw_finish_reason == "tool_calls"
    assert response.message.tool_calls == [
        ToolCall(
            id="call-123",
            name="search_code",
            arguments={"query": "LLMClient", "limit": 5},
        )
    ]


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"name": "read_file", "arguments": {"path": "README.md"}}\n```',
        '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>',
        '{"name": "read_file", "arguments": {"path": "README.md"}}',
    ],
)
def test_openai_compatible_client__salvages_plain_text_tool_call_shapes(
    content: str,
) -> None:
    fake_client = _FakeClient(response=_completion(content=content))
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Read it")], tools=TOOL_SCHEMAS)

    assert response.stop_reason is StopReason.tool_use
    assert response.raw_finish_reason == "stop"
    assert response.recovered_tool_calls is True
    assert response.message.content == ""
    assert response.message.tool_calls == [
        ToolCall(
            id="call-salvaged-0",
            name="read_file",
            arguments={"path": "README.md"},
        )
    ]
    assert len(fake_client.completions.calls) == 1


def test_openai_compatible_client__salvages_plain_text_tool_call_array() -> None:
    fake_client = _FakeClient(
        response=_completion(
            content=(
                "["
                '{"name": "read_file", "arguments": {"path": "README.md"}},'
                '{"name": "search_code", "arguments": {"query": "LLMClient"}}'
                "]"
            )
        )
    )
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete(
        [LLMMessage(role=Role.user, content="Read and search")], tools=TOOL_SCHEMAS
    )

    assert response.stop_reason is StopReason.tool_use
    assert response.recovered_tool_calls is True
    assert response.message.content == ""
    assert response.message.tool_calls == [
        ToolCall(
            id="call-salvaged-0",
            name="read_file",
            arguments={"path": "README.md"},
        ),
        ToolCall(
            id="call-salvaged-1",
            name="search_code",
            arguments={"query": "LLMClient"},
        ),
    ]
    assert len(fake_client.completions.calls) == 1


def test_openai_compatible_client__does_not_salvage_normal_prose() -> None:
    content = (
        'I can call read_file with {"name": "read_file", "arguments": {"path": "README.md"}} '
        "after you confirm the target."
    )
    fake_client = _FakeClient(response=_completion(content=content))
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Read it")], tools=TOOL_SCHEMAS)

    assert response.stop_reason is StopReason.end_turn
    assert response.recovered_tool_calls is False
    assert response.message.content == content
    assert response.message.tool_calls == []
    assert len(fake_client.completions.calls) == 1


def test_openai_compatible_client__structured_tool_calls_prevent_salvage() -> None:
    content = '{"name": "read_file", "arguments": {"path": "README.md"}}'
    fake_client = _FakeClient(
        response=_completion(
            finish_reason="tool_calls",
            content=content,
            tool_calls=[
                _tool_call(
                    call_id="call-structured",
                    name="search_code",
                    arguments='{"query": "LLMClient"}',
                )
            ],
        )
    )
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Search")], tools=TOOL_SCHEMAS)

    assert response.stop_reason is StopReason.tool_use
    assert response.recovered_tool_calls is False
    assert response.message.content == content
    assert response.message.tool_calls == [
        ToolCall(
            id="call-structured",
            name="search_code",
            arguments={"query": "LLMClient"},
        )
    ]
    assert len(fake_client.completions.calls) == 1


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name": "read_file", "arguments": </tool_call>',
        '<tool_call>{"name": "read_file", "arguments": {}, "extra": true}</tool_call>',
    ],
)
def test_openai_compatible_client__invalid_tagged_tool_call_text_returns_end_turn(
    content: str,
) -> None:
    fake_client = _FakeClient(response=_completion(content=content))
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Read it")], tools=TOOL_SCHEMAS)

    assert response.stop_reason is StopReason.end_turn
    assert response.recovered_tool_calls is False
    assert response.message.content == content
    assert response.message.tool_calls == []
    assert len(fake_client.completions.calls) == 1


def test_openai_compatible_client__does_not_salvage_when_tools_are_not_offered() -> None:
    content = '{"name": "read_file", "arguments": {"path": "README.md"}}'
    fake_client = _FakeClient(response=_completion(content=content))
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Read it")])

    assert response.stop_reason is StopReason.end_turn
    assert response.recovered_tool_calls is False
    assert response.message.content == content
    assert response.message.tool_calls == []
    assert len(fake_client.completions.calls) == 1


@pytest.mark.parametrize(
    ("finish_reason", "expected_stop_reason"),
    [
        ("length", StopReason.max_tokens),
        ("content_filter", StopReason.refusal),
        ("unexpected", StopReason.end_turn),
    ],
)
def test_openai_compatible_client__maps_finish_reasons(
    finish_reason: str,
    expected_stop_reason: StopReason,
) -> None:
    fake_client = _FakeClient(
        response=_completion(finish_reason=finish_reason, model="unknown-model")
    )
    client = OpenAICompatibleClient(model="unknown-model", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Hello")])

    assert response.stop_reason is expected_stop_reason
    assert response.raw_finish_reason == finish_reason
    assert response.usage.cost_usd is None


def test_openai_compatible_client__prepends_system_message() -> None:
    fake_client = _FakeClient(response=_completion())
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    client.complete([LLMMessage(role=Role.user, content="Hello")], system="Be precise.")

    assert fake_client.completions.calls[0]["messages"] == [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Hello"},
    ]


def test_openai_compatible_client__passes_tools_through() -> None:
    fake_client = _FakeClient(response=_completion())
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)
    schema = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": {"type": "object"},
            },
        }
    ]

    client.complete([LLMMessage(role=Role.user, content="Read")], tools=schema)

    assert fake_client.completions.calls[0]["tools"] is schema


def test_openai_compatible_client__serializes_tool_role_messages() -> None:
    fake_client = _FakeClient(response=_completion())
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    client.complete(
        [
            LLMMessage(
                role=Role.tool,
                content='{"ok": true}',
                tool_call_id="call-1",
            )
        ]
    )

    assert fake_client.completions.calls[0]["messages"] == [
        {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call-1"}
    ]


def test_openai_compatible_client__wraps_sdk_errors() -> None:
    fake_client = _FakeClient(error=RuntimeError("sdk down"))
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    with pytest.raises(LLMError, match="sdk down"):
        client.complete([LLMMessage(role=Role.user, content="Hello")])


def test_openai_compatible_client__raises_llm_error_for_invalid_tool_arguments() -> None:
    fake_client = _FakeClient(
        response=_completion(
            finish_reason="tool_calls",
            tool_calls=[_tool_call(arguments="{not-json")],
        )
    )
    client = OpenAICompatibleClient(model="deepseek-v4-pro", client=fake_client)

    with pytest.raises(LLMError, match="Invalid tool call arguments"):
        client.complete([LLMMessage(role=Role.user, content="Hello")])


def test_build_llm_client__uses_openai_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients = []

    class _FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url
            self.chat = SimpleNamespace(completions=_FakeCompletions(response=_completion()))
            created_clients.append(self)

    monkeypatch.setattr(llm_client_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    settings = Settings(_env_file=None)

    client = build_llm_client(settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert created_clients[0].api_key == "sk-test-openai"
    assert created_clients[0].base_url == "https://example.test/v1"


def test_anthropic_client__normalizes_text_completion_usage_and_cost() -> None:
    fake_client = _FakeAnthropicClient(
        response=_anthropic_response(
            content=[_anthropic_text("Ready.")],
            tokens_in=1000,
            tokens_out=500,
        )
    )
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Hello")])

    assert response.message == LLMMessage(role=Role.assistant, content="Ready.")
    assert response.stop_reason is StopReason.end_turn
    assert response.usage.tokens_in == 1000
    assert response.usage.tokens_out == 500
    assert response.usage.cost_usd == pytest.approx(0.0175)
    assert response.model == "claude-opus-4-8"
    assert response.raw_finish_reason == "end_turn"
    assert response.recovered_tool_calls is False
    assert fake_client.messages.calls[0]["max_tokens"] == 4096
    assert "thinking" not in fake_client.messages.calls[0]


def test_anthropic_client__normalizes_tool_use_blocks_without_json_parsing() -> None:
    arguments = {"query": "LLMClient", "limit": 5}
    fake_client = _FakeAnthropicClient(
        response=_anthropic_response(
            stop_reason="tool_use",
            content=[
                _anthropic_tool_use(
                    call_id="toolu-123",
                    name="search_code",
                    arguments=arguments,
                )
            ],
        )
    )
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Find it")])

    assert response.stop_reason is StopReason.tool_use
    assert response.raw_finish_reason == "tool_use"
    assert response.message.tool_calls == [
        ToolCall(
            id="toolu-123",
            name="search_code",
            arguments={"query": "LLMClient", "limit": 5},
        )
    ]


@pytest.mark.parametrize(
    ("stop_reason", "expected_stop_reason"),
    [
        ("tool_use", StopReason.tool_use),
        ("end_turn", StopReason.end_turn),
        ("max_tokens", StopReason.max_tokens),
        ("pause_turn", StopReason.pause_turn),
        ("refusal", StopReason.refusal),
        ("stop_sequence", StopReason.end_turn),
        ("unexpected", StopReason.end_turn),
    ],
)
def test_anthropic_client__maps_stop_reasons(
    stop_reason: str,
    expected_stop_reason: StopReason,
) -> None:
    fake_client = _FakeAnthropicClient(
        response=_anthropic_response(
            stop_reason=stop_reason,
            content=[] if stop_reason == "refusal" else None,
            model="unknown-model",
        )
    )
    client = AnthropicClient(model="unknown-model", client=fake_client)

    response = client.complete([LLMMessage(role=Role.user, content="Hello")])

    assert response.stop_reason is expected_stop_reason
    assert response.raw_finish_reason == stop_reason
    assert response.usage.cost_usd is None
    if stop_reason == "refusal":
        assert response.message.content == ""
        assert response.message.tool_calls == []


def test_anthropic_client__extracts_system_messages_to_top_level_param() -> None:
    fake_client = _FakeAnthropicClient(response=_anthropic_response())
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    client.complete(
        [
            LLMMessage(role=Role.system, content="Message system A."),
            LLMMessage(role=Role.user, content="Hello"),
            LLMMessage(role=Role.system, content="Message system B."),
        ],
        system="Runtime system.",
    )

    call = fake_client.messages.calls[0]
    assert call["system"] == "Runtime system.\n\nMessage system A.\n\nMessage system B."
    assert call["messages"] == [{"role": "user", "content": "Hello"}]


def test_anthropic_client__serializes_tool_role_messages_as_user_tool_result() -> None:
    fake_client = _FakeAnthropicClient(response=_anthropic_response())
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    client.complete(
        [
            LLMMessage(
                role=Role.tool,
                content='{"ok": true}',
                tool_call_id="toolu-1",
            )
        ]
    )

    assert fake_client.messages.calls[0]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu-1",
                    "content": '{"ok": true}',
                }
            ],
        }
    ]


def test_anthropic_client__serializes_assistant_tool_calls_as_tool_use_blocks() -> None:
    fake_client = _FakeAnthropicClient(response=_anthropic_response())
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    client.complete(
        [
            LLMMessage(
                role=Role.assistant,
                content="Reading file.",
                tool_calls=[
                    ToolCall(
                        id="toolu-1",
                        name="read_file",
                        arguments={"path": "README.md"},
                    )
                ],
            )
        ]
    )

    assert fake_client.messages.calls[0]["messages"] == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Reading file."},
                {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
            ],
        }
    ]


def test_anthropic_client__translates_openai_tool_schemas() -> None:
    fake_client = _FakeAnthropicClient(response=_anthropic_response())
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    client.complete([LLMMessage(role=Role.user, content="Read")], tools=TOOL_SCHEMAS)

    assert fake_client.messages.calls[0]["tools"] == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": {"type": "object"},
        },
        {
            "name": "search_code",
            "description": "Search code.",
            "input_schema": {"type": "object"},
        },
    ]


def test_anthropic_client__wraps_sdk_errors() -> None:
    fake_client = _FakeAnthropicClient(error=RuntimeError("sdk down"))
    client = AnthropicClient(model="claude-opus-4-8", client=fake_client)

    with pytest.raises(LLMError, match="sdk down"):
        client.complete([LLMMessage(role=Role.user, content="Hello")])


def test_build_llm_client__uses_anthropic_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_clients = []

    class _FakeAnthropic:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key
            self.messages = _FakeAnthropicMessages(response=_anthropic_response())
            created_clients.append(self)

    monkeypatch.setattr(llm_client_module.anthropic, "Anthropic", _FakeAnthropic)
    monkeypatch.setenv("REPOPILOT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-anthropic")
    monkeypatch.setenv("REPOPILOT_MODEL", "claude-fable-5")
    settings = Settings(_env_file=None)

    client = build_llm_client(settings)

    assert isinstance(client, AnthropicClient)
    assert created_clients[0].api_key == "sk-test-anthropic"
    client.complete([LLMMessage(role=Role.user, content="Hello")])
    assert created_clients[0].messages.calls[0]["model"] == "claude-fable-5"


def test_llm_message__rejects_role_specific_field_violations() -> None:
    tool_call = ToolCall(id="call-1", name="read_file", arguments={})

    with pytest.raises(ValidationError):
        LLMMessage(role=Role.user, tool_calls=[tool_call])

    with pytest.raises(ValidationError):
        LLMMessage(role=Role.assistant, tool_call_id="call-1")


def test_llm_response__requires_assistant_message() -> None:
    with pytest.raises(ValidationError):
        LLMResponse(
            message=LLMMessage(role=Role.user, content="Nope."),
            stop_reason=StopReason.end_turn,
            usage=Usage(tokens_in=0, tokens_out=0),
            model="test-model",
            raw_finish_reason="stop",
        )
