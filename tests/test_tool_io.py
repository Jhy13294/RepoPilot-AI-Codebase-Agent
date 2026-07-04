import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from app.schemas.tool_io import ErrorType, ToolError, ToolMeta, ToolResult

EXPECTED_ERROR_NAMES = (
    "InvalidArgsError",
    "PathJailError",
    "NotFoundError",
    "BinaryFileError",
    "ToolTimeoutError",
    "PatchApplyError",
    "TestExecutionError",
    "ApprovalDeniedError",
    "InternalToolError",
)


class _SamplePayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    count: int


def test_tool_result__success_constructor_builds_success_result() -> None:
    meta = ToolMeta(tool_name="read_file", latency_ms=12)
    payload = _SamplePayload(path="app/config.py", count=1)

    result = ToolResult.success(data=payload, meta=meta)

    assert result.ok is True
    assert result.data == payload
    assert result.error is None
    assert result.meta == meta


def test_tool_result__failure_constructor_builds_failure_result() -> None:
    meta = ToolMeta(tool_name="read_file", latency_ms=12, truncated=True)
    error = ToolError(
        type=ErrorType.NotFoundError,
        message="Read a different path; this file does not exist.",
        details={"path": "missing.py"},
    )

    result = ToolResult.failure(error=error, meta=meta)

    assert result.ok is False
    assert result.data is None
    assert result.error == error
    assert result.meta == meta


def test_tool_result__rejects_success_with_error() -> None:
    meta = ToolMeta(tool_name="search_code", latency_ms=3)
    error = ToolError(type=ErrorType.InvalidArgsError, message="Use a valid regex pattern.")

    with pytest.raises(ValidationError):
        ToolResult(ok=True, error=error, meta=meta)


def test_tool_result__rejects_failure_without_error() -> None:
    meta = ToolMeta(tool_name="search_code", latency_ms=3)

    with pytest.raises(ValidationError):
        ToolResult(ok=False, meta=meta)


def test_tool_result__rejects_failure_with_data() -> None:
    meta = ToolMeta(tool_name="search_code", latency_ms=3)
    payload = _SamplePayload(path="app/config.py", count=1)
    error = ToolError(type=ErrorType.InternalToolError, message="Retry after re-reading context.")

    with pytest.raises(ValidationError):
        ToolResult(ok=False, data=payload, error=error, meta=meta)


def test_tool_meta__rejects_mutation_when_frozen() -> None:
    meta = ToolMeta(tool_name="read_file", latency_ms=12)

    with pytest.raises(ValidationError):
        meta.truncated = True


def test_tool_meta__rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        ToolMeta(tool_name="read_file", latency_ms=-1)


def test_error_type__members_match_documented_taxonomy() -> None:
    assert tuple(ErrorType.__members__) == EXPECTED_ERROR_NAMES
    assert tuple(error_type.value for error_type in ErrorType) == EXPECTED_ERROR_NAMES


def test_tool_result__round_trips_json_with_payload_model() -> None:
    class SamplePayload(BaseModel):
        model_config = ConfigDict(frozen=True)

        path: str
        count: int

    meta = ToolMeta(tool_name="get_file_tree", latency_ms=5)
    payload = SamplePayload(path="app/schemas/tool_io.py", count=2)
    result = ToolResult.success(data=payload, meta=meta)

    encoded = result.model_dump_json()
    restored = ToolResult.model_validate_json(encoded)

    assert restored.ok is True
    assert restored.error is None
    assert restored.meta == meta
    assert restored.data is not None
    assert restored.data.model_dump() == {"path": "app/schemas/tool_io.py", "count": 2}
