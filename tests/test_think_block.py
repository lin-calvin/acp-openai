from __future__ import annotations

from acp.schema import (
    AgentThoughtChunk,
    ContentToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from acp_openai_middleware.agent_manager import (
    _build_think_block,
    _format_tool_call,
    _tool_call_content_to_text,
)
from acp_openai_middleware.schemas import ChatMessage


class TestToolCallContentToText:
    def test_empty_list(self):
        assert _tool_call_content_to_text(None) == ""
        assert _tool_call_content_to_text([]) == ""

    def test_text_content(self):
        content = ContentToolCallContent(
            type="content",
            content=TextContentBlock(type="text", text="reading file"),
        )
        assert _tool_call_content_to_text([content]) == "reading file"

    def test_diff_content(self):
        from acp.schema import FileEditToolCallContent

        content = FileEditToolCallContent(
            type="diff",
            path="src/main.py",
            old_text="hello",
            new_text="world",
        )
        assert "src/main.py" in _tool_call_content_to_text([content])
        assert "hello" in _tool_call_content_to_text([content])
        assert "world" in _tool_call_content_to_text([content])

    def test_terminal_content(self):
        from acp.schema import TerminalToolCallContent

        content = TerminalToolCallContent(
            type="terminal",
            terminal_id="term_123",
        )
        assert "term_123" in _tool_call_content_to_text([content])

    def test_multiple_contents(self):
        c1 = ContentToolCallContent(
            type="content",
            content=TextContentBlock(type="text", text="first"),
        )
        c2 = ContentToolCallContent(
            type="content",
            content=TextContentBlock(type="text", text="second"),
        )
        result = _tool_call_content_to_text([c1, c2])
        assert "first" in result
        assert "second" in result
        assert " | " in result


class TestFormatToolCall:
    def test_toolcall_start_with_all_fields(self):
        tc = ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc_1",
            title="read_file",
            kind="read",
            status="in_progress",
        )
        result = _format_tool_call(tc)
        assert result == (
            "Tool call: read_file\n"
            "  kind: read\n"
            "  status: in_progress"
        )

    def test_toolcall_start_minimal(self):
        tc = ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc_1",
            title="search",
        )
        result = _format_tool_call(tc)
        assert result == "Tool call: search"

    def test_toolcall_progress(self):
        tc = ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="tc_1",
            title="read_file",
            status="completed",
        )
        result = _format_tool_call(tc)
        assert "completed" in result

    def test_toolcall_with_content(self):
        tc = ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc_1",
            title="write_file",
            kind="edit",
            content=[
                ContentToolCallContent(
                    type="content",
                    content=TextContentBlock(type="text", text="saved to disk"),
                )
            ],
        )
        result = _format_tool_call(tc)
        assert "saved to disk" in result

    def test_toolcall_with_raw_input_dict(self):
        tc = ToolCallStart(
            session_update="tool_call",
            tool_call_id="tc_1",
            title="search",
            kind="search",
            raw_input={"pattern": "foo", "path": "/tmp"},
        )
        result = _format_tool_call(tc)
        assert "input:" in result
        assert "    pattern: foo" in result
        assert "    path: /tmp" in result

    def test_toolcall_with_raw_output_scalar(self):
        tc = ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="tc_1",
            title="search",
            status="completed",
            raw_output="found 5 matches",
        )
        result = _format_tool_call(tc)
        assert "output: found 5 matches" in result
        assert "completed" in result

    def test_toolcall_with_input_list_and_output_scalar(self):
        tc = ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id="tc_1",
            title="grep",
            kind="search",
            status="completed",
            raw_input=["rg", "pattern", "."],
            raw_output="3 results",
        )
        result = _format_tool_call(tc)
        assert "input:" in result
        assert "    - rg" in result
        assert "    - pattern" in result
        assert "    - ." in result
        assert "output: 3 results" in result


class TestFormatYamlValue:
    def test_dict(self):
        from acp_openai_middleware.agent_manager import _format_yaml_value

        value = {"key": "val"}
        result = _format_yaml_value(value, indent=1)
        assert result == "    key: val"

    def test_nested_dict(self):
        from acp_openai_middleware.agent_manager import _format_yaml_value

        value = {"outer": {"inner": "v"}}
        result = _format_yaml_value(value, indent=1)
        assert result == "    outer:\n      inner: v"

    def test_list(self):
        from acp_openai_middleware.agent_manager import _format_yaml_value

        value = [1, 2, 3]
        result = _format_yaml_value(value, indent=1)
        assert result == "    - 1\n    - 2\n    - 3"

    def test_empty_dict(self):
        from acp_openai_middleware.agent_manager import _format_yaml_value

        assert _format_yaml_value({}, indent=1) == "{}"

    def test_empty_list(self):
        from acp_openai_middleware.agent_manager import _format_yaml_value

        assert _format_yaml_value([], indent=1) == "[]"


class TestBuildThinkBlock:
    def test_empty(self):
        assert _build_think_block([]) == ""

    def test_single_entry(self):
        entry = "Tool call: read_file\n  kind: read"
        result = _build_think_block([entry])
        assert result == "<think>\nTool call: read_file\n  kind: read\n</think>\n\n"

    def test_multiple_entries(self):
        parts = [
            "Tool call: read_file\n  kind: read\n  status: in_progress",
            "Tool call: read_file\n  kind: read\n  status: completed",
            "Tool call: search\n  kind: search\n  status: in_progress",
            "Tool call: search\n  kind: search\n  status: completed",
        ]
        result = _build_think_block(parts)
        assert result.startswith("<think>\n")
        assert result.endswith("\n</think>\n\n")
        for part in parts:
            assert part in result


class TestThinkPartsIntegration:
    def test_agent_thought_chunk_text(self):
        chunk = AgentThoughtChunk(
            session_update="agent_thought_chunk",
            content=TextContentBlock(type="text", text="I should read the file first"),
        )
        assert isinstance(chunk.content, TextContentBlock)
        assert chunk.content.text == "I should read the file first"

    def test_clear_pending_clears_think_parts(self):
        from acp_openai_middleware.agent_manager import NamespaceState

        ns = NamespaceState.__new__(NamespaceState)
        ns._pending_chunks = []
        ns._last_usage = None
        ns._think_parts = ["some thought"]
        ns.clear_pending()
        assert ns._think_parts == []
