"""
Tests for AI Generator functionality and tool calling.
"""

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator and tool calling functionality."""

    def test_init(self, test_config):
        """Test AIGenerator initialization."""
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        assert generator.model == test_config.ANTHROPIC_MODEL
        assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch("anthropic.Anthropic")
    def test_generate_response_without_tools(
        self, mock_anthropic_class, test_config, mock_anthropic_response
    ):
        """Test basic response generation without tools."""
        # Setup mock client
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response("What is AI?")

        # Verify API call
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args[1]

        assert call_args["model"] == test_config.ANTHROPIC_MODEL
        assert call_args["messages"][0]["content"] == "What is AI?"
        assert call_args["messages"][0]["role"] == "user"
        assert "tools" not in call_args

        assert result == "This is a test AI response."

    @patch("anthropic.Anthropic")
    def test_generate_response_with_tools(
        self,
        mock_anthropic_class,
        test_config,
        mock_anthropic_response,
        mock_tool_manager,
    ):
        """Test response generation with tools available."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        tools = mock_tool_manager.get_tool_definitions()
        result = generator.generate_response(
            "What is covered in lesson 1?", tools=tools
        )

        # Verify API call includes tools
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}

        assert result == "This is a test AI response."

    @patch("anthropic.Anthropic")
    def test_generate_response_with_conversation_history(
        self, mock_anthropic_class, test_config, mock_anthropic_response
    ):
        """Test response generation with conversation history."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        history = "User: Previous question\nAI: Previous answer"
        result = generator.generate_response(
            "Follow-up question", conversation_history=history
        )

        # Verify history is included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content

        assert result == "This is a test AI response."

    @patch("anthropic.Anthropic")
    def test_tool_execution_flow(
        self,
        mock_anthropic_class,
        test_config,
        mock_anthropic_tool_response,
        mock_tool_manager,
    ):
        """Test tool execution when Claude decides to use tools."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First call returns tool use, second call returns final response
        final_response = Mock()
        final_response.content = [Mock(text="Final answer after tool use")]

        mock_client.messages.create.side_effect = [
            mock_anthropic_tool_response,
            final_response,
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        tools = mock_tool_manager.get_tool_definitions()
        result = generator.generate_response(
            "What is covered in lesson 1?", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="test query"
        )

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

        assert result == "Final answer after tool use"

    @patch("anthropic.Anthropic")
    def test_handle_tool_execution_multiple_tools(
        self, mock_anthropic_class, test_config, mock_tool_manager
    ):
        """Test handling multiple tool calls in one response."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create mock response with multiple tool calls
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"

        tool1 = Mock()
        tool1.type = "tool_use"
        tool1.name = "search_course_content"
        tool1.id = "tool_1"
        tool1.input = {"query": "first query"}

        tool2 = Mock()
        tool2.type = "tool_use"
        tool2.name = "get_course_outline"
        tool2.id = "tool_2"
        tool2.input = {"course_name": "AI Course"}

        initial_response.content = [tool1, tool2]

        final_response = Mock()
        final_response.content = [Mock(text="Final response")]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        # Test the tool execution method directly
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system",
            "model": "test-model",
        }

        mock_client.messages.create.return_value = final_response
        mock_tool_manager.execute_tool.side_effect = ["result1", "result2"]

        result = generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Verify both tools were called
        expected_calls = [
            call("search_course_content", query="first query"),
            call("get_course_outline", course_name="AI Course"),
        ]
        mock_tool_manager.execute_tool.assert_has_calls(expected_calls)

        assert result == "Final response"

    @patch("anthropic.Anthropic")
    def test_api_error_handling(self, mock_anthropic_class, test_config):
        """Test handling of Anthropic API errors."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        with pytest.raises(Exception) as exc_info:
            generator.generate_response("Test query")

        assert "API Error" in str(exc_info.value)

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling(
        self,
        mock_anthropic_class,
        test_config,
        mock_anthropic_tool_response,
        mock_tool_manager,
    ):
        """Test handling of tool execution errors."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Tool execution fails
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        final_response = Mock()
        final_response.content = [Mock(text="Response despite tool error")]

        mock_client.messages.create.side_effect = [
            mock_anthropic_tool_response,
            final_response,
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        tools = mock_tool_manager.get_tool_definitions()

        # Should still handle gracefully and make final API call
        with pytest.raises(Exception) as exc_info:
            generator.generate_response(
                "Test", tools=tools, tool_manager=mock_tool_manager
            )

        assert "Tool execution failed" in str(exc_info.value)

    def test_system_prompt_content(self, test_config):
        """Test that system prompt contains expected guidance."""
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )

        system_prompt = generator.SYSTEM_PROMPT

        # Check for key instruction elements
        assert "Course outline/structure queries" in system_prompt
        assert "Content-specific questions" in system_prompt
        assert "General knowledge questions" in system_prompt
        assert "Sequential reasoning" in system_prompt  # Updated for sequential tools
        assert "Brief, Concise and focused" in system_prompt

    @patch("anthropic.Anthropic")
    def test_message_construction(
        self, mock_anthropic_class, test_config, mock_anthropic_response
    ):
        """Test proper message structure construction."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        query = "Test query"
        history = "Previous conversation"

        generator.generate_response(query, conversation_history=history)

        call_args = mock_client.messages.create.call_args[1]

        # Verify message structure
        messages = call_args["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == query

        # Verify system prompt includes history
        system = call_args["system"]
        assert generator.SYSTEM_PROMPT in system
        assert history in system

    @patch("anthropic.Anthropic")
    def test_tool_result_message_format(
        self, mock_anthropic_class, test_config, mock_tool_manager
    ):
        """Test that tool results are properly formatted in messages."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create initial tool use response
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"
        initial_response.content = [
            Mock(
                type="tool_use",
                name="search_course_content",
                id="tool_123",
                input={"query": "test"},
            )
        ]

        final_response = Mock()
        final_response.content = [Mock(text="Final answer")]
        mock_client.messages.create.return_value = final_response

        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        base_params = {
            "messages": [{"role": "user", "content": "original query"}],
            "system": "system prompt",
            "model": "test-model",
        }

        generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Check the final API call messages
        final_call_args = mock_client.messages.create.call_args[1]
        messages = final_call_args["messages"]

        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"  # Original query
        assert messages[1]["role"] == "assistant"  # Tool use
        assert messages[2]["role"] == "user"  # Tool results

        # Check tool result format
        tool_results = messages[2]["content"]
        assert isinstance(tool_results, list)
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Tool result"

    @pytest.mark.parametrize(
        "query,should_use_tools",
        [
            ("What is covered in lesson 1?", True),
            ("Tell me about the course outline", True),
            ("Show me MCP development content", True),
            ("What is artificial intelligence?", False),  # General knowledge
            ("How does machine learning work?", False),  # General knowledge
            ("Explain deep learning concepts", False),  # General knowledge
        ],
    )
    @patch("anthropic.Anthropic")
    def test_tool_usage_decision(
        self,
        mock_anthropic_class,
        test_config,
        mock_anthropic_response,
        query,
        should_use_tools,
    ):
        """Test that AI makes correct decisions about when to use tools."""
        # Note: This test verifies the system prompt contains guidance,
        # but actual tool usage decisions are made by Claude based on the prompt
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        tools = [{"name": "search_course_content", "description": "Search courses"}]
        generator.generate_response(query, tools=tools if should_use_tools else None)

        call_args = mock_client.messages.create.call_args[1]

        if should_use_tools:
            # Tools were provided - system should have guidance on when to use them
            system_prompt = call_args["system"]
            assert "Course outline/structure queries" in system_prompt
            assert "Content-specific questions" in system_prompt
        else:
            # No tools provided for general knowledge questions
            assert "tools" not in call_args

    @patch("anthropic.Anthropic")
    def test_empty_query_handling(
        self, mock_anthropic_class, test_config, mock_anthropic_response
    ):
        """Test handling of empty or None queries."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = mock_anthropic_response

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        # Test empty string
        result = generator.generate_response("")
        assert result == "This is a test AI response."

        # Verify empty query was passed through
        call_args = mock_client.messages.create.call_args[1]
        assert call_args["messages"][0]["content"] == ""
