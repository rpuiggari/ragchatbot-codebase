"""
Tests for AI Generator sequential tool calling functionality.
"""

import time
from unittest.mock import Mock, call, patch

import pytest
from ai_generator import AIGenerator


class TestAIGeneratorSequential:
    """Test suite for AIGenerator sequential tool calling functionality."""

    @pytest.fixture
    def mock_sequential_responses(self):
        """Create mock responses for sequential tool calling scenarios."""
        # Create mock tool content blocks with proper name attributes
        tool_1 = Mock()
        tool_1.type = "tool_use"
        tool_1.name = "search_course_content"
        tool_1.id = "tool_1"
        tool_1.input = {"query": "neural networks", "lesson_number": 3}

        tool_2 = Mock()
        tool_2.type = "tool_use"
        tool_2.name = "get_course_outline"
        tool_2.id = "tool_2"
        tool_2.input = {"course_name": "AI Course"}

        tool_3 = Mock()
        tool_3.type = "tool_use"
        tool_3.name = "search_course_content"
        tool_3.id = "tool_3"
        tool_3.input = {"query": "introduction"}

        tool_4 = Mock()
        tool_4.type = "tool_use"
        tool_4.name = "search_course_content"
        tool_4.id = "tool_4"
        tool_4.input = {"query": "topic1"}

        tool_5 = Mock()
        tool_5.type = "tool_use"
        tool_5.name = "search_course_content"
        tool_5.id = "tool_5"
        tool_5.input = {"query": "topic2"}

        return {
            "two_round_sequence": [
                # Round 1: First tool call
                Mock(stop_reason="tool_use", content=[tool_1]),
                # Round 2: Second tool call
                Mock(stop_reason="tool_use", content=[tool_2]),
                # Final response
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="Neural networks in lesson 3 cover deep learning fundamentals and fit into the overall course as an advanced topic building on earlier concepts."
                        )
                    ],
                ),
            ],
            "single_tool_only": [
                # Single tool call then done
                Mock(stop_reason="tool_use", content=[tool_3]),
                # Final response
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="The introduction covers basic AI concepts and terminology."
                        )
                    ],
                ),
            ],
            "max_rounds_exceeded": [
                # Round 1
                Mock(stop_reason="tool_use", content=[tool_4]),
                # Round 2 (max reached)
                Mock(stop_reason="tool_use", content=[tool_5]),
                # Final response (no more tools allowed)
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="Based on the information gathered from both searches, here's the comparison."
                        )
                    ],
                ),
            ],
            "no_tools_needed": [
                # Direct answer without tools
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="This is a general knowledge answer that doesn't require course-specific tools."
                        )
                    ],
                )
            ],
        }

    @pytest.fixture
    def mock_sequential_tool_manager(self):
        """Mock tool manager for sequential scenarios."""
        mock_manager = Mock()
        mock_manager.execution_log = []

        def mock_execute_tool(tool_name, **kwargs):
            execution_entry = {
                "tool": tool_name,
                "params": kwargs,
                "timestamp": time.time(),
            }
            mock_manager.execution_log.append(execution_entry)

            if tool_name == "search_course_content":
                query = kwargs.get("query", "unknown")
                return f"Search results for: {query}"
            elif tool_name == "get_course_outline":
                course = kwargs.get("course_name", "unknown")
                return f"Course outline for: {course}"
            return "Mock tool result"

        mock_manager.execute_tool.side_effect = mock_execute_tool
        mock_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get course outline"},
        ]

        return mock_manager

    @patch("anthropic.Anthropic")
    def test_sequential_two_round_flow(
        self,
        mock_anthropic_class,
        test_config,
        mock_sequential_responses,
        mock_sequential_tool_manager,
    ):
        """Test basic two-round sequential tool calling."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = mock_sequential_responses[
            "two_round_sequence"
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="What does lesson 3 cover about neural networks and how does it fit in the overall course?",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Verify API calls were made for both rounds + final
        assert mock_client.messages.create.call_count == 3

        # Verify tools were executed in sequence
        assert mock_sequential_tool_manager.execute_tool.call_count == 2
        tool_calls = mock_sequential_tool_manager.execute_tool.call_args_list

        # First call should be search
        assert tool_calls[0][0][0] == "search_course_content"
        assert tool_calls[0][1]["query"] == "neural networks"

        # Second call should be outline
        assert tool_calls[1][0][0] == "get_course_outline"
        assert tool_calls[1][1]["course_name"] == "AI Course"

        # Verify final response
        assert "Neural networks in lesson 3" in result
        assert "fit into the overall course" in result

    @patch("anthropic.Anthropic")
    def test_single_round_completion(
        self,
        mock_anthropic_class,
        test_config,
        mock_sequential_responses,
        mock_sequential_tool_manager,
    ):
        """Test that single tool call completes without forcing second round."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = mock_sequential_responses[
            "single_tool_only"
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="What does the introduction cover?",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Should only make 2 API calls (tool round + final response)
        assert mock_client.messages.create.call_count == 2

        # Should only execute one tool
        assert mock_sequential_tool_manager.execute_tool.call_count == 1

        # Verify correct response
        assert "introduction covers basic AI concepts" in result

    @patch("anthropic.Anthropic")
    def test_max_rounds_enforcement(
        self,
        mock_anthropic_class,
        test_config,
        mock_sequential_responses,
        mock_sequential_tool_manager,
    ):
        """Test that max rounds limit is enforced."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = mock_sequential_responses[
            "max_rounds_exceeded"
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="Compare topic1 with topic2",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Should make exactly 3 calls: round1 + round2 + final
        assert mock_client.messages.create.call_count == 3

        # Should execute exactly 2 tools (hitting the limit)
        assert mock_sequential_tool_manager.execute_tool.call_count == 2

        # Verify response indicates completion
        assert "Based on the information gathered" in result

    @patch("anthropic.Anthropic")
    def test_no_tools_needed_flow(
        self,
        mock_anthropic_class,
        test_config,
        mock_sequential_responses,
        mock_sequential_tool_manager,
    ):
        """Test flow when Claude decides no tools are needed."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = mock_sequential_responses[
            "no_tools_needed"
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="What is artificial intelligence?",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Should only make 1 API call (Claude decides no tools needed)
        assert mock_client.messages.create.call_count == 1

        # Should not execute any tools
        assert mock_sequential_tool_manager.execute_tool.call_count == 0

        # Should return general knowledge response
        assert "general knowledge answer" in result

    @patch("anthropic.Anthropic")
    def test_tool_execution_error_handling(
        self, mock_anthropic_class, test_config, mock_sequential_tool_manager
    ):
        """Test handling of tool execution errors in sequential rounds."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First round: tool call
        # Second round: another tool call
        # Final: answer despite error
        mock_client.messages.create.side_effect = [
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="search_course_content",
                        id="tool_1",
                        input={"query": "test"},
                    )
                ],
            ),
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="get_course_outline",
                        id="tool_2",
                        input={"course_name": "test"},
                    )
                ],
            ),
            Mock(
                stop_reason="end_turn",
                content=[Mock(text="Partial answer despite tool error")],
            ),
        ]

        # Mock first tool success, second tool failure
        mock_sequential_tool_manager.execute_tool.side_effect = [
            "First tool success",
            Exception("Tool execution failed"),
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="Test error handling",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Should complete all rounds despite tool error
        assert mock_client.messages.create.call_count == 3
        assert mock_sequential_tool_manager.execute_tool.call_count == 2

        # Should return final response
        assert "Partial answer despite tool error" in result

    @patch("anthropic.Anthropic")
    def test_conversation_history_integration(
        self, mock_anthropic_class, test_config, mock_sequential_tool_manager
    ):
        """Test that conversation history is properly integrated in sequential calls."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            Mock(
                stop_reason="end_turn",
                content=[Mock(text="Response with history context")],
            )
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        conversation_history = "User: Previous question\nAI: Previous answer"

        result = generator.generate_response_with_sequential_tools(
            query="Follow-up question",
            conversation_history=conversation_history,
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Verify history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        assert "Previous conversation:" in system_content
        assert conversation_history in system_content

        assert result == "Response with history context"

    @patch("anthropic.Anthropic")
    def test_round_context_in_system_prompt(
        self, mock_anthropic_class, test_config, mock_sequential_tool_manager
    ):
        """Test that round context is added to system prompts."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = [
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="search_course_content",
                        id="tool_1",
                        input={"query": "test"},
                    )
                ],
            ),
            Mock(stop_reason="end_turn", content=[Mock(text="Final answer")]),
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="Test round context",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=2,
        )

        # Check that both API calls included round context
        calls = mock_client.messages.create.call_args_list

        # First round should mention it's round 1 of 2
        first_system = calls[0][1]["system"]
        assert "round 1 of 2" in first_system
        assert "potentially one more round" in first_system

        # Second round should mention it's round 2 of 2 (final)
        second_system = calls[1][1]["system"]
        assert "round 2 of 2" in second_system
        assert "final opportunity" in second_system

    def test_should_continue_tool_rounds_logic(self, test_config):
        """Test the termination logic for tool rounds."""
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )

        # Test max rounds reached
        response_with_tools = Mock(
            stop_reason="tool_use", content=[Mock(type="tool_use")]
        )
        assert not generator._should_continue_tool_rounds(
            response_with_tools, 2, 2
        )  # At max
        assert generator._should_continue_tool_rounds(
            response_with_tools, 1, 2
        )  # Under max

        # Test no tool use
        response_without_tools = Mock(
            stop_reason="end_turn", content=[Mock(type="text")]
        )
        assert not generator._should_continue_tool_rounds(response_without_tools, 1, 2)

        # Test empty content
        response_empty = Mock(stop_reason="tool_use", content=[])
        assert not generator._should_continue_tool_rounds(response_empty, 1, 2)

        # Test mixed content with tool_use
        response_mixed = Mock(
            stop_reason="tool_use", content=[Mock(type="text"), Mock(type="tool_use")]
        )
        assert generator._should_continue_tool_rounds(response_mixed, 1, 2)

    def test_build_round_context_formatting(self, test_config):
        """Test round context formatting for different scenarios."""
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        base_prompt = "Base system prompt"

        # Test first round
        context_1 = generator._build_round_context(base_prompt, 1, 2)
        assert "round 1 of 2" in context_1
        assert "potentially one more round" in context_1
        assert base_prompt in context_1

        # Test final round
        context_2 = generator._build_round_context(base_prompt, 2, 2)
        assert "round 2 of 2" in context_2
        assert "final opportunity" in context_2
        assert base_prompt in context_2

    @patch("anthropic.Anthropic")
    def test_fallback_to_regular_generation(self, mock_anthropic_class, test_config):
        """Test fallback when no tools or tool_manager provided."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.return_value = Mock(
            content=[Mock(text="Regular response")]
        )

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        # Mock the regular generate_response method
        generator.generate_response = Mock(return_value="Fallback response")

        # Test with no tools
        result = generator.generate_response_with_sequential_tools(
            query="Test query", tools=None, tool_manager=None, max_rounds=2
        )

        # Should have called the regular method
        generator.generate_response.assert_called_once_with(
            "Test query", None, None, None
        )
        assert result == "Fallback response"

    @pytest.mark.parametrize(
        "max_rounds,expected_calls",
        [
            (1, 2),  # 1 round + final response
            (2, 3),  # 2 rounds + final response
            (3, 4),  # 3 rounds + final response (if we allowed it)
        ],
    )
    @patch("anthropic.Anthropic")
    def test_different_max_rounds_settings(
        self,
        mock_anthropic_class,
        test_config,
        mock_sequential_tool_manager,
        max_rounds,
        expected_calls,
    ):
        """Test behavior with different max_rounds settings."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Create responses for all possible rounds
        responses = []
        for i in range(max_rounds):
            responses.append(
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="search_course_content",
                            id=f"tool_{i}",
                            input={"query": f"test{i}"},
                        )
                    ],
                )
            )
        responses.append(
            Mock(stop_reason="end_turn", content=[Mock(text="Final response")])
        )

        mock_client.messages.create.side_effect = responses

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="Test max rounds",
            tools=mock_sequential_tool_manager.get_tool_definitions(),
            tool_manager=mock_sequential_tool_manager,
            max_rounds=max_rounds,
        )

        # Should make exactly the expected number of API calls
        assert mock_client.messages.create.call_count == expected_calls

        # Should execute exactly max_rounds tools
        assert mock_sequential_tool_manager.execute_tool.call_count == max_rounds

        assert result == "Final response"
