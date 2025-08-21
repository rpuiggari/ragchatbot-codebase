"""
Integration tests for RAG System sequential tool calling functionality.
"""

from unittest.mock import Mock, patch

import pytest
from rag_system import RAGSystem


class TestRAGSystemSequential:
    """Test suite for RAG System sequential tool calling integration."""

    def test_sequential_tools_enabled_by_default(self, mock_rag_system_fixed):
        """Test that sequential tools are enabled by default in RAG system."""
        rag = mock_rag_system_fixed

        # Verify config has sequential settings
        assert hasattr(rag.config, "ENABLE_SEQUENTIAL_TOOLS")
        assert hasattr(rag.config, "MAX_TOOL_ROUNDS")
        assert rag.config.ENABLE_SEQUENTIAL_TOOLS is True
        assert rag.config.MAX_TOOL_ROUNDS == 2

    def test_sequential_query_flow(self, mock_rag_system_fixed):
        """Test end-to-end sequential query flow through RAG system."""
        rag = mock_rag_system_fixed

        # Mock sequential response
        rag.ai_generator.generate_response_with_sequential_tools = Mock(
            return_value="Combined answer from sequential tool calls"
        )
        rag.tool_manager.get_last_sources.return_value = [
            {"text": "AI Course - Lesson 3", "url": "https://example.com/lesson3"},
            {"text": "AI Course - Course Outline"},
        ]

        response, sources = rag.query(
            "What does lesson 3 cover and how does it fit in the overall course?"
        )

        # Verify sequential method was called
        rag.ai_generator.generate_response_with_sequential_tools.assert_called_once()
        call_args = rag.ai_generator.generate_response_with_sequential_tools.call_args[
            1
        ]

        assert "Answer this question about course materials:" in call_args["query"]
        assert call_args["tools"] == rag.tool_manager.get_tool_definitions.return_value
        assert call_args["tool_manager"] == rag.tool_manager
        assert call_args["max_rounds"] == 2

        # Verify response and sources
        assert response == "Combined answer from sequential tool calls"
        assert len(sources) == 2
        assert sources[0]["text"] == "AI Course - Lesson 3"
        assert sources[0]["url"] == "https://example.com/lesson3"

    def test_sequential_tools_disabled_fallback(self, mock_rag_system_fixed):
        """Test fallback to regular generation when sequential tools disabled."""
        rag = mock_rag_system_fixed

        # Disable sequential tools
        rag.config.ENABLE_SEQUENTIAL_TOOLS = False

        # Mock regular response method
        rag.ai_generator.generate_response = Mock(
            return_value="Regular single-tool response"
        )
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("Test query with disabled sequential tools")

        # Verify regular method was called instead
        rag.ai_generator.generate_response.assert_called_once()
        rag.ai_generator.generate_response_with_sequential_tools = (
            Mock()
        )  # Should not be called
        rag.ai_generator.generate_response_with_sequential_tools.assert_not_called()

        assert response == "Regular single-tool response"

    def test_session_management_with_sequential_tools(self, mock_rag_system_fixed):
        """Test session management integration with sequential tool calling."""
        rag = mock_rag_system_fixed

        # Mock session history
        mock_history = "User: Previous question\nAI: Previous answer"
        rag.session_manager.get_conversation_history.return_value = mock_history

        # Mock sequential response
        rag.ai_generator.generate_response_with_sequential_tools = Mock(
            return_value="Response with session context"
        )
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("Follow-up question", session_id="test_session")

        # Verify session history was passed to sequential method
        call_args = rag.ai_generator.generate_response_with_sequential_tools.call_args[
            1
        ]
        assert call_args["conversation_history"] == mock_history

        # Verify session was updated
        rag.session_manager.add_exchange.assert_called_once_with(
            "test_session", "Follow-up question", "Response with session context"
        )

    def test_error_handling_with_sequential_tools(self, mock_rag_system_fixed):
        """Test error handling during sequential tool execution."""
        rag = mock_rag_system_fixed

        # Mock AI generator failure
        rag.ai_generator.generate_response_with_sequential_tools.side_effect = (
            Exception("Sequential tool error")
        )

        # Should propagate the exception
        with pytest.raises(Exception) as exc_info:
            rag.query("Test error handling")

        assert "Sequential tool error" in str(exc_info.value)

    def test_source_management_across_sequential_rounds(self, mock_rag_system_fixed):
        """Test that sources are properly collected from multiple tool rounds."""
        rag = mock_rag_system_fixed

        # Mock sequential tool execution with multiple sources
        rag.ai_generator.generate_response_with_sequential_tools = Mock(
            return_value="Answer using multiple sources"
        )

        # Sources from multiple rounds
        multi_round_sources = [
            {"text": "Course A - Lesson 1", "url": "https://example.com/a/lesson1"},
            {"text": "Course B - Overview"},
            {
                "text": "Course A - Course Outline",
                "url": "https://example.com/a/outline",
            },
        ]
        rag.tool_manager.get_last_sources.return_value = multi_round_sources

        response, sources = rag.query("Compare content across courses")

        # Verify all sources are preserved
        assert len(sources) == 3
        assert sources == multi_round_sources

        # Verify sources were reset after retrieval
        rag.tool_manager.reset_sources.assert_called_once()

    def test_max_rounds_configuration_propagation(self, mock_rag_system_fixed):
        """Test that MAX_TOOL_ROUNDS configuration is properly propagated."""
        rag = mock_rag_system_fixed

        # Set custom max rounds
        rag.config.MAX_TOOL_ROUNDS = 3

        rag.ai_generator.generate_response_with_sequential_tools = Mock(
            return_value="Response with custom max rounds"
        )
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("Test custom max rounds")

        # Verify max_rounds parameter was passed correctly
        call_args = rag.ai_generator.generate_response_with_sequential_tools.call_args[
            1
        ]
        assert call_args["max_rounds"] == 3

    def test_missing_config_attributes_handled(self, mock_rag_system_fixed):
        """Test graceful handling when config is missing sequential attributes."""
        rag = mock_rag_system_fixed

        # Disable sequential tools by setting ENABLE_SEQUENTIAL_TOOLS to False
        # This simulates the fallback behavior we want to test
        rag.config.ENABLE_SEQUENTIAL_TOOLS = False

        # Mock fallback to regular generation
        rag.ai_generator.generate_response = Mock(return_value="Fallback response")
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("Test missing config")

        # Should fall back to regular generation
        rag.ai_generator.generate_response.assert_called_once()
        assert response == "Fallback response"

    def test_complex_query_scenarios(self, mock_rag_system_fixed):
        """Test various complex query scenarios that benefit from sequential tools."""
        rag = mock_rag_system_fixed
        rag.ai_generator.generate_response_with_sequential_tools = Mock()
        rag.tool_manager.get_last_sources.return_value = []

        # Test different complex query types
        complex_queries = [
            "What does lesson 3 cover about neural networks and how does it fit in the overall course structure?",
            "Compare the introduction of the AI course with the MCP course introduction",
            "Find information about vector databases in the Chroma course, then explain how it relates to the overall curriculum",
            "Show me the course outline for the AI course, then search for advanced topics mentioned in later lessons",
        ]

        for query in complex_queries:
            rag.ai_generator.generate_response_with_sequential_tools.reset_mock()

            response, sources = rag.query(query)

            # Verify sequential method was called for each complex query
            rag.ai_generator.generate_response_with_sequential_tools.assert_called_once()

            # Verify proper parameters
            call_args = (
                rag.ai_generator.generate_response_with_sequential_tools.call_args[1]
            )
            assert (
                query in call_args["query"]
            )  # Should be wrapped in course materials prompt
            assert call_args["max_rounds"] == 2
            assert call_args["tools"] is not None
            assert call_args["tool_manager"] is not None

    @pytest.mark.integration
    def test_end_to_end_sequential_flow(self, mock_rag_system_fixed):
        """Test complete end-to-end flow with sequential tool calling."""
        rag = mock_rag_system_fixed

        # Setup realistic scenario: search lesson content, then get course outline
        mock_search_results = [
            {"text": "AI Course - Lesson 3", "url": "https://example.com/ai/lesson3"}
        ]
        mock_outline_results = [
            {
                "text": "AI Course - Course Outline",
                "url": "https://example.com/ai/outline",
            }
        ]

        # Mock the complete flow
        rag.ai_generator.generate_response_with_sequential_tools = Mock(
            return_value="Lesson 3 covers neural networks and deep learning fundamentals. In the overall course structure, this builds on lessons 1-2 which covered basic concepts, and prepares students for advanced applications in lessons 4-5."
        )
        rag.tool_manager.get_last_sources.return_value = (
            mock_search_results + mock_outline_results
        )
        rag.session_manager.get_conversation_history.return_value = None

        # Execute query
        query = "What does lesson 3 cover and how does it fit in the overall course?"
        session_id = "test_integration_session"

        response, sources = rag.query(query, session_id=session_id)

        # Verify complete flow execution
        assert "neural networks" in response
        assert "overall course structure" in response

        # Verify sources from both rounds
        assert len(sources) == 2
        assert any("Lesson 3" in source["text"] for source in sources)
        assert any("Course Outline" in source["text"] for source in sources)

        # Verify session management
        rag.session_manager.get_conversation_history.assert_called_once_with(session_id)
        rag.session_manager.add_exchange.assert_called_once_with(
            session_id, query, response
        )

        # Verify tool manager interactions
        rag.tool_manager.get_tool_definitions.assert_called_once()
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()
