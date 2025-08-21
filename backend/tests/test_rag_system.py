"""
Integration tests for RAG System functionality.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem


class TestRAGSystem:
    """Test suite for RAG System integration and end-to-end functionality."""

    @pytest.fixture
    def mock_rag_system(self, mock_rag_system_fixed):
        """Use the properly configured mock RAG system from conftest.py"""
        return mock_rag_system_fixed

    def test_init(self, test_config, mock_rag_system):
        """Test RAG System initialization."""
        rag = mock_rag_system

        assert rag.config == test_config
        assert rag.document_processor is not None
        assert rag.vector_store is not None
        assert rag.ai_generator is not None
        assert rag.session_manager is not None
        assert rag.tool_manager is not None
        assert rag.search_tool is not None
        assert rag.outline_tool is not None

    def test_query_without_session(self, mock_rag_system):
        """Test basic query processing without session management."""
        rag = mock_rag_system

        # Mock AI generator response
        rag.ai_generator.generate_response.return_value = "Test AI response"
        rag.tool_manager.get_last_sources.return_value = []

        response, sources = rag.query("What is AI?")

        # Verify AI generator was called correctly
        rag.ai_generator.generate_response.assert_called_once()
        call_args = rag.ai_generator.generate_response.call_args[1]

        assert (
            "Answer this question about course materials: What is AI?"
            in call_args["query"]
        )
        assert call_args["conversation_history"] is None
        assert call_args["tools"] == rag.tool_manager.get_tool_definitions.return_value
        assert call_args["tool_manager"] == rag.tool_manager

        assert response == "Test AI response"
        assert sources == []

        # Verify source management
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()

    def test_query_with_session(self, mock_rag_system):
        """Test query processing with session management."""
        rag = mock_rag_system

        # Mock session manager
        mock_history = "Previous conversation context"
        rag.session_manager.get_conversation_history.return_value = mock_history

        # Mock AI response
        rag.ai_generator.generate_response.return_value = "AI response with context"
        rag.tool_manager.get_last_sources.return_value = [{"text": "Source 1"}]

        response, sources = rag.query("Follow up question", session_id="test_session")

        # Verify session management
        rag.session_manager.get_conversation_history.assert_called_once_with(
            "test_session"
        )
        rag.session_manager.add_exchange.assert_called_once_with(
            "test_session", "Follow up question", "AI response with context"
        )

        # Verify AI generator got the history
        call_args = rag.ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] == mock_history

        assert response == "AI response with context"
        assert sources == [{"text": "Source 1"}]

    def test_query_with_sources(self, mock_rag_system):
        """Test query processing that returns sources from tool usage."""
        rag = mock_rag_system

        # Mock sources from tool execution
        mock_sources = [
            {"text": "Course A - Lesson 1", "url": "https://example.com/lesson1"},
            {"text": "Course B - Lesson 2", "url": "https://example.com/lesson2"},
        ]

        rag.ai_generator.generate_response.return_value = (
            "Answer based on course content"
        )
        rag.tool_manager.get_last_sources.return_value = mock_sources

        response, sources = rag.query("What is covered in lesson 1?")

        assert response == "Answer based on course content"
        assert sources == mock_sources

        # Verify sources were retrieved and reset
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()

    @pytest.mark.parametrize(
        "query,expected_prompt_contains",
        [
            ("What is AI?", "Answer this question about course materials: What is AI?"),
            (
                "Show me lesson 1",
                "Answer this question about course materials: Show me lesson 1",
            ),
            ("", "Answer this question about course materials: "),
        ],
    )
    def test_query_prompt_construction(
        self, mock_rag_system, query, expected_prompt_contains
    ):
        """Test that queries are properly wrapped in course materials context."""
        rag = mock_rag_system
        rag.ai_generator.generate_response.return_value = "Response"
        rag.tool_manager.get_last_sources.return_value = []

        rag.query(query)

        call_args = rag.ai_generator.generate_response.call_args[1]
        assert expected_prompt_contains in call_args["query"]

    def test_add_course_document_success(
        self, mock_rag_system, sample_course, sample_course_chunks
    ):
        """Test successful course document addition."""
        rag = mock_rag_system

        # Mock document processor
        rag.document_processor.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        course, chunk_count = rag.add_course_document("/path/to/course.txt")

        # Verify document processing
        rag.document_processor.process_course_document.assert_called_once_with(
            "/path/to/course.txt"
        )

        # Verify vector store operations
        rag.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        rag.vector_store.add_course_content.assert_called_once_with(
            sample_course_chunks
        )

        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

    def test_add_course_document_failure(self, mock_rag_system):
        """Test handling of document processing failure."""
        rag = mock_rag_system

        # Mock processing failure
        rag.document_processor.process_course_document.side_effect = Exception(
            "Processing failed"
        )

        course, chunk_count = rag.add_course_document("/invalid/path.txt")

        assert course is None
        assert chunk_count == 0

        # Vector store should not be called
        rag.vector_store.add_course_metadata.assert_not_called()
        rag.vector_store.add_course_content.assert_not_called()

    def test_add_course_folder_success(self, mock_rag_system, multiple_courses):
        """Test successful course folder processing."""
        rag = mock_rag_system

        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = ["course1.txt", "course2.pdf", "course3.docx", "ignored.jpg"]
            for filename in test_files:
                with open(os.path.join(temp_dir, filename), "w") as f:
                    f.write("test content")

            # Mock existing course titles (empty initially)
            rag.vector_store.get_existing_course_titles.return_value = []

            # Mock document processing for valid files
            course_results = [
                (
                    multiple_courses[0],
                    [
                        CourseChunk(
                            content="chunk1",
                            course_title=multiple_courses[0].title,
                            chunk_index=0,
                        )
                    ],
                ),
                (
                    multiple_courses[1],
                    [
                        CourseChunk(
                            content="chunk2",
                            course_title=multiple_courses[1].title,
                            chunk_index=0,
                        )
                    ],
                ),
                (
                    multiple_courses[2],
                    [
                        CourseChunk(
                            content="chunk3",
                            course_title=multiple_courses[2].title,
                            chunk_index=0,
                        )
                    ],
                ),
            ]

            rag.document_processor.process_course_document.side_effect = course_results

            total_courses, total_chunks = rag.add_course_folder(temp_dir)

            # Should process 3 valid files (txt, pdf, docx)
            assert rag.document_processor.process_course_document.call_count == 3
            assert total_courses == 3
            assert total_chunks == 3

            # Should add all courses to vector store
            assert rag.vector_store.add_course_metadata.call_count == 3
            assert rag.vector_store.add_course_content.call_count == 3

    def test_add_course_folder_with_existing_courses(
        self, mock_rag_system, multiple_courses
    ):
        """Test course folder processing with some existing courses."""
        rag = mock_rag_system

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            for i in range(3):
                with open(os.path.join(temp_dir, f"course{i}.txt"), "w") as f:
                    f.write("test content")

            # Mock existing course titles (one already exists)
            existing_titles = [multiple_courses[0].title]
            rag.vector_store.get_existing_course_titles.return_value = existing_titles

            # Mock document processing
            course_results = [
                (
                    multiple_courses[0],
                    [
                        CourseChunk(
                            content="chunk1",
                            course_title=multiple_courses[0].title,
                            chunk_index=0,
                        )
                    ],
                ),  # Existing
                (
                    multiple_courses[1],
                    [
                        CourseChunk(
                            content="chunk2",
                            course_title=multiple_courses[1].title,
                            chunk_index=0,
                        )
                    ],
                ),  # New
                (
                    multiple_courses[2],
                    [
                        CourseChunk(
                            content="chunk3",
                            course_title=multiple_courses[2].title,
                            chunk_index=0,
                        )
                    ],
                ),  # New
            ]

            rag.document_processor.process_course_document.side_effect = course_results

            total_courses, total_chunks = rag.add_course_folder(temp_dir)

            # Should process all files but only add 2 new courses
            assert rag.document_processor.process_course_document.call_count == 3
            assert total_courses == 2  # Only 2 new courses added
            assert total_chunks == 2  # Only 2 new sets of chunks

            # Should only add new courses to vector store
            assert rag.vector_store.add_course_metadata.call_count == 2
            assert rag.vector_store.add_course_content.call_count == 2

    def test_add_course_folder_clear_existing(self, mock_rag_system):
        """Test course folder processing with clear_existing=True."""
        rag = mock_rag_system

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file
            with open(os.path.join(temp_dir, "course.txt"), "w") as f:
                f.write("test content")

            rag.vector_store.get_existing_course_titles.return_value = []
            rag.document_processor.process_course_document.return_value = (
                Mock(title="Test Course"),
                [Mock()],
            )

            rag.add_course_folder(temp_dir, clear_existing=True)

            # Should clear existing data first
            rag.vector_store.clear_all_data.assert_called_once()

    def test_add_course_folder_nonexistent_path(self, mock_rag_system):
        """Test handling of nonexistent folder path."""
        rag = mock_rag_system

        total_courses, total_chunks = rag.add_course_folder("/nonexistent/path")

        assert total_courses == 0
        assert total_chunks == 0
        rag.document_processor.process_course_document.assert_not_called()

    def test_get_course_analytics(self, mock_rag_system):
        """Test course analytics retrieval."""
        rag = mock_rag_system

        # Mock vector store responses
        rag.vector_store.get_course_count.return_value = 5
        rag.vector_store.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
            "Course D",
            "Course E",
        ]

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course A" in analytics["course_titles"]

        rag.vector_store.get_course_count.assert_called_once()
        rag.vector_store.get_existing_course_titles.assert_called_once()

    def test_error_handling_in_query(self, mock_rag_system):
        """Test error handling during query processing."""
        rag = mock_rag_system

        # Mock AI generator failure
        rag.ai_generator.generate_response.side_effect = Exception(
            "AI service unavailable"
        )

        with pytest.raises(Exception) as exc_info:
            rag.query("Test query")

        assert "AI service unavailable" in str(exc_info.value)

    def test_session_management_error_handling(self, mock_rag_system):
        """Test error handling in session management."""
        rag = mock_rag_system

        # Mock session manager failure
        rag.session_manager.get_conversation_history.side_effect = Exception(
            "Session error"
        )
        rag.ai_generator.generate_response.return_value = "Response"
        rag.tool_manager.get_last_sources.return_value = []

        # Should handle session error gracefully and continue with query
        with pytest.raises(Exception) as exc_info:
            rag.query("Test query", session_id="failing_session")

        assert "Session error" in str(exc_info.value)

    @pytest.mark.integration
    def test_end_to_end_content_query_flow(self, mock_rag_system):
        """Test complete flow for content-specific query."""
        rag = mock_rag_system

        # Setup realistic mock responses
        mock_sources = [
            {"text": "AI Course - Lesson 1", "url": "https://example.com/ai-lesson1"}
        ]

        rag.ai_generator.generate_response.return_value = (
            "Here's what lesson 1 covers: basic AI concepts..."
        )
        rag.tool_manager.get_last_sources.return_value = mock_sources
        rag.session_manager.get_conversation_history.return_value = None

        # Execute content-specific query
        query = "What is covered in lesson 1 of the AI course?"
        session_id = "test_session_123"

        response, sources = rag.query(query, session_id=session_id)

        # Verify full flow
        assert response == "Here's what lesson 1 covers: basic AI concepts..."
        assert sources == mock_sources

        # Verify all components were called
        rag.session_manager.get_conversation_history.assert_called_once_with(session_id)
        rag.ai_generator.generate_response.assert_called_once()
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()
        rag.session_manager.add_exchange.assert_called_once_with(
            session_id, query, response
        )

        # Verify AI generator received tools and tool manager
        call_args = rag.ai_generator.generate_response.call_args[1]
        assert call_args["tools"] is not None
        assert call_args["tool_manager"] is not None

    @pytest.mark.integration
    def test_end_to_end_general_query_flow(self, mock_rag_system):
        """Test complete flow for general knowledge query."""
        rag = mock_rag_system

        # Setup mock for general knowledge response (no sources expected)
        rag.ai_generator.generate_response.return_value = (
            "Artificial intelligence is a field of computer science..."
        )
        rag.tool_manager.get_last_sources.return_value = (
            []
        )  # No sources for general knowledge

        # Execute general knowledge query
        query = "What is artificial intelligence?"

        response, sources = rag.query(query)

        # Verify response
        assert response == "Artificial intelligence is a field of computer science..."
        assert sources == []

        # Still should have tools available, but Claude should decide not to use them
        call_args = rag.ai_generator.generate_response.call_args[1]
        assert call_args["tools"] is not None  # Tools available
        assert call_args["tool_manager"] is not None  # Tool manager available

    def test_tool_manager_integration(self, mock_rag_system):
        """Test integration with tool manager."""
        rag = mock_rag_system

        # Verify tool manager has both tools available
        assert rag.search_tool is not None
        assert rag.outline_tool is not None
        assert rag.tool_manager is not None

        # Test tool manager methods are called during query
        rag.ai_generator.generate_response.return_value = "Test response"
        rag.tool_manager.get_tool_definitions.return_value = ["mock_tools"]
        rag.tool_manager.get_last_sources.return_value = []

        rag.query("Test query")

        # Verify tool manager methods were called
        rag.tool_manager.get_tool_definitions.assert_called_once()
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()

        # Verify the AI generator received the tool definitions
        ai_call_args = rag.ai_generator.generate_response.call_args[1]
        assert ai_call_args["tools"] == ["mock_tools"]
        assert ai_call_args["tool_manager"] == rag.tool_manager
