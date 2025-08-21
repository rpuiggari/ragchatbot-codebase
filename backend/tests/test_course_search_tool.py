"""
Tests for CourseSearchTool functionality.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool execute method and related functionality."""

    def test_init(self, mock_vector_store):
        """Test CourseSearchTool initialization."""
        tool = CourseSearchTool(mock_vector_store)
        assert tool.store == mock_vector_store
        assert tool.last_sources == []

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly structured."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Validate schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_basic_query(self, mock_vector_store, mock_search_results):
        """Test basic query execution with successful results."""
        # Setup mock
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Verify search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=None
        )

        # Verify result formatting
        assert "[Introduction to Artificial Intelligence - Lesson 1]" in result
        assert "This is content from AI course lesson 1" in result
        assert len(tool.last_sources) == 3

        # Check source structure
        source = tool.last_sources[0]
        assert source["text"] == "Introduction to Artificial Intelligence - Lesson 1"
        assert "url" not in source  # No URL in this mock

    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test query execution with course name filtering."""
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="AI Course")

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="AI Course", lesson_number=None
        )
        assert result is not None

    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test query execution with lesson number filtering."""
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name=None, lesson_number=2
        )
        assert result is not None

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test query execution with both course and lesson filters."""
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="AI Course", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="test query", course_name="AI Course", lesson_number=2
        )
        assert result is not None

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results."""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("non-existent query")

        assert result == "No relevant content found."
        assert len(tool.last_sources) == 0

    def test_execute_empty_results_with_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test handling of empty results with filtering information."""
        mock_vector_store.search.return_value = empty_search_results

        tool = CourseSearchTool(mock_vector_store)

        # Test with course filter
        result = tool.execute("test", course_name="NonExistent")
        assert "No relevant content found in course 'NonExistent'." in result

        # Test with lesson filter
        result = tool.execute("test", lesson_number=999)
        assert "No relevant content found in lesson 999." in result

        # Test with both filters
        result = tool.execute("test", course_name="NonExistent", lesson_number=999)
        assert (
            "No relevant content found in course 'NonExistent' in lesson 999." in result
        )

    def test_execute_search_error(self, mock_vector_store, error_search_results):
        """Test handling of search errors."""
        mock_vector_store.search.return_value = error_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        assert result == "Search service unavailable"
        assert len(tool.last_sources) == 0

    def test_get_lesson_link_success(self, mock_vector_store, course_catalog_metadata):
        """Test successful lesson link retrieval."""
        # Mock the course catalog get method
        mock_vector_store.course_catalog.get.return_value = course_catalog_metadata

        tool = CourseSearchTool(mock_vector_store)
        link = tool._get_lesson_link("Introduction to Artificial Intelligence", 1)

        assert link == "https://example.com/ai-lesson1"
        mock_vector_store.course_catalog.get.assert_called_once_with(
            ids=["Introduction to Artificial Intelligence"]
        )

    def test_get_lesson_link_no_metadata(self, mock_vector_store):
        """Test lesson link retrieval when no metadata exists."""
        mock_vector_store.course_catalog.get.return_value = None

        tool = CourseSearchTool(mock_vector_store)
        link = tool._get_lesson_link("NonExistent Course", 1)

        assert link is None

    def test_get_lesson_link_no_lesson_match(
        self, mock_vector_store, course_catalog_metadata
    ):
        """Test lesson link retrieval when lesson number doesn't match."""
        mock_vector_store.course_catalog.get.return_value = course_catalog_metadata

        tool = CourseSearchTool(mock_vector_store)
        link = tool._get_lesson_link("Introduction to Artificial Intelligence", 999)

        assert link is None

    def test_get_lesson_link_json_error(self, mock_vector_store):
        """Test lesson link retrieval with malformed JSON."""
        bad_metadata = {
            "metadatas": [{"title": "Test Course", "lessons_json": "invalid json"}]
        }
        mock_vector_store.course_catalog.get.return_value = bad_metadata

        tool = CourseSearchTool(mock_vector_store)
        link = tool._get_lesson_link("Test Course", 1)

        assert link is None

    def test_get_lesson_link_exception_handling(self, mock_vector_store):
        """Test lesson link retrieval exception handling."""
        mock_vector_store.course_catalog.get.side_effect = Exception("Database error")

        tool = CourseSearchTool(mock_vector_store)
        link = tool._get_lesson_link("Test Course", 1)

        assert link is None

    def test_format_results_with_links(self, mock_vector_store):
        """Test result formatting with lesson links."""
        # Create search results
        results = SearchResults(
            documents=["Content from lesson 1", "Content from lesson 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Test Course", "lesson_number": 2, "chunk_index": 1},
            ],
            distances=[0.1, 0.2],
        )

        # Mock lesson link retrieval
        tool = CourseSearchTool(mock_vector_store)
        tool._get_lesson_link = Mock()
        tool._get_lesson_link.side_effect = [
            "https://example.com/lesson1",  # First call
            "https://example.com/lesson2",  # Second call
        ]

        formatted = tool._format_results(results)

        # Verify formatting
        assert "[Test Course - Lesson 1]" in formatted
        assert "[Test Course - Lesson 2]" in formatted
        assert "Content from lesson 1" in formatted
        assert "Content from lesson 2" in formatted

        # Verify sources with URLs
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["url"] == "https://example.com/lesson1"
        assert tool.last_sources[1]["text"] == "Test Course - Lesson 2"
        assert tool.last_sources[1]["url"] == "https://example.com/lesson2"

    def test_format_results_without_lesson_numbers(self, mock_vector_store):
        """Test result formatting when lesson numbers are None."""
        results = SearchResults(
            documents=["General course content"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": None, "chunk_index": 0}
            ],
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)

        assert "[Test Course]" in formatted
        assert "General course content" in formatted

        # Source should not have lesson info
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert "url" not in tool.last_sources[0]

    def test_format_results_unknown_course(self, mock_vector_store):
        """Test result formatting with unknown course metadata."""
        results = SearchResults(
            documents=["Some content"],
            metadata=[{"chunk_index": 0}],  # Missing course_title and lesson_number
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vector_store)
        formatted = tool._format_results(results)

        assert "[unknown]" in formatted
        assert "Some content" in formatted

    def test_multiple_result_formatting(self, mock_vector_store):
        """Test formatting of multiple search results."""
        results = SearchResults(
            documents=[
                "First piece of content",
                "Second piece of content",
                "Third piece of content",
            ],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Course B", "lesson_number": 2, "chunk_index": 1},
                {"course_title": "Course A", "lesson_number": 3, "chunk_index": 2},
            ],
            distances=[0.1, 0.2, 0.3],
        )

        tool = CourseSearchTool(mock_vector_store)
        tool._get_lesson_link = Mock(return_value=None)  # No links for simplicity

        formatted = tool._format_results(results)

        # Should contain all three results with proper headers
        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted
        assert "[Course A - Lesson 3]" in formatted
        assert "First piece of content" in formatted
        assert "Second piece of content" in formatted
        assert "Third piece of content" in formatted

        # Should have proper source tracking
        assert len(tool.last_sources) == 3

    def test_source_reset_between_searches(
        self, mock_vector_store, mock_search_results
    ):
        """Test that sources are properly reset between searches."""
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)

        # First search
        result1 = tool.execute("first query")
        sources1_count = len(tool.last_sources)
        assert sources1_count > 0

        # Second search with different results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results

        result2 = tool.execute("second query")
        assert len(tool.last_sources) == 0  # Should be reset to empty

    @pytest.mark.parametrize(
        "query,course,lesson,expected_calls",
        [
            ("test", None, None, 1),
            ("test", "course", None, 1),
            ("test", None, 1, 1),
            ("test", "course", 1, 1),
            ("", None, None, 1),  # Empty query should still call search
        ],
    )
    def test_execute_parameter_combinations(
        self,
        mock_vector_store,
        mock_search_results,
        query,
        course,
        lesson,
        expected_calls,
    ):
        """Test execute method with various parameter combinations."""
        mock_vector_store.search.return_value = mock_search_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query, course_name=course, lesson_number=lesson)

        assert mock_vector_store.search.call_count == expected_calls
        mock_vector_store.search.assert_called_with(
            query=query, course_name=course, lesson_number=lesson
        )
