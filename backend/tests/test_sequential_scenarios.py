"""
End-to-end scenario tests for sequential tool calling functionality.
"""

from unittest.mock import Mock, patch

import pytest
from ai_generator import AIGenerator

# Test data for various sequential scenarios
SEQUENTIAL_SCENARIOS = [
    {
        "name": "search_then_outline",
        "query": "What does lesson 2 cover and how does it fit in the overall course?",
        "expected_tools": ["search_course_content", "get_course_outline"],
        "expected_rounds": 2,
        "description": "Search specific lesson content, then get course outline for context",
    },
    {
        "name": "outline_then_search",
        "query": "Show me the AI course structure, then find details about advanced topics",
        "expected_tools": ["get_course_outline", "search_course_content"],
        "expected_rounds": 2,
        "description": "Get course overview first, then search for specific content",
    },
    {
        "name": "comparison_query",
        "query": "Compare the introduction of the MCP course with the AI course introduction",
        "expected_tools": ["search_course_content", "search_course_content"],
        "expected_rounds": 2,
        "description": "Search same topic across different courses for comparison",
    },
    {
        "name": "context_building",
        "query": "Explain vector databases from the Chroma course and how they relate to the overall curriculum",
        "expected_tools": ["search_course_content", "get_course_outline"],
        "expected_rounds": 2,
        "description": "Search specific concept, then get context from course structure",
    },
]

ERROR_SCENARIOS = [
    {
        "name": "tool_failure_first_round",
        "failures": [
            {
                "round": 1,
                "tool": "search_course_content",
                "error": "Search service unavailable",
            }
        ],
        "expected_behavior": "continue_to_second_round",
    },
    {
        "name": "tool_failure_second_round",
        "failures": [
            {"round": 2, "tool": "get_course_outline", "error": "Outline service down"}
        ],
        "expected_behavior": "complete_with_partial_results",
    },
    {
        "name": "api_error_between_rounds",
        "failures": [{"round": 2, "type": "api_error", "error": "Rate limited"}],
        "expected_behavior": "return_error_with_context",
    },
]


class TestSequentialScenarios:
    """Test realistic sequential tool calling scenarios."""

    @pytest.fixture
    def scenario_mock_responses(self):
        """Mock API responses for different scenario types."""
        return {
            "search_then_outline": [
                # Round 1: Search for lesson content
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="search_course_content",
                            id="search_1",
                            input={"query": "lesson 2", "lesson_number": 2},
                        )
                    ],
                ),
                # Round 2: Get course outline for context
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="get_course_outline",
                            id="outline_1",
                            input={"course_name": "AI Course"},
                        )
                    ],
                ),
                # Final response
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="Lesson 2 covers machine learning basics including supervised and unsupervised learning. In the course structure, it builds on lesson 1's introduction and prepares students for lesson 3's deep dive into neural networks."
                        )
                    ],
                ),
            ],
            "outline_then_search": [
                # Round 1: Get course outline
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="get_course_outline",
                            id="outline_1",
                            input={"course_name": "AI Course"},
                        )
                    ],
                ),
                # Round 2: Search for advanced topics mentioned in outline
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="search_course_content",
                            id="search_1",
                            input={
                                "query": "advanced topics",
                                "course_name": "AI Course",
                            },
                        )
                    ],
                ),
                # Final response
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="The AI course covers 5 lessons total, with advanced topics in lessons 4-5 including reinforcement learning, neural architecture search, and deployment strategies."
                        )
                    ],
                ),
            ],
            "comparison_query": [
                # Round 1: Search MCP course intro
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="search_course_content",
                            id="search_1",
                            input={"query": "introduction", "course_name": "MCP"},
                        )
                    ],
                ),
                # Round 2: Search AI course intro
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="search_course_content",
                            id="search_2",
                            input={"query": "introduction", "course_name": "AI"},
                        )
                    ],
                ),
                # Final comparison
                Mock(
                    stop_reason="end_turn",
                    content=[
                        Mock(
                            text="Both courses start with foundational concepts, but MCP focuses on building context-aware applications while AI covers general machine learning principles."
                        )
                    ],
                ),
            ],
        }

    @pytest.fixture
    def scenario_tool_results(self):
        """Mock tool execution results for scenarios."""
        return {
            "search_course_content": {
                "lesson 2": "Machine learning basics: supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data.",
                "introduction MCP": "MCP (Model Context Protocol) enables AI applications to connect with external data sources and tools for richer context.",
                "introduction AI": "Artificial Intelligence involves creating systems that can perform tasks typically requiring human intelligence.",
                "advanced topics": "Advanced AI topics include reinforcement learning, neural architecture search, transfer learning, and model deployment.",
            },
            "get_course_outline": {
                "AI Course": "**AI Course**: 1) Introduction, 2) Machine Learning Basics, 3) Neural Networks, 4) Advanced Applications, 5) Deployment",
                "MCP Course": "**MCP Course**: 1) MCP Introduction, 2) Building Applications, 3) Context Integration, 4) Advanced Features",
            },
        }

    @pytest.mark.parametrize("scenario", SEQUENTIAL_SCENARIOS, ids=lambda s: s["name"])
    @patch("anthropic.Anthropic")
    def test_sequential_scenario(
        self,
        mock_anthropic_class,
        test_config,
        scenario,
        scenario_mock_responses,
        scenario_tool_results,
    ):
        """Test various sequential tool calling scenarios."""

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Setup mock responses for this scenario
        if scenario["name"] in scenario_mock_responses:
            mock_client.messages.create.side_effect = scenario_mock_responses[
                scenario["name"]
            ]
        else:
            # Default response pattern
            mock_client.messages.create.side_effect = [
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name=tool,
                            id=f"tool_{i}",
                            input={"query": "test"},
                        )
                    ],
                )
                for i, tool in enumerate(scenario["expected_tools"])
            ] + [
                Mock(
                    stop_reason="end_turn",
                    content=[Mock(text="Final scenario response")],
                )
            ]

        # Setup tool manager
        mock_tool_manager = Mock()

        def mock_execute_tool(tool_name, **kwargs):
            if tool_name == "search_course_content":
                # Return relevant results based on query
                query = kwargs.get("query", "")
                course = kwargs.get("course_name", "")
                key = f"{query} {course}".strip()
                return scenario_tool_results["search_course_content"].get(
                    key, f"Search results for: {query}"
                )
            elif tool_name == "get_course_outline":
                course = kwargs.get("course_name", "Unknown Course")
                return scenario_tool_results["get_course_outline"].get(
                    course, f"Outline for: {course}"
                )
            return "Default tool result"

        mock_tool_manager.execute_tool.side_effect = mock_execute_tool
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search courses"},
            {"name": "get_course_outline", "description": "Get course outline"},
        ]

        # Execute scenario
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query=scenario["query"],
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_rounds=scenario["expected_rounds"],
        )

        # Verify expected behavior
        assert (
            mock_client.messages.create.call_count == scenario["expected_rounds"] + 1
        )  # +1 for final response
        assert mock_tool_manager.execute_tool.call_count == scenario["expected_rounds"]

        # Verify tools were called in expected order
        tool_calls = mock_tool_manager.execute_tool.call_args_list
        for i, expected_tool in enumerate(scenario["expected_tools"]):
            actual_tool = tool_calls[i][0][0]  # First positional argument is tool name
            assert (
                actual_tool == expected_tool
            ), f"Round {i+1}: expected {expected_tool}, got {actual_tool}"

        # Verify response contains expected content (scenario-specific checks)
        assert result is not None
        assert len(result) > 0

    @pytest.mark.parametrize("error_scenario", ERROR_SCENARIOS, ids=lambda s: s["name"])
    @patch("anthropic.Anthropic")
    def test_error_scenarios(self, mock_anthropic_class, test_config, error_scenario):
        """Test error handling in various sequential scenarios."""

        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Setup responses with errors at specified points
        responses = []
        for round_num in range(1, 3):  # Up to 2 rounds
            responses.append(
                Mock(
                    stop_reason="tool_use",
                    content=[
                        Mock(
                            type="tool_use",
                            name="search_course_content",
                            id=f"tool_{round_num}",
                            input={"query": f"test{round_num}"},
                        )
                    ],
                )
            )
        responses.append(
            Mock(
                stop_reason="end_turn",
                content=[Mock(text="Final response despite errors")],
            )
        )

        mock_client.messages.create.side_effect = responses

        # Setup tool manager with failures
        mock_tool_manager = Mock()
        tool_results = []

        for failure in error_scenario["failures"]:
            if failure.get("tool"):
                # Tool execution failure
                if failure["round"] == 1:
                    tool_results.append(Exception(failure["error"]))
                    tool_results.append("Second tool success")
                else:
                    tool_results.append("First tool success")
                    tool_results.append(Exception(failure["error"]))

        if not tool_results:
            tool_results = ["Tool success", "Tool success"]  # Default success

        mock_tool_manager.execute_tool.side_effect = tool_results
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"}
        ]

        # Execute with error conditions
        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        if "api_error" in str(error_scenario["failures"]):
            # Simulate API error
            mock_client.messages.create.side_effect = [
                responses[0],
                Exception("API Rate Limited"),
                responses[-1],
            ]

            with pytest.raises(Exception) as exc_info:
                generator.generate_response_with_sequential_tools(
                    query="Test API error scenario",
                    tools=mock_tool_manager.get_tool_definitions(),
                    tool_manager=mock_tool_manager,
                    max_rounds=2,
                )
            assert "API Rate Limited" in str(exc_info.value)
        else:
            # Tool execution errors should be handled gracefully
            result = generator.generate_response_with_sequential_tools(
                query="Test tool error scenario",
                tools=mock_tool_manager.get_tool_definitions(),
                tool_manager=mock_tool_manager,
                max_rounds=2,
            )

            # Should complete despite tool errors
            assert result == "Final response despite errors"

            # Should have attempted all expected tool calls
            expected_calls = (
                len([f for f in error_scenario["failures"] if f.get("tool")]) or 2
            )
            assert mock_tool_manager.execute_tool.call_count == expected_calls

    @patch("anthropic.Anthropic")
    def test_complex_multi_course_comparison(self, mock_anthropic_class, test_config):
        """Test complex scenario comparing concepts across multiple courses."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Simulate comparing AI concepts across AI course and MCP course
        mock_client.messages.create.side_effect = [
            # Round 1: Search AI course for machine learning
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="search_course_content",
                        id="search_1",
                        input={"query": "machine learning", "course_name": "AI Course"},
                    )
                ],
            ),
            # Round 2: Search MCP course for machine learning applications
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="search_course_content",
                        id="search_2",
                        input={
                            "query": "machine learning applications",
                            "course_name": "MCP Course",
                        },
                    )
                ],
            ),
            # Final comparison response
            Mock(
                stop_reason="end_turn",
                content=[
                    Mock(
                        text="The AI course covers machine learning fundamentals and algorithms, while the MCP course focuses on practical applications of ML in context-aware systems."
                    )
                ],
            ),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "AI Course ML content: Supervised learning, unsupervised learning, model training and evaluation.",
            "MCP Course ML applications: Using trained models in applications, context integration, real-world deployment.",
        ]
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content"}
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="How does machine learning coverage differ between the AI course and MCP course?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        # Verify both courses were searched
        tool_calls = mock_tool_manager.execute_tool.call_args_list
        assert len(tool_calls) == 2

        # First call should search AI course
        assert "machine learning" in str(tool_calls[0])
        # Second call should search MCP course
        assert "machine learning applications" in str(tool_calls[1]) or "MCP" in str(
            tool_calls[1]
        )

        # Verify comparative response
        assert "AI course covers" in result
        assert "MCP course focuses" in result
        assert "fundamentals" in result or "applications" in result

    @patch("anthropic.Anthropic")
    def test_progressive_context_building(self, mock_anthropic_class, test_config):
        """Test scenario where each tool call builds on previous context."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Scenario: Get course outline first to understand structure, then search for specific advanced topic
        mock_client.messages.create.side_effect = [
            # Round 1: Get course outline to understand structure
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="get_course_outline",
                        id="outline_1",
                        input={"course_name": "AI Course"},
                    )
                ],
            ),
            # Round 2: Based on outline, search for specific advanced topic from later lessons
            Mock(
                stop_reason="tool_use",
                content=[
                    Mock(
                        type="tool_use",
                        name="search_course_content",
                        id="search_1",
                        input={
                            "query": "reinforcement learning",
                            "course_name": "AI Course",
                            "lesson_number": 4,
                        },
                    )
                ],
            ),
            # Final contextual response
            Mock(
                stop_reason="end_turn",
                content=[
                    Mock(
                        text="Reinforcement learning is covered in lesson 4 as an advanced topic. Based on the course structure, this builds on supervised/unsupervised learning from lessons 2-3 and prepares students for the deployment focus in lesson 5."
                    )
                ],
            ),
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course Outline: 1) Intro, 2) ML Basics, 3) Neural Networks, 4) Advanced Topics (RL, Transfer Learning), 5) Deployment",
            "Lesson 4 Reinforcement Learning: Agent-environment interaction, rewards, Q-learning, policy gradients, applications in games and robotics.",
        ]
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "get_course_outline"},
            {"name": "search_course_content"},
        ]

        generator = AIGenerator(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        generator.client = mock_client

        result = generator.generate_response_with_sequential_tools(
            query="Tell me about reinforcement learning in the AI course and how it fits in the overall learning progression",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_rounds=2,
        )

        # Verify progressive context building
        tool_calls = mock_tool_manager.execute_tool.call_args_list
        assert len(tool_calls) == 2

        # First should get outline for structure
        first_call = tool_calls[0][0][0]
        assert first_call == "get_course_outline"

        # Second should search specific content informed by outline
        second_call = tool_calls[1][0][0]
        assert second_call == "search_course_content"
        second_params = tool_calls[1][1]
        assert "reinforcement learning" in second_params.get("query", "")

        # Result should show contextual understanding
        assert "lesson 4" in result
        assert "builds on" in result or "prepares students" in result
        assert "course structure" in result or "learning progression" in result
