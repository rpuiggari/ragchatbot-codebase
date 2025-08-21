from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Content Search Tool**: For searching specific course content and detailed educational materials
2. **Course Outline Tool**: For getting complete course structures, lesson lists, and course navigation information

Tool Usage Guidelines:
- **Course outline/structure queries**: Use the outline tool to get course title, course link, and complete lesson information
- **Content-specific questions**: Use the content search tool for detailed course material
- **General knowledge questions**: Answer using existing knowledge without tool usage
- **Sequential reasoning**: For complex queries, you may use up to 2 tool calls in sequence to gather comprehensive information
- **Multi-step queries**: Break complex questions into logical steps (e.g., search content first, then get course outline to provide context)
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Sequential Tool Usage Examples:
- **Cross-reference queries**: "What does lesson 3 cover and how does it fit in the overall course?" → Search lesson 3 content, then get course outline
- **Comparison queries**: "Compare topic X between two courses" → Search topic X in first course, then search topic X in second course
- **Context building**: "Explain advanced concept Y from course Z" → Get course outline to understand structure, then search for concept Y

Response Protocol:
- **Course outline requests** (e.g. "show me the lessons", "what's covered in this course"): Use outline tool first
- **Course-specific content questions**: Use content search tool first  
- **General knowledge questions**: Answer using existing knowledge without tools
- **Complex multi-part questions**: Use tools sequentially as needed (max 2 rounds)
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def generate_response_with_sequential_tools(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with support for sequential tool calling (up to max_rounds).

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """
        if not tools or not tool_manager:
            # Fall back to regular generation if no tools
            return self.generate_response(
                query, conversation_history, tools, tool_manager
            )

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history
        messages = [{"role": "user", "content": query}]

        # Execute sequential tool rounds
        for round_num in range(1, max_rounds + 1):
            # Prepare API call parameters for this round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": self._build_round_context(
                    system_content, round_num, max_rounds
                ),
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check if this response contains tool use and we haven't exceeded max rounds
            if response.stop_reason == "tool_use" and round_num < max_rounds:
                # Execute tools for this round and continue
                messages = self._execute_tool_round(messages, response, tool_manager)
            elif response.stop_reason == "tool_use" and round_num == max_rounds:
                # We've hit max rounds but Claude wants to use tools - execute tools then make final call
                messages = self._execute_tool_round(messages, response, tool_manager)
                break  # Exit loop to make final call
            else:
                # No tool use - return this response directly
                return response.content[0].text if response.content else ""

        # If we've exhausted all rounds, make final call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text if final_response.content else ""

    def _execute_tool_round(
        self, messages: List[Dict], response, tool_manager
    ) -> List[Dict]:
        """
        Execute tools for a single round and update message history.

        Args:
            messages: Current message history
            response: Claude's response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Updated message history including tool results
        """
        # Add AI's tool use response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}",
                        }
                    )

        # Add tool results as user message if we have any
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages

    def _should_continue_tool_rounds(
        self, response, current_round: int, max_rounds: int
    ) -> bool:
        """
        Determine if we should continue with more tool rounds.

        Args:
            response: Claude's response to check
            current_round: Current round number (1-indexed)
            max_rounds: Maximum allowed rounds

        Returns:
            True if we should continue, False otherwise
        """
        # Stop if we've reached max rounds
        if current_round >= max_rounds:
            return False

        # Stop if response doesn't contain tool use
        if response.stop_reason != "tool_use":
            return False

        # Stop if no tool_use content blocks found
        if not any(block.type == "tool_use" for block in response.content):
            return False

        return True

    def _build_round_context(
        self, base_system_prompt: str, round_num: int, max_rounds: int
    ) -> str:
        """
        Build system prompt with round context information.

        Args:
            base_system_prompt: Base system prompt
            round_num: Current round number (1-indexed)
            max_rounds: Maximum allowed rounds

        Returns:
            System prompt with round context
        """
        round_context = f"\n\nTool Round Context: This is round {round_num} of {max_rounds} maximum tool calling rounds."

        if round_num == max_rounds:
            round_context += (
                " This is your final opportunity to use tools - make it count!"
            )
        elif round_num == 1:
            round_context += " You may use tools now, and potentially one more round if needed for complex queries."

        return base_system_prompt + round_context
