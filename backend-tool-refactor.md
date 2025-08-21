
# Refactor `@backend/ai_generator.py` for Sequential Tool Calling

Refactor `@backend/ai_generator.py` to support **sequential tool calling** where Claude can make up to **2 tool calls** in separate API rounds.

---

## Current Behavior
- Claude makes 1 tool call → tools are removed from API params → final response.  
- If Claude wants another tool call after seeing results, it can’t (gets empty response).  

---

## Desired Behavior
- Each tool call should be a **separate API request** where Claude can reason about previous results.  
- Support for **complex queries** requiring:
  - Multiple searches for comparisons,  
  - Multi-part questions,  
  - Information drawn from different courses/lessons.  

### Example Flow
1. **User:** “Search for a course that discusses the same topic as lesson 4 of course X”  
2. **Claude:** Get course outline for course X → gets title of lesson 4  
3. **Claude:** Uses the title to search for a course that discusses the same topic → returns course information  
4. **Claude:** Provides complete answer  

---

## Requirements
- Maximum **2 sequential rounds** per user query.  
- Terminate when:  
  - (a) 2 rounds completed  
  - (b) Claude's response has no `tool_use` blocks  
  - (c) Tool call fails  
- Preserve **conversation context** between rounds.  
- Handle **tool execution errors** gracefully.  

---

## Notes
- Update the system prompt in `@backend/ai_generator.py`.  
- Update the test in `@backend/tests/test_ai_generator.py`.  
- Write tests that verify **external behavior** (API calls made, tools executed, results returned) rather than internal state details.  


Do you want me to also add a “Next Steps” section with bullet points on how to implement (code + tests), so it’s more like a dev task spec?


Use two parallel subagents to brainstorm possible plans. Do not implement any code.