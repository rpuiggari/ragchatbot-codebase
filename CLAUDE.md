# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Running

This is a RAG (Retrieval-Augmented Generation) chatbot system for course materials using Python 3.13+, uv package manager, and Anthropic's Claude API.

### Essential Commands

```bash
# Install dependencies
uv sync

# Start application (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000

# Access at http://localhost:8000 (web) or http://localhost:8000/docs (API)
```

### Code Quality Tools

```bash
# Format code (black + isort)
./scripts/format.sh

# Run all quality checks (black, isort, flake8, mypy, tests)
./scripts/quality.sh

# Individual tools
uv run black .                    # Format code
uv run isort .                    # Sort imports
uv run black --check --diff .     # Check formatting
uv run isort --check-only --diff . # Check import sorting
uv run flake8 .                   # Lint code
uv run mypy backend/ main.py       # Type checking
cd backend && uv run pytest       # Run tests
```

### Environment Setup
- Create `.env` file with `ANTHROPIC_API_KEY=your_api_key_here`
- Application automatically loads course documents from `docs/` folder on startup

## Architecture Overview

This system implements a **tool-based RAG architecture** where Claude decides when and how to search, rather than always retrieving context upfront.

### Core Components

**RAGSystem** (`rag_system.py`): Main orchestrator that coordinates all components and handles the complete query flow from user input to AI response.

**Tool-Based Search** (`search_tools.py`): Implements `CourseSearchTool` that Claude can invoke to search course content. Uses abstract `Tool` interface for extensibility.

**Dual Vector Storage** (`vector_store.py`): ChromaDB with two collections:
- `course_catalog`: Course metadata for semantic course name matching
- `course_content`: Text chunks with course/lesson context

**Document Processing** (`document_processor.py`): Parses structured course documents following this format:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [url]
[content...]
```

**AI Generator** (`ai_generator.py`): Anthropic Claude API integration with tool calling support. Handles tool execution flow and conversation history.

**Session Management** (`session_manager.py`): Maintains conversation context across queries with configurable history limits.

### Key Architectural Patterns

**Smart Chunking**: Text is split on sentence boundaries with configurable overlap (800 chars, 100 overlap). Chunks include course title and lesson context for better retrieval.

**Tool Decision Making**: Claude autonomously decides when to search based on query type. General knowledge questions don't trigger search; course-specific queries do.

**Course Name Resolution**: Supports partial course name matching through semantic search in the course catalog before content search.

**Context Enhancement**: Retrieved chunks are prefixed with course and lesson information to improve AI understanding.

## Data Flow

1. **Document Ingestion**: Files → DocumentProcessor → Course/CourseChunk objects → VectorStore (dual collections)
2. **Query Processing**: User query → RAGSystem → AIGenerator with tools → Optional tool execution → Claude response  
3. **Tool Execution**: CourseSearchTool → VectorStore semantic search → Context passed to Claude → Final answer

## Configuration

All configuration centralized in `config.py` using dataclasses:
- Model: claude-sonnet-4-20250514
- Embedding: all-MiniLM-L6-v2
- ChromaDB path: `./chroma_db` 
- Chunk size/overlap: 800/100 characters
- Max search results: 5
- Conversation history: 2 exchanges

## Development Notes

**No formal test suite exists** - tests should be added for core components, especially document processing and vector operations.

**Frontend is static files** served by FastAPI with cache-busting headers for development. Main logic in `frontend/script.js`.

**Vector database persists** in `backend/chroma_db/` and is initialized with documents from `docs/` on startup.

**Error handling** includes UTF-8 encoding fallbacks, graceful search failures, and comprehensive API error responses.

When adding new course documents, they must follow the structured format above. The system automatically processes `.txt`, `.pdf`, and `.docx` files.

For extending search capabilities, implement the `Tool` interface in `search_tools.py` and register with `ToolManager`.