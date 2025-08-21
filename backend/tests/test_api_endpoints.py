"""
API endpoint tests for the RAG system FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json

from config import Config


@pytest.fixture
def test_app():
    """Create a test FastAPI app without static file mounts."""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Union
    
    # Import the models from app.py
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        url: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Union[str, SourceItem]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # Create test app
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Mock RAG system
    mock_rag_system = Mock()
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            # Create session if not provided
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            # Mock query processing
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}
    
    # Store mock for test access
    app.state.mock_rag_system = mock_rag_system
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture
def mock_rag_responses():
    """Mock responses for RAG system."""
    return {
        "query_response": (
            "This is a test response about AI fundamentals.",
            [
                "Source 1: Introduction to AI concepts",
                {"text": "Source 2: AI applications", "url": "https://example.com/lesson2"}
            ]
        ),
        "analytics": {
            "total_courses": 3,
            "course_titles": [
                "Introduction to Artificial Intelligence",
                "Advanced Retrieval for AI with Chroma",
                "MCP: Build Rich-Context AI Apps with Anthropic"
            ]
        }
    }


class TestQueryEndpoint:
    """Test the /api/query endpoint."""
    
    def test_query_success(self, client, test_app, mock_rag_responses):
        """Test successful query processing."""
        # Setup mock
        mock_rag = test_app.state.mock_rag_system
        mock_rag.query.return_value = mock_rag_responses["query_response"]
        mock_rag.session_manager.create_session.return_value = "new-session-456"
        
        # Make request
        response = client.post("/api/query", json={
            "query": "What is artificial intelligence?"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response about AI fundamentals."
        assert len(data["sources"]) == 2
        assert data["session_id"] == "new-session-456"
        
        # Verify mock was called correctly
        mock_rag.query.assert_called_once_with("What is artificial intelligence?", "new-session-456")
    
    def test_query_with_session_id(self, client, test_app, mock_rag_responses):
        """Test query with existing session ID."""
        # Setup mock
        mock_rag = test_app.state.mock_rag_system
        mock_rag.query.return_value = mock_rag_responses["query_response"]
        
        # Make request with session ID
        response = client.post("/api/query", json={
            "query": "Tell me about machine learning",
            "session_id": "existing-session-789"
        })
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-789"
        
        # Verify mock was called with existing session
        mock_rag.query.assert_called_once_with("Tell me about machine learning", "existing-session-789")
    
    def test_query_empty_request(self, client):
        """Test query with empty request body."""
        response = client.post("/api/query", json={})
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_query_field(self, client):
        """Test query with missing query field."""
        response = client.post("/api/query", json={
            "session_id": "test-session"
        })
        assert response.status_code == 422  # Validation error
    
    def test_query_internal_error(self, client, test_app):
        """Test query endpoint with internal error."""
        # Setup mock to raise exception
        mock_rag = test_app.state.mock_rag_system
        mock_rag.query.side_effect = Exception("Database connection failed")
        mock_rag.session_manager.create_session.return_value = "error-session"
        
        # Make request
        response = client.post("/api/query", json={
            "query": "What is AI?"
        })
        
        # Verify error response
        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]
    
    def test_query_malformed_json(self, client):
        """Test query with malformed JSON."""
        response = client.post("/api/query", data="invalid json")
        assert response.status_code == 422


class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""
    
    def test_courses_success(self, client, test_app, mock_rag_responses):
        """Test successful course stats retrieval."""
        # Setup mock
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = mock_rag_responses["analytics"]
        
        # Make request
        response = client.get("/api/courses")
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Artificial Intelligence" in data["course_titles"]
        
        # Verify mock was called
        mock_rag.get_course_analytics.assert_called_once()
    
    def test_courses_internal_error(self, client, test_app):
        """Test courses endpoint with internal error."""
        # Setup mock to raise exception
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.side_effect = Exception("Vector store unavailable")
        
        # Make request
        response = client.get("/api/courses")
        
        # Verify error response
        assert response.status_code == 500
        assert "Vector store unavailable" in response.json()["detail"]
    
    def test_courses_no_parameters_needed(self, client, test_app, mock_rag_responses):
        """Test that courses endpoint doesn't require parameters."""
        # Setup mock
        mock_rag = test_app.state.mock_rag_system
        mock_rag.get_course_analytics.return_value = mock_rag_responses["analytics"]
        
        # Make request with query parameters (should be ignored)
        response = client.get("/api/courses?unused=param")
        
        # Should still work
        assert response.status_code == 200


class TestRootEndpoint:
    """Test the root / endpoint."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns basic info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Course Materials RAG System" in data["message"]


class TestRequestResponseModels:
    """Test request and response model validation."""
    
    def test_source_item_model(self, client, test_app):
        """Test SourceItem model with and without URL."""
        # Setup mock with mixed source types
        mock_rag = test_app.state.mock_rag_system
        mock_rag.query.return_value = (
            "Test answer",
            [
                "Plain text source",
                {"text": "Source with URL", "url": "https://example.com/lesson"}
            ]
        )
        mock_rag.session_manager.create_session.return_value = "test-session"
        
        # Make request
        response = client.post("/api/query", json={
            "query": "test query"
        })
        
        # Verify mixed source types are handled correctly
        assert response.status_code == 200
        data = response.json()
        sources = data["sources"]
        
        assert len(sources) == 2
        assert sources[0] == "Plain text source"
        assert isinstance(sources[1], dict)
        assert sources[1]["text"] == "Source with URL"
        assert sources[1]["url"] == "https://example.com/lesson"
    
    def test_query_request_validation(self, client):
        """Test QueryRequest model validation."""
        # Test with valid request
        response = client.post("/api/query", json={
            "query": "valid query",
            "session_id": "optional-session"
        })
        assert response.status_code != 422  # Should not be validation error
        
        # Test with invalid types
        response = client.post("/api/query", json={
            "query": 123,  # Should be string
            "session_id": "valid"
        })
        assert response.status_code == 422
        
        response = client.post("/api/query", json={
            "query": "valid",
            "session_id": 123  # Should be string or null
        })
        assert response.status_code == 422


class TestCORSAndMiddleware:
    """Test CORS and middleware functionality."""
    
    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        response = client.options("/api/query")
        
        # Check for CORS headers (TestClient may not set all headers)
        # This is more of a smoke test since TestClient doesn't fully simulate browser CORS
        assert response.status_code in [200, 405]  # OPTIONS may not be implemented
    
    def test_trusted_host_middleware(self, client):
        """Test that requests are accepted (trusted host allows all)."""
        response = client.get("/")
        assert response.status_code == 200


class TestEndToEndScenarios:
    """Test realistic end-to-end scenarios."""
    
    def test_query_course_workflow(self, client, test_app):
        """Test a realistic workflow: get courses, then query."""
        mock_rag = test_app.state.mock_rag_system
        
        # Setup analytics mock
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 2,
            "course_titles": ["AI Course", "ML Course"]
        }
        
        # Setup query mock
        mock_rag.query.return_value = (
            "The AI course covers neural networks and deep learning.",
            ["Course content about neural networks"]
        )
        mock_rag.session_manager.create_session.return_value = "workflow-session"
        
        # First, get available courses
        courses_response = client.get("/api/courses")
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        assert courses_data["total_courses"] == 2
        
        # Then query about a specific course
        query_response = client.post("/api/query", json={
            "query": "What does the AI course teach about neural networks?"
        })
        assert query_response.status_code == 200
        query_data = query_response.json()
        assert "neural networks" in query_data["answer"]
        
        # Verify both endpoints were called
        mock_rag.get_course_analytics.assert_called_once()
        mock_rag.query.assert_called_once()
    
    def test_session_persistence(self, client, test_app):
        """Test session persistence across multiple queries."""
        mock_rag = test_app.state.mock_rag_system
        mock_rag.query.return_value = ("Response", ["Source"])
        
        # First query creates session
        mock_rag.session_manager.create_session.return_value = "persistent-session"
        response1 = client.post("/api/query", json={"query": "First question"})
        session_id = response1.json()["session_id"]
        
        # Second query uses same session
        response2 = client.post("/api/query", json={
            "query": "Follow up question",
            "session_id": session_id
        })
        
        assert response2.json()["session_id"] == session_id
        
        # Verify calls were made with correct session IDs
        calls = mock_rag.query.call_args_list
        assert len(calls) == 2
        assert calls[0][0][1] == "persistent-session"  # First call
        assert calls[1][0][1] == "persistent-session"  # Second call