import subprocess
import time
from pathlib import Path

import pytest
import requests


@pytest.fixture(scope="session", autouse=True)
def start_server():
    """Start the RAG chatbot server before running tests."""
    # Change to backend directory
    backend_dir = Path(__file__).parent.parent / "backend"

    # Start the server
    server_process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app:app", "--port", "8000"],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            if i < max_retries - 1:
                time.sleep(1)
                continue
            else:
                server_process.terminate()
                raise Exception("Server failed to start")

    yield

    # Cleanup: terminate server
    server_process.terminate()
    server_process.wait()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context for all tests."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }
