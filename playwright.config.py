import pytest
from playwright.sync_api import Playwright


def pytest_configure(config):
    """Configure pytest with Playwright settings."""
    pass


def pytest_addoption(parser):
    """Add command line options for Playwright tests."""
    group = parser.getgroup("playwright", "Playwright")
    group.addoption(
        "--browser",
        action="store",
        default="chromium",
        help="Browser to run tests with (chromium, firefox, webkit)",
    )
    group.addoption(
        "--headed", action="store_true", default=False, help="Run tests in headed mode"
    )
    group.addoption(
        "--slowmo",
        action="store",
        type=int,
        default=0,
        help="Slow down operations by specified milliseconds",
    )


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args, pytestconfig):
    """Configure browser context arguments."""
    return {
        **browser_context_args,
        "viewport": {"width": 1280, "height": 720},
        "ignore_https_errors": True,
    }


@pytest.fixture(scope="session")
def browser_args(pytestconfig):
    """Configure browser arguments."""
    return {
        "headless": not pytestconfig.getoption("--headed"),
        "slow_mo": pytestconfig.getoption("--slowmo"),
    }


@pytest.fixture(scope="session")
def base_url():
    """Base URL for the RAG chatbot application."""
    return "http://localhost:8000"
