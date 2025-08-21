import time

import pytest
from playwright.sync_api import Page, expect


class TestRAGChatbot:
    """Test suite for the RAG chatbot frontend."""

    def test_page_loads(self, page: Page, base_url: str):
        """Test that the main page loads successfully."""
        page.goto(base_url)
        expect(page).to_have_title("RAG Chatbot")

    def test_ui_elements_present(self, page: Page, base_url: str):
        """Test that all required UI elements are present."""
        page.goto(base_url)

        # Check for main elements
        expect(page.locator("h1")).to_contain_text("RAG Chatbot")
        expect(page.locator("#user-input")).to_be_visible()
        expect(page.locator("#send-btn")).to_be_visible()
        expect(page.locator("#chat-messages")).to_be_visible()

    def test_send_message(self, page: Page, base_url: str):
        """Test sending a message to the chatbot."""
        page.goto(base_url)

        # Type a test message
        test_message = "Hello, this is a test message"
        page.fill("#user-input", test_message)

        # Click send button
        page.click("#send-btn")

        # Wait for message to appear in chat
        expect(page.locator(".user-message")).to_contain_text(test_message)

    def test_chatbot_response(self, page: Page, base_url: str):
        """Test that the chatbot responds to messages."""
        page.goto(base_url)

        # Send a general question
        test_message = "What is artificial intelligence?"
        page.fill("#user-input", test_message)
        page.click("#send-btn")

        # Wait for user message
        expect(page.locator(".user-message")).to_contain_text(test_message)

        # Wait for bot response (with timeout)
        page.wait_for_selector(".bot-message", timeout=10000)
        bot_response = page.locator(".bot-message").first
        expect(bot_response).to_be_visible()

    def test_course_specific_query(self, page: Page, base_url: str):
        """Test a course-specific query to trigger RAG functionality."""
        page.goto(base_url)

        # Send a course-specific question
        test_message = "Tell me about the course content"
        page.fill("#user-input", test_message)
        page.click("#send-btn")

        # Wait for user message
        expect(page.locator(".user-message")).to_contain_text(test_message)

        # Wait for bot response
        page.wait_for_selector(".bot-message", timeout=15000)
        bot_response = page.locator(".bot-message").first
        expect(bot_response).to_be_visible()

    def test_empty_message_handling(self, page: Page, base_url: str):
        """Test that empty messages are handled properly."""
        page.goto(base_url)

        # Try to send empty message
        page.click("#send-btn")

        # Should not create any message elements
        expect(page.locator(".user-message")).to_have_count(0)

    def test_multiple_messages(self, page: Page, base_url: str):
        """Test sending multiple messages in sequence."""
        page.goto(base_url)

        messages = ["First test message", "Second test message", "Third test message"]

        for i, message in enumerate(messages, 1):
            page.fill("#user-input", message)
            page.click("#send-btn")

            # Wait for user message to appear
            expect(page.locator(".user-message")).to_have_count(i)

            # Clear input for next message
            page.fill("#user-input", "")

    def test_input_clearing(self, page: Page, base_url: str):
        """Test that input field is cleared after sending a message."""
        page.goto(base_url)

        test_message = "Test input clearing"
        page.fill("#user-input", test_message)
        page.click("#send-btn")

        # Input should be cleared after sending
        expect(page.locator("#user-input")).to_have_value("")

    def test_keyboard_shortcut(self, page: Page, base_url: str):
        """Test sending message with Enter key."""
        page.goto(base_url)

        test_message = "Test keyboard shortcut"
        page.fill("#user-input", test_message)
        page.press("#user-input", "Enter")

        # Message should be sent
        expect(page.locator(".user-message")).to_contain_text(test_message)

    def test_api_error_handling(self, page: Page, base_url: str):
        """Test handling of API errors."""
        page.goto(base_url)

        # This test would need the backend to be stopped to simulate an error
        # For now, we'll just verify the UI doesn't crash with a normal message
        test_message = "Test error handling"
        page.fill("#user-input", test_message)
        page.click("#send-btn")

        # Should still show user message even if backend fails
        expect(page.locator(".user-message")).to_contain_text(test_message)
