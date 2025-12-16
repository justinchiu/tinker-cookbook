#!/usr/bin/env python3
"""
Mock conversation fixtures for taubench testing.

These fixtures provide pre-defined conversation sequences from real GPT-4.1
user simulator interactions to allow testing without API calls.
"""

from dataclasses import dataclass


@dataclass
class MockUserResponse:
    """A mocked user simulator response."""
    content: str
    terminated: bool = False
    truncated: bool = False
    reward: float = 0.0


@dataclass
class MockToolResponse:
    """A mocked tool execution response."""
    content: str
    terminated: bool = False
    truncated: bool = False
    reward: float = 0.0


# =============================================================================
# Simple Retail Return Scenario
# =============================================================================

# This is a simplified version of a real conversation from:
# data/longest_success_retail.json
# Task: Customer wants to return some items

RETAIL_RETURN_USER_RESPONSES = [
    # Initial greeting from user
    MockUserResponse(
        content="Hey there! So I need to return some stuff from a couple different orders - I got some bookshelves and jigsaw puzzles that I want to send back. Can you help me with that?",
    ),
    # User provides email
    MockUserResponse(
        content="Oh sure! My email is lucas.brown9344@example.com",
    ),
    # User asks to look up orders
    MockUserResponse(
        content="Could you look them up for me? I don't have the order IDs handy right now.",
    ),
    # User confirms items to return
    MockUserResponse(
        content="Yes, I want to return both bookshelves and the jigsaw puzzle from order W6239298. All of those need to go back!",
    ),
    # User chooses refund method
    MockUserResponse(
        content="Just put it back on my Mastercard, that's easiest!",
    ),
    # User confirms first return
    MockUserResponse(
        content="Yes, I confirm!",
    ),
    # User confirms second return
    MockUserResponse(
        content="Yes, confirm that one too!",
    ),
    # Final acknowledgment - task complete
    MockUserResponse(
        content="That's all, thanks!",
        terminated=True,
        reward=1.0,
    ),
]

RETAIL_RETURN_TOOL_RESPONSES = {
    # find_user_id_by_email response
    "find_user_id_by_email": MockToolResponse(
        content="lucas_brown_6720",
    ),
    # get_user_details response
    "get_user_details": MockToolResponse(
        content='{"user_id": "lucas_brown_6720", "name": {"first_name": "Lucas", "last_name": "Brown"}, "address": {"address1": "921 Park Avenue", "address2": "Suite 892", "city": "Chicago", "country": "USA", "state": "IL", "zip": "60612"}, "email": "lucas.brown9344@example.com", "payment_methods": {"credit_card_2112420": {"source": "credit_card", "id": "credit_card_2112420", "brand": "mastercard", "last_four": "1276"}}, "orders": ["#W6239298", "#W8660475", "#W1154986", "#W9218746", "#W4860251"]}',
    ),
    # get_order_details responses (multiple orders)
    "get_order_details:#W6239298": MockToolResponse(
        content='{"order_id": "#W6239298", "user_id": "lucas_brown_6720", "address": {"address1": "921 Park Avenue", "address2": "Suite 892", "city": "Chicago", "country": "USA", "state": "IL", "zip": "60612"}, "items": [{"name": "Bookshelf", "product_id": "8600330539", "item_id": "4900661478", "price": 463.04, "options": {"material": "glass", "color": "black", "height": "5 ft"}}, {"name": "E-Reader", "product_id": "3801771308", "item_id": "9494281769", "price": 252.06, "options": {"screen size": "8-inch", "connectivity": "Wi-Fi", "storage": "8GB"}}, {"name": "Jigsaw Puzzle", "product_id": "1808611083", "item_id": "3614853563", "price": 46.99, "options": {"pieces": "2000", "theme": "art", "difficulty level": "intermediate"}}, {"name": "Water Bottle", "product_id": "8310926033", "item_id": "2366567022", "price": 54.04, "options": {"capacity": "1000ml", "material": "stainless steel", "color": "blue"}}], "status": "delivered", "fulfillments": [{"tracking_id": ["682308736931"], "item_ids": ["4900661478", "9494281769", "3614853563", "2366567022"]}], "payment_history": [{"transaction_type": "payment", "amount": 816.13, "payment_method_id": "credit_card_2112420"}]}',
    ),
    "get_order_details:#W8660475": MockToolResponse(
        content='{"order_id": "#W8660475", "user_id": "lucas_brown_6720", "address": {"address1": "921 Park Avenue", "address2": "Suite 892", "city": "Chicago", "country": "USA", "state": "IL", "zip": "60612"}, "items": [{"name": "Office Chair", "product_id": "4794339885", "item_id": "8323284863", "price": 511.24, "options": {"material": "fabric", "color": "blue", "armrest": "adjustable", "backrest height": "standard"}}, {"name": "Bookshelf", "product_id": "8600330539", "item_id": "8479046075", "price": 451.01, "options": {"material": "wood", "color": "white", "height": "5 ft"}}], "status": "delivered", "fulfillments": [{"tracking_id": ["866377615705"], "item_ids": ["8323284863", "8479046075"]}], "payment_history": [{"transaction_type": "payment", "amount": 962.25, "payment_method_id": "credit_card_2112420"}]}',
    ),
    # return_delivered_order_items responses
    "return_delivered_order_items:#W6239298": MockToolResponse(
        content='{"order_id": "#W6239298", "user_id": "lucas_brown_6720", "status": "return requested", "return_items": ["3614853563", "4900661478"], "return_payment_method_id": "credit_card_2112420"}',
    ),
    "return_delivered_order_items:#W8660475": MockToolResponse(
        content='{"order_id": "#W8660475", "user_id": "lucas_brown_6720", "status": "return requested", "return_items": ["8479046075"], "return_payment_method_id": "credit_card_2112420"}',
    ),
}


# =============================================================================
# Simple Lookup Scenario (for quick tests)
# =============================================================================

RETAIL_SIMPLE_LOOKUP_USER_RESPONSES = [
    # Initial user request
    MockUserResponse(
        content="Hi! I'd like to check on my order status please.",
    ),
    # User provides name and zip
    MockUserResponse(
        content="My name is John Smith and my zip code is 12345.",
    ),
    # User satisfied with response
    MockUserResponse(
        content="Great, thanks for the info!",
        terminated=True,
        reward=1.0,
    ),
]

RETAIL_SIMPLE_LOOKUP_TOOL_RESPONSES = {
    "find_user_id_by_name_zip": MockToolResponse(
        content="john_smith_1234",
    ),
    "get_user_details": MockToolResponse(
        content='{"user_id": "john_smith_1234", "name": {"first_name": "John", "last_name": "Smith"}, "orders": ["#W1234567"]}',
    ),
    "get_order_details:#W1234567": MockToolResponse(
        content='{"order_id": "#W1234567", "user_id": "john_smith_1234", "status": "pending", "items": [{"name": "Widget", "price": 29.99}]}',
    ),
}


# =============================================================================
# Conversation with ask_sonnet (for testing ask_sonnet modes)
# =============================================================================

RETAIL_WITH_ASK_SONNET_USER_RESPONSES = [
    # Initial complex request
    MockUserResponse(
        content="I need to cancel my pending order and also return a delivered order, plus change my address for future orders.",
    ),
    # Confirmation of first action
    MockUserResponse(
        content="Yes, please cancel the pending one. The reason is I ordered by mistake.",
    ),
    # Final acknowledgment
    MockUserResponse(
        content="Perfect, that's all I needed. Thanks!",
        terminated=True,
        reward=1.0,
    ),
]

# Mock responses from Sonnet for ask_sonnet calls
SONNET_MOCK_RESPONSES = {
    # Direct injection mode: Sonnet returns a tool call
    "direct_injection_tool_call": '<tool_call>\n{"name": "find_user_id_by_email", "arguments": {"email": "test@example.com"}}\n</tool_call>',

    # Direct injection mode: Sonnet returns a text response
    "direct_injection_text": "Hello! I'd be happy to help you today. Could you please provide your email address so I can look up your account?",

    # Conditioning mode: Sonnet provides advice
    "conditioning_advice": "I recommend first authenticating the user, then handling their requests one at a time. Start by asking for their email or name and zip code to look up their account.",
}


# =============================================================================
# Utility class for managing mock conversation state
# =============================================================================

class MockConversation:
    """
    Helper class to track conversation state and return appropriate mock responses.

    Usage:
        mock_conv = MockConversation(
            user_responses=RETAIL_RETURN_USER_RESPONSES,
            tool_responses=RETAIL_RETURN_TOOL_RESPONSES,
        )

        # Get next user response
        response = mock_conv.next_user_response()

        # Get tool response by name
        tool_result = mock_conv.get_tool_response("find_user_id_by_email")
    """

    def __init__(
        self,
        user_responses: list[MockUserResponse],
        tool_responses: dict[str, MockToolResponse] | None = None,
    ):
        self.user_responses = user_responses
        self.tool_responses = tool_responses or {}
        self._user_index = 0

    def next_user_response(self) -> MockUserResponse:
        """Get the next user response in sequence."""
        if self._user_index >= len(self.user_responses):
            # Return a terminal response if we've exhausted the list
            return MockUserResponse(
                content="I think that's everything, thanks!",
                terminated=True,
                reward=0.0,
            )

        response = self.user_responses[self._user_index]
        self._user_index += 1
        return response

    def get_tool_response(self, tool_name: str, order_id: str | None = None) -> MockToolResponse:
        """
        Get a tool response by name, optionally with order_id disambiguation.

        Args:
            tool_name: Name of the tool
            order_id: Optional order ID for disambiguating multiple responses

        Returns:
            MockToolResponse for the tool
        """
        # Try specific key with order_id first
        if order_id:
            key = f"{tool_name}:{order_id}"
            if key in self.tool_responses:
                return self.tool_responses[key]

        # Fall back to tool name only
        if tool_name in self.tool_responses:
            return self.tool_responses[tool_name]

        # Default response for unknown tools
        return MockToolResponse(
            content='{"error": "Unknown tool or mock not configured"}',
        )

    def reset(self):
        """Reset conversation state."""
        self._user_index = 0


# =============================================================================
# Pre-configured mock conversations
# =============================================================================

def get_retail_return_conversation() -> MockConversation:
    """Get a pre-configured retail return conversation."""
    return MockConversation(
        user_responses=RETAIL_RETURN_USER_RESPONSES,
        tool_responses=RETAIL_RETURN_TOOL_RESPONSES,
    )


def get_retail_simple_lookup_conversation() -> MockConversation:
    """Get a pre-configured simple lookup conversation."""
    return MockConversation(
        user_responses=RETAIL_SIMPLE_LOOKUP_USER_RESPONSES,
        tool_responses=RETAIL_SIMPLE_LOOKUP_TOOL_RESPONSES,
    )


def get_ask_sonnet_conversation() -> MockConversation:
    """Get a pre-configured conversation for ask_sonnet testing."""
    return MockConversation(
        user_responses=RETAIL_WITH_ASK_SONNET_USER_RESPONSES,
        tool_responses=RETAIL_SIMPLE_LOOKUP_TOOL_RESPONSES,
    )
