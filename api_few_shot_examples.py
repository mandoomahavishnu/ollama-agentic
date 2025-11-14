"""
API Few-Shot Examples - Parameter Mappings
===========================================

This file contains few-shot examples for the API Endpoint Agent.
These examples teach the agent how to map natural language to API parameters.

Each example shows:
- natural_language: The user's query
- api_params: The API parameters to use
- explanation: Why these parameters were chosen

These are stored as vectors and retrieved via semantic search when needed.

Note: These examples parallel your SQL few-shot examples but map to API calls instead.
"""

api_few_shot_examples = [
    # ==========================================================================
    # SHIPPING STATUS QUERIES
    # ==========================================================================
    {
        "natural_language": "Show me all sales orders shipped today",
        "api_params": {
            "shipped": "not_empty",
            "shipped_date_from": "{today}",
            "shipped_date_to": "{today}"
        },
        "explanation": "User wants orders that have been shipped today. Use shipped='not_empty' to filter for shipped orders, and set both shipped_date_from and shipped_date_to to today.",
        "confidence": 1.0
    },
    {
        "natural_language": "What shipped out today?",
        "api_params": {
            "shipped": "not_empty",
            "shipped_date_from": "{today}",
            "shipped_date_to": "{today}"
        },
        "explanation": "Same as 'shipped today' - filter for shipped orders on today's date.",
        "confidence": 1.0
    },
    {
        "natural_language": "Find orders for customer Amazon",
        "api_params": {
            "customer": "Amazon"
        },
        "explanation": "Filter by customer name using partial match (LIKE search).",
        "confidence": 1.0
    },
    {
        "natural_language": "Find orders for PO number 12345",
        "api_params": {
            "po_num": "12345"
        },
        "explanation": "Use po_num parameter for purchase order lookups (partial match).",
        "confidence": 1.0
    },
    {
        "natural_language": "What orders are assigned to crew T1",
        "api_params": {
            "crew": "T1"
        },
        "explanation": "Filter by crew assignment using exact crew code.",
        "confidence": 1.0
    },
    {
        "natural_language": "Show me orders with missing items",
        "api_params": {
            "item_status": "missing"
        },
        "explanation": "Use item_status='missing' to find orders with missing items.",
        "confidence": 1.0
    },
]

