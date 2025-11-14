"""
Routing Examples for Query Classification - ENHANCED WITH API ENDPOINT SUPPORT
========================================

This file contains curated examples for training the query router to correctly
classify user queries into the appropriate agent.

AGENTS SUPPORTED:
- nl2sql: Complex database queries, aggregations, analytics
- api_endpoint: Sales order queries, shipping, customers, crews
- personal_assistant: Task/reminder management
- general_chat: Conversations, greetings
- document_rag: Document searches

Guidelines for adding examples:
- Keep examples balanced across all agents (same number per agent)
- Use realistic, domain-specific language
- Include variations and edge cases
- Add confidence scores (0.8-1.0 for clear examples, 0.6-0.7 for ambiguous ones)
"""

from typing import Dict, List, Tuple
import random

# =============================================================================
# API ENDPOINT EXAMPLES - REST API queries for sales orders (NEW!)
# =============================================================================

API_ENDPOINT_EXAMPLES = [
    # Sales order queries (API handles these)
    ("Show me all sales orders shipped today", 0.95),
    ("What shipped out today?", 0.95),
    ("Find orders for customer Amazon", 0.92),
    ("Show sales orders for KCC", 0.93),
    ("What orders are shipping to FNS?", 0.91),
    ("List orders for PO number 12345", 0.94),
    ("Show me orders assigned to crew T1", 0.92),
    ("What crew is working on this order?", 0.90),
    ("Show orders with missing items", 0.89),
    ("Find sales orders that haven't shipped yet", 0.88),
    ("What orders are in production?", 0.87),
    ("Show unshipped orders", 0.88),
    ("List orders scheduled to ship today", 0.91),
    ("Show orders for customer ABC Company", 0.93),
    ("What's the status of sales order 123?", 0.92),
    ("Find orders shipping via UPS", 0.89),
    ("Show orders picked up by OLD DOMINION", 0.88),
    ("List orders with tracking numbers", 0.87),
    ("Show orders completed today", 0.90),
    ("What orders are ready to ship?", 0.89),
    
    # Status queries (API better for real-time status)
    ("What's the status of this order?", 0.91),
    ("Is order 12345 shipped?", 0.93),
    ("Check order status for PO 67890", 0.92),
    ("Has this order been processed?", 0.90),
    ("What stage is this order at?", 0.88),
    ("Show order progress", 0.87),
    ("Is this order complete?", 0.89),
    ("Check if order shipped", 0.91),
    
    # Specific field queries (API is faster)
    ("Show tracking number for order 12345", 0.91),
    ("What's the ship date for PO 67890?", 0.90),
    ("Show carrier for this order", 0.89),
    ("What crew is assigned to order 12345?", 0.91),
    ("Show customer name for PO 67890", 0.90),
    ("Get order details for 12345", 0.92),
    ("Show shipping address for this order", 0.88),
    ("What's the delivery date?", 0.87),
    
    # Time-sensitive queries (API preferred for freshness)
    ("Show orders shipping out right now", 0.93),
    ("What's being loaded on the truck?", 0.89),
    ("Show current production orders", 0.90),
    ("What orders are active?", 0.88),
    ("List today's shipments", 0.92),
    ("Show ongoing orders", 0.87),
    ("What's happening now in production?", 0.86),
    
    # Customer-specific queries (API handles well)
    ("Show all Amazon orders", 0.91),
    ("List orders for Walmart", 0.91),
    ("Find orders from Target", 0.90),
    ("Show KCC sales orders", 0.92),
    ("What orders do we have for FNS?", 0.91),
    ("Display customer orders", 0.88),
    ("Show orders by customer name", 0.89),
    
    # Crew assignment queries (API is authoritative)
    ("What is T1 working on?", 0.93),
    ("Show T2 crew assignments", 0.92),
    ("What orders are assigned to T3?", 0.93),
    ("List crew workload", 0.88),
    ("Show unassigned orders", 0.87),
    ("Which crew has order 12345?", 0.91),
    ("Show crew status", 0.86),
    ("What's each crew working on?", 0.89),
]

# =============================================================================
# NL2SQL EXAMPLES - Data retrieval and database queries
# =============================================================================

NL2SQL_EXAMPLES = [
    # Aggregation queries (SQL required)
    ("How many items are there?", 0.95),
    ("How many boxes did we ship in March?", 0.94),
    ("Count total orders by customer", 0.93),
    ("Show sum of boxes shipped", 0.92),
    ("Average boxes per order", 0.91),
    ("Total containers received this year", 0.93),
    
    # Inventory and items (SQL better)
    ("Do we have any sto inbound from FNS today?", 0.92),
    ("Show all current orders that are in warehouse", 0.90),
    ("what is item number of item 584512091?", 0.93),
    ("is there any items without any upc code?", 0.87),
    ("what is master box quantity of item 584512091?", 0.94),
    ("do we have 584512091 in stock?", 0.95),
    ("where are the empty racks?", 0.88),
    ("what is in SETLOC?", 0.90),
    ("what items are assigned?", 0.89),
    ("is 584512091 assigned?", 0.94),
    
    # Container and pallet queries (SQL domain)
    ("How many containers did we unload so far?", 0.92),
    ("How many containers did we unload in February, 2025?", 0.91),
    ("where was pallet 24110800033 from?", 0.87),
    ("what items will expire soon?", 0.86),
    ("what was in container number BEAU6277469?", 0.92),
    ("what was the container number that we unloaded yesterday?", 0.93),
    ("how many pallets of 584512679 did we receive on 2024-11-11?", 0.91),
    ("list all containers that we have received so far", 0.91),
    
    # Historical analysis (SQL required)
    ("show total number of boxes that we have received, total number of boxes that we have shipped out, difference between two and total number of boxes that warehouse finished, in February 2025", 0.85),
    ("how many boxes of 584510753 did we receive on 2024-11-11?", 0.92),
    ("what did we receive on 2025-2-20?", 0.93),
    ("List what we have received from FNS in January 2025", 0.90),
    ("how many boxes were sold each month of the year?", 0.90),
    ("how many boxes were finished each month of the year in warehouse?", 0.89),
    
    # Complex joins and breakdowns (SQL domain)
    ("Can you break it down by vendors?", 0.88),
    ("Show warehouse activity report by month", 0.89),
    ("List top 5 most finished items", 0.87),
    ("What are the items and quantity of all POs finished in warehouse in December?", 0.86),
    ("how many containers did we unload each month", 0.89),
    
    # Expired/returned items (SQL better)
    ("do we have any expired items?", 0.94),
    ("was any item returned in January 2025?", 0.89),
    ("was any item returned?", 0.90),
    ("what company returned?", 0.87),
    
    # Database schema queries (SQL only)
    ("How many tables are in the database?", 0.88),
    ("Show database structure", 0.85),
    ("List all table names", 0.86),
]

# =============================================================================
# PERSONAL ASSISTANT EXAMPLES - Task and reminder management
# =============================================================================

PERSONAL_ASSISTANT_EXAMPLES = [
    # Conversational meeting and appointment mentions
    ("I have a meeting with Thomas at 3 pm today", 0.95),
    ("meeting with the supplier tomorrow at 10", 0.95),
    ("got an appointment with the warehouse manager this afternoon", 0.90),
    ("seeing the logistics team at 2 pm", 0.90),
    ("lunch meeting with the client today", 0.88),
    ("conference call scheduled for 4 pm", 0.90),
    ("team standup at 9 am tomorrow", 0.88),
    ("quarterly review meeting next week", 0.85),
    ("budget meeting on Friday morning", 0.85),
    ("safety training session this Thursday", 0.83),
    
    # Natural task expressions
    ("I need to call the supplier about that order", 0.95),
    ("gotta remember to check the inventory levels", 0.92),
    ("should follow up with the customer tomorrow", 0.90),
    ("need to review those shipping costs", 0.88),
    ("have to update the safety protocols", 0.85),
    ("must contact the warehouse about delivery", 0.87),
    ("should schedule equipment maintenance soon", 0.85),
    ("need to prepare the monthly report", 0.83),
    ("have to order new equipment", 0.80),
    ("should call about the warranty issue", 0.78),
    
    # Reminder-style conversations
    ("remind me to email the logistics team", 0.95),
    ("don't let me forget about the pickup schedule", 0.92),
    ("make sure I remember the compliance audit", 0.90),
    ("help me remember to submit those reports", 0.88),
    ("I should set a reminder for tomorrow", 0.85),
    
    # Task management
    ("add task to follow up with client", 0.93),
    ("create reminder for inventory check", 0.91),
    ("schedule task for next week", 0.89),
    ("mark task as complete", 0.87),
    ("show my tasks for today", 0.90),
]

# =============================================================================
# GENERAL CHAT EXAMPLES - Casual conversations
# =============================================================================

GENERAL_CHAT_EXAMPLES = [
    # Greetings and small talk
    ("Hello", 0.98),
    ("Hi there", 0.97),
    ("Good morning", 0.96),
    ("How are you?", 0.95),
    ("What's up?", 0.94),
    ("Hey", 0.96),
    ("Good afternoon", 0.95),
    
    # Thanks and acknowledgments
    ("Thank you", 0.96),
    ("Thanks a lot", 0.95),
    ("I appreciate it", 0.93),
    ("That's helpful", 0.92),
    ("Got it, thanks", 0.90),
    
    # Capability questions
    ("What can you do?", 0.95),
    ("How can you help me?", 0.94),
    ("What are your capabilities?", 0.93),
    ("Tell me about yourself", 0.91),
    ("What features do you have?", 0.90),
    
    # General questions
    ("How does this work?", 0.88),
    ("Can you explain that?", 0.87),
    ("I need help", 0.85),
    ("I have a question", 0.84),
    
    # Casual conversation
    ("That's interesting", 0.90),
    ("I see", 0.88),
    ("Okay", 0.85),
    ("Alright", 0.84),
]

# =============================================================================
# DOCUMENT RAG EXAMPLES - Document searches
# =============================================================================

DOCUMENT_RAG_EXAMPLES = [
    ("Search for safety protocol document", 0.95),
    ("Find the SOP for receiving", 0.93),
    ("Show me shipping guidelines", 0.92),
    ("Look up quality control procedures", 0.91),
    ("Find documents about warehouse layout", 0.90),
    ("Search employee handbook", 0.89),
    ("Show training materials", 0.88),
    ("Find policy documents", 0.87),
]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_examples_by_agent(agent_name: str) -> List[Tuple[str, float]]:
    """
    Get examples for a specific agent
    
    Args:
        agent_name: 'nl2sql', 'api_endpoint', 'personal_assistant', 'general_chat', or 'document_rag'
    
    Returns:
        List of tuples: (query_text, confidence_score)
    """
    agent_mapping = {
        "nl2sql": NL2SQL_EXAMPLES,
        "api_endpoint": API_ENDPOINT_EXAMPLES,
        "personal_assistant": PERSONAL_ASSISTANT_EXAMPLES,
        "general_chat": GENERAL_CHAT_EXAMPLES,
        "document_rag": DOCUMENT_RAG_EXAMPLES 
    }
    
    return agent_mapping.get(agent_name, [])

def get_all_examples() -> List[Tuple[str, str, float]]:
    """
    Get all routing examples from all agents
    
    Returns:
        List of tuples: (query_text, agent_name, confidence)
    """
    examples = []
    
    for agent in ["nl2sql", "api_endpoint", "personal_assistant", "general_chat", "document_rag"]:
        agent_examples = get_examples_by_agent(agent)
        for query, confidence in agent_examples:
            examples.append((query, agent, confidence))
    
    return examples

def get_balanced_examples(examples_per_agent: int = 20) -> List[Tuple[str, str, float]]:
    """
    Get a balanced set of examples with equal representation per agent
    
    Args:
        examples_per_agent: Number of examples to select per agent
    
    Returns:
        List of balanced examples
    """
    balanced = []
    
    for agent in ["nl2sql", "api_endpoint", "personal_assistant", "general_chat", "document_rag"]:
        agent_examples = get_examples_by_agent(agent)
        
        # Randomly sample or take first N examples
        if len(agent_examples) > examples_per_agent:
            # Prioritize high-confidence examples
            sorted_examples = sorted(agent_examples, key=lambda x: x[1], reverse=True)
            selected = sorted_examples[:examples_per_agent]
        else:
            selected = agent_examples
        
        # Add to balanced set
        for query, confidence in selected:
            balanced.append((query, agent, confidence))
    
    return balanced

def get_example_stats() -> Dict[str, int]:
    """
    Get statistics about the routing examples
    
    Returns:
        Dictionary with counts per agent and total
    """
    stats = {
        "nl2sql": len(NL2SQL_EXAMPLES),
        "api_endpoint": len(API_ENDPOINT_EXAMPLES),
        "personal_assistant": len(PERSONAL_ASSISTANT_EXAMPLES),
        "general_chat": len(GENERAL_CHAT_EXAMPLES),
        "document_rag": len(DOCUMENT_RAG_EXAMPLES),
    }
    stats["total"] = sum(stats.values())
    
    return stats

def validate_examples() -> Dict[str, List[str]]:
    """
    Validate examples for potential issues
    
    Returns:
        Dictionary of validation issues found
    """
    issues = {
        "duplicates": [],
        "low_confidence": [],
        "ambiguous": []
    }
    
    all_examples = get_all_examples()
    queries_seen = set()
    
    for query, agent, confidence in all_examples:
        # Check for duplicates
        if query in queries_seen:
            issues["duplicates"].append(query)
        queries_seen.add(query)
        
        # Check for low confidence
        if confidence < 0.7:
            issues["low_confidence"].append(f"{query} ({confidence})")
        
        # Check for potentially ambiguous queries (very simple heuristic)
        if len(query.split()) <= 2:
            issues["ambiguous"].append(query)
    
    return issues

def add_example(query: str, agent: str, confidence: float = 0.9) -> bool:
    """
    Add a new example to the appropriate list (for runtime additions)
    Note: This modifies the module-level lists
    
    Args:
        query: The query text
        agent: Target agent name
        confidence: Confidence score (0.0-1.0)
    
    Returns:
        True if added successfully, False otherwise
    """
    if agent == "nl2sql":
        NL2SQL_EXAMPLES.append((query, confidence))
        return True
    elif agent == "api_endpoint":
        API_ENDPOINT_EXAMPLES.append((query, confidence))
        return True
    elif agent == "personal_assistant":
        PERSONAL_ASSISTANT_EXAMPLES.append((query, confidence))
        return True
    elif agent == "general_chat":
        GENERAL_CHAT_EXAMPLES.append((query, confidence))
        return True
    elif agent == "document_rag":
        DOCUMENT_RAG_EXAMPLES.append((query, confidence))
        return True
    else:
        return False

# =============================================================================
# MAIN EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    """Test and validate the routing examples"""
    
    print("🧪 ROUTING EXAMPLES VALIDATION")
    print("=" * 60)
    
    # Show statistics
    stats = get_example_stats()
    print(f"\n📊 Example Statistics:")
    for agent, count in stats.items():
        if agent != "total":
            print(f"  {agent:20s}: {count:3d} examples")
    print(f"  {'TOTAL':20s}: {stats['total']:3d} examples")
    
    # Check balance
    agent_counts = [
        stats["nl2sql"], 
        stats["api_endpoint"],
        stats["personal_assistant"], 
        stats["general_chat"],
        stats["document_rag"]
    ]
    balance_ratio = max(agent_counts) / min(agent_counts) if min(agent_counts) > 0 else float('inf')
    
    print(f"\n⚖️  Balance Ratio: {balance_ratio:.2f}")
    if balance_ratio <= 1.5:
        print("✅ Examples are well balanced")
    elif balance_ratio <= 2.5:
        print(f"⚠️  Examples moderately imbalanced")
    else:
        print(f"🚨 Examples heavily imbalanced")
    
    # Validate examples
    print(f"\n🔍 Validation Results:")
    issues = validate_examples()
    
    for issue_type, issue_list in issues.items():
        if issue_list:
            print(f"  {issue_type}: {len(issue_list)} found")
            if len(issue_list) <= 3:
                for item in issue_list:
                    print(f"    - {item}")
            else:
                print(f"    - {issue_list[0]}")
                print(f"    - ... and {len(issue_list)-1} more")
        else:
            print(f"  {issue_type}: None found ✅")
    
    # Show sample balanced set
    print(f"\n📝 Sample Balanced Set (5 per agent):")
    balanced = get_balanced_examples(5)
    for query, agent, confidence in balanced:
        print(f"  {agent:20s} | {confidence:.2f} | '{query}'")
    
    print(f"\n✅ Routing examples validation complete!")