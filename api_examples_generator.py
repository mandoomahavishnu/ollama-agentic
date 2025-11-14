"""
API Examples Generator - REVISED FOR ACTUAL API
===============================================

This file generates API few-shot examples matching the ACTUAL parameters
from salesorder.php. Parameter names match exactly what the API expects.
"""

from typing import List, Dict, Any


def generate_api_examples() -> List[Dict[str, Any]]:
    """
    Generate API examples using ACTUAL parameter names from the PHP API.
    
    Structure:
    {
        "natural_language": "user query",
        "endpoint": "salesorder|inventory|inbound",
        "api_params": {"param": "value"},
        "confidence": 0.0-1.0
    }
    """
    
    examples = []
    
    # =========================================================================
    # SALES ORDER EXAMPLES (salesorder endpoint)
    # =========================================================================
    
    # Shipping status queries
    examples.extend([
        {
            "natural_language": "Show me all sales orders shipped today",
            "endpoint": "outbound",
            "api_params": {
                "shipped": "not_empty",
                "shipped_date_from": "{today}",
                "shipped_date_to": "{today}",
		        
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What are we shipping out today?",
            "endpoint": "oubound",
            "api_params": {
                
                "etd_from": "{today}",
                "etd_to": "{today}",
		        
        },
        },
        {
            "natural_language": "What shipped out today?",
            "endpoint": "outbound",
            "api_params": {
                "shipped": "not_empty",
                "shipped_date_from": "{today}",
                "shipped_date_to": "{today}",
		    
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show orders shipped this week",
            "endpoint": "outbound",
            "api_params": {
                "shipped": "not_empty",
                "shipped_date_from": "{week_start}",
                "shipped_date_to": "{week_end}",
		        
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders shipped yesterday?",
            "endpoint": "outbound",
            "api_params": {
                "shipped": "not_empty",
                "shipped_date_from": "{yesterday}",
                "shipped_date_to": "{yesterday}",
		
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show me orders that haven't shipped yet today",
            "endpoint": "outbound",
            "api_params": {
                "shipped": "empty",
		    
		        "etd_from": "{today}",
		        "etd_to": "{today}"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders are waiting to ship?",
            "endpoint": "outbound",
            "api_params": {
                "shipped": "empty",
                   
		
		
	},
            "confidence": 1.0
        },
    ])
    
    # Customer queries
    examples.extend([
        {
            "natural_language": "Find orders for customer Amazon",
            "endpoint": "salesorder",
            "api_params": {
                "customer": "Amazon",
                "status" : 'empty'
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show me Walmart orders",
            "endpoint": "salesorder",
            "api_params": {
                "customer": "Walmart"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders do we have for Target?",
            "endpoint": "salesorder",
            "api_params": {
                "customer": "Target"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show Costco orders shipped today",
            "endpoint": "outbound",
            "api_params": {
                "account": "Costco",
                "shipped": "not_empty",
                "shipped_date_from": "{today}",
                "shipped_date_to": "{today}"
            },
            "confidence": 1.0
        },
    ])
    
    # PO number queries
    examples.extend([
        {
            "natural_language": "Find orders for PO number 12345",
            "endpoint": "salesorder",
            "api_params": {
                "cust_po_num": "12345"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show me purchase order 11319997",
            "endpoint": "salesorder",
            "api_params": {
                "cust_po_num": "11319997"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Look up PO 2024-001",
            "endpoint": "salesorder",
            "api_params": {
                "cust_po_num": "2024-001"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "is there anything missing for po 1234?",
            "endpoint": "salesorder_detail",
            "api_params": {
                "po_num": "1234",
                "item_status": "missing",
		"status": "not_cancelled"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "is there anything ETA items for po 1234?",
            "endpoint": "salesorder_detail",
            "api_params": {
                "po_num": "1234",
                "item_status": "ETA",
		"status": "not_cancelled"
            },
            "confidence": 1.0
        },

    ])
    
    # SO number queries
    examples.extend([
        {
            "natural_language": "Show me sales order SO12345",
            "endpoint": "salesorder",
            "api_params": {
                "so_num": "SO12345"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Look up order 12345",
            "endpoint": "salesorder",
            "api_params": {
                "so_num": "12345"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What is the customer for so 123456?",
            "endpoint": "salesorder",
            "api_params": {
                "so_num": "123456"
            },
            "confidence": 1.0
        },
    ])
    
    # Crew assignment queries
    examples.extend([
        {
            "natural_language": "What orders are assigned to crew T1",
            "endpoint": "warehouse",
            "api_params": {
                "crew": "T1",
                "routing_date_is": "not_empty",
                "summary": "empty",
		

            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show me what T2 is working on",
            "endpoint": "warehouse",
            "api_params": {
                "crew": "T2",
                "routing_date_is": "not_empty",
                "summary": "empty",
		
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What is crew T1 working on now?",
            "endpoint": "warehouse",
            "api_params": {
                "crew": "T1",
                "summary": "empty",
		
            },
            "confidence": 1.0
        },
        {
            "natural_language": "show pick ticket details of what T1 is working on now",
            "endpoint": "salesorder_detail",
            "api_params": {
                "crew": "T1",
                "summary": "empty",
		"status": "not_cancelled"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show T3 crew assignments",
            "endpoint": "warehouse",
            "api_params": {
                "crew": "T3",
		
                "summary": "empty"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders are not assigned to any crew?",
            "endpoint": "warehouse",
            "api_params": {
                "crew": "empty",
                "routing_date_is": "not_empty",
                "summary": "empty",
		
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what is each crew working on?",
            "endpoint": "salesorder",
            "api_params": {
                "crew": "not_empty",
                "summary": "empty",
		
            },
            "confidence": 1.0
        },
    ])
    
    # Item status queries
    examples.extend([
        {
            "natural_language": "Show me orders with missing items",
            "endpoint": "salesorder_detail",
            "api_params": {
                "item_status": "missing",
		        "status": "empty",
                "summary": "empty"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders have missing items?",
            "endpoint": "salesorder_detail",
            "api_params": {
                "item_status": "missing",
		        "status": "empty",
                "summary": "empty"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Find orders that need repacking",
            "endpoint": "salesorder_detail",
            "api_params": {
                "item_status": "Repacking",
		        "status": "empty",
                "summary": "empty",
                "routing_date" : "not_empty"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show orders with 'ETA items'",
            "endpoint": "salesorder_detail",
            "api_params": {
                "item_status": "ETA",
		        "status": "empty",
                "summary": "empty",
                "routing_date" : "not_empty"

            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders are short and have missing items?",
            "endpoint": "salesorder_detail",
            "api_params": {
                "item_status": "ETA+missing",
		"status": "not_cancelled",
                "summary": "empty"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "is there anything missing for po's that are currently being worked on?",
            "endpoint": "salesorder_detail",
            "api_params": {
                "item_status": "missing",
                "summary": "empty",
                "status": "not_cancelled",
                "crew": "not_empty"
            },
            "confidence": 1.0
        },
    ])
    
    # Processing status queries
    examples.extend([
        {
            "natural_language": "Show order details currently being processed",
            "endpoint": "salesorder_detail",
            "api_params": {
                "summary": "empty",
                "crew": "not_empty",
		        "status": "not_cancelled"

            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders are in progress?",
            "endpoint": "warehouse",
            "api_params": {
                "summary": "empty",
                "crew": "not_empty",
		
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show completed orders in November, 2025",
            "endpoint": "warehouse",
            "api_params": {
                "summary": "y",
		        "process_end_from" : "2025-11-01",
                "process_end_to" : "2025-11-31"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders finished today?",
            "endpoint": "warehouse",
            "api_params": {
                "process_end_from": "{today}",
                "process_end_to": "{today}",
		

            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show orders that started processing today",
            "endpoint": "warehouse",
            "api_params": {
                "process_start_from": "{today}",
                "process_start_to": "{today}",
		

            },
            "confidence": 1.0
        },
    ])
    
    # Item/product queries
    examples.extend([
        {
            "natural_language": "Find orders for product 584512091",
            "endpoint": "salesorder",
            "api_params": {
                "sap_code": "584512091",
		"status": "not_cancelled"

            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show orders for item 1234",
            "endpoint": "salesorder",
            "api_params": {
                "item_num": "1234",
		"status": "not_cancelled"

            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders have SAP code 584510753?",
            "endpoint": "salesorder",
            "api_params": {
                "sap_code": "584510753",
		"status": "not_cancelled"

            },
            "confidence": 1.0
        },
    ])
    
    # Location queries
    examples.extend([
        {
            "natural_language": "Show picking activity at 21-X-1",
            "endpoint": "picktix",
            "api_params": {
                "location": "21-X-1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What was picked from location 12-X-1?",
            "endpoint": "picktix",
            "api_params": {
                "location": "12-X-1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "where was the product 1234 picked from, for po 56789",
            "endpoint": "warehouse",
            "api_params": {
                "sap_code": "1234",
                "po_num": "56789"
            },
            "confidence": 1.0
        },
    ])
    
    
    # ETD queries
    examples.extend([
        {
            "natural_language": "Show orders with ETD this week",
            "endpoint": "outbound",
            "api_params": {
                "etd_from": "{week_start}",
                "etd_to": "{week_end}",
                

            },
            "confidence": 1.0
        },
        {
            "natural_language": "What orders are scheduled to ship next week?",
            "endpoint": "outbound",
            "api_params": {
                "etd_from": "{next_week_start}",
                "etd_to": "{next_week_end}",
		
            },
            "confidence": 1.0
        },
    ])
    
    # Carrier queries
    examples.extend([
        {
            "natural_language": "Show UPS shipments",
            "endpoint": "outbound",
            "api_params": {
                "carrier": "UPS"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "What shipped via FedEx today?",
            "endpoint": "outbound",
            "api_params": {
                "carrier": "FedEx",
                "shipped_date_from": "{today}",
                "shipped_date_to": "{today}"
            },
            "confidence": 1.0
        },
    ])
    
    # Combined complex queries
    examples.extend([
        {
            "natural_language": "Show Amazon orders shipped this week",
            "endpoint": "outbound",
            "api_params": {
                "account": "Amazon",
                "shipped": "not_empty",
                "shipped_date_from": "{week_start}",
                "shipped_date_to": "{week_end}"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "Show T1 orders with missing items",
            "endpoint": "salesorder_detail",
            "api_params": {
                "crew": "T1",
                "summary": "empty",
                "item_status": "missing",
		        "status": "not_cancelled"

            },
            "confidence": 1.0
        },
    ])
    
    # =========================================================================
    # INVENTORY EXAMPLES (inventory endpoint)
    # =========================================================================
    
    # Stock queries
    examples.extend([
        {
            "natural_language": "do we have 584512091 in stock?",
            "endpoint": "inventory",
            "api_params": {
                "item_code": "584512091",
                "units_min": "1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "How many units of product 1234 do we have in stock?",
            "endpoint": "inventory",
            "api_params": {
                "item_code": "1234",
                "units_min": "1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "where are the product 1234 in stock",
            "endpoint": "inventory",
            "api_params": {
                "item_code": "1234",
                "units_min": "1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what is the product 1234",
            "endpoint": "inventory",
            "api_params": {
                "item_code": "1234"
            },
            "confidence": 1.0
        },
    ])
    
    # Bin location queries
    examples.extend([
        {
            "natural_language": "what products are in bin 12-X-1?",
            "endpoint": "inventory",
            "api_params": {
                "bin": "12-X-1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what products are in scrap?",
            "endpoint": "inventory",
            "api_params": {
                "bin": "SCRAP"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what products are in PCNLOC?",
            "endpoint": "inventory",
            "api_params": {
                "bin": "PCNLOC"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what is in SETLOC?",
            "endpoint": "inventory",
            "api_params": {
                "bin": "SETLOC"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "where are the empty racks?",
            "endpoint": "inventory",
            "api_params": {
                "units_max": "0"
            },
            "confidence": 0.9
        },
    ])
    
    # Assignment queries
    examples.extend([
        {
            "natural_language": "what items are assigned?",
            "endpoint": "inventory",
            "api_params": {
                "assigned_status": "assigned"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "is 584512091 assigned?",
            "endpoint": "inventory",
            "api_params": {
                "item_code": "584512091",
                "assigned_status": "assigned"
            },
            "confidence": 1.0
        },
    ])
    
    # Expiration queries
    examples.extend([
        {
            "natural_language": "what items will expire soon?",
            "endpoint": "inventory",
            "api_params": {
                "exp_date_from": "{today}",
                "exp_date_to": "{next_week_end}",
                "units_min": "1"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "show me expiring products",
            "endpoint": "inventory",
            "api_params": {
                "exp_date_from": "{today}",
                "exp_date_to": "{next_month_end}",
                "units_min": "1"
            },
            "confidence": 0.9
        },
    ])
    
    # =========================================================================
    # INBOUND EXAMPLES (inbound endpoint)
    # =========================================================================
    
    # Receiving queries
    examples.extend([
        {
            "natural_language": "what did we receive today?",
            "endpoint": "inbound",
            "api_params": {
                "date_from": "{today}",
                "date_to": "{today}"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what did we receive from FNS today?",
            "endpoint": "inbound",
            "api_params": {
                "vendor": "FNS",
                "date_from": "{today}",
                "date_to": "{today}"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "did we receive any returns today?",
            "endpoint": "inbound",
            "api_params": {
                "vendor": "RETURN",
                "date_from": "{today}",
                "date_to": "{today}"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "did we receive any containers today?",
            "endpoint": "inbound",
            "api_params": {
                "vendor": "CONTAINER",
                "date_from": "{today}",
                "date_to": "{today}"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what did we receive on 2025-2-20?",
            "endpoint": "inbound",
            "api_params": {
                "date_from": "2025-02-20",
                "date_to": "2025-02-20"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "what products did we receive from FNS on 2025-2-20?",
            "endpoint": "inbound",
            "api_params": {
                "vendor" : "FNS",
                "date_from": "2025-02-20",
                "date_to": "2025-02-20"
            },
            "confidence": 1.0
        },
    ])
    
    # Container queries
    examples.extend([
        {
            "natural_language": "How many containers did we unload so far?",
            "endpoint": "inbound",
            "api_params": {
                "vendor": "CONTAINER"
            },
            "confidence": 0.9
        },
        {
            "natural_language": "how many containers did we receive in July 2025?",
            "endpoint": "inbound",
            "api_params": {
                "vendor": "CONTAINER",
                "date_from": "2025-07-01",
                "date_to": "2025-07-31"
            },
            "confidence": 1.0
        },
        {
            "natural_language": "did we receive any containers on 2025-2-20?",
            "endpoint": "inbound",
            "api_params": {
                "vendor": "CONTAINER",
                "date_from": "2025-02-20",
                "date_to": "2025-02-20"
            },
            "confidence": 1.0
        },
    ])
    
    # Pallet queries
    examples.extend([
        {
            "natural_language": "where was pallet 24110800033 from?",
            "endpoint": "inbound",
            "api_params": {
                "pallet_num": "24110800033"
            },
            "confidence": 1.0
        },
    ])
    
    # Item receiving queries
    examples.extend([
        {
            "natural_language": "how many boxes of 584510753 did we receive on 2024-11-11?",
            "endpoint": "inbound",
            "api_params": {
                "item_code": "584510753",
                "date_from": "2024-11-11",
                "date_to": "2024-11-11"
            },
            "confidence": 1.0
        },
    ])
    
    return examples


def get_examples_by_endpoint(endpoint_name: str) -> List[Dict[str, Any]]:
    """Get all examples for a specific endpoint"""
    all_examples = generate_api_examples()
    return [ex for ex in all_examples if ex["endpoint"] == endpoint_name]


def get_example_count_by_endpoint() -> Dict[str, int]:
    """Get count of examples per endpoint"""
    all_examples = generate_api_examples()
    counts = {}
    for ex in all_examples:
        endpoint = ex["endpoint"]
        counts[endpoint] = counts.get(endpoint, 0) + 1
    return counts


if __name__ == "__main__":
    # Test the generator
    examples = generate_api_examples()
    print(f"Generated {len(examples)} API examples")
    print("\nExamples by endpoint:")
    for endpoint, count in get_example_count_by_endpoint().items():
        print(f"  {endpoint}: {count} examples")
    
    # Show a few examples
    print("\nSample examples:")
    for ex in examples[:3]:
        print(f"\nQuery: {ex['natural_language']}")
        print(f"Endpoint: {ex['endpoint']}")
        print(f"Params: {ex['api_params']}")
