"""
API Endpoints Configuration
============================

This file defines all available API endpoints with their:
- Description (for endpoint detection)
- Parameters (for parameter mapping)
- Keywords (for routing)

Each endpoint and parameter has rich descriptions that are embedded
for semantic search during query processing.
"""

# =============================================================================
# ENDPOINT DEFINITIONS
# =============================================================================

API_ENDPOINTS = {
    "SALESORDER DETAIL": {
        "url": "/salesorder_detail.php",
        "description": """
        combined, (warehouse, salesorder, picktix, outbound, repack), details endpoint for querying order information including:
        
        - SKUitems/products missing from pick ticket location with quantity in salesorder
        - ETA items (products/items/SKU that has less picked quantity from pick ticket than order quantity from salesorder)
        - Repack/Reapcking items (products/items/SKU that has to be repacked. Only shows the final repacked products/items/SKU in pick ticket and salesorder) 
        """,
        "keywords": [
                     
            "missing", "repacking", "eta item", "tracking"
        ],
        "parameters": {
            # === ORDER IDENTIFICATION ===
            "so_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Sales order number. Matches partial or full SO numbers.",
                "examples": ["SO12345", "12345", "SO-2024-001"]
            },
            "po_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Customer's purchase order number. Matches partial PO numbers.",
                "examples": ["PO123", "11319997", "2024-PO-456"]
            },

            "status": {
                "type": "string",
                "operator": "exact",
                "description": """Status of the sales order. 'ALL' for all sales order, "empty" for not closed(C) and not cancelled(Ca) orders, "not empty" for closed (C) or cancelled (Ca) orders, "not cancelled" for 
			                    closed(C) or not cancalled orders""",
                "examples": ["All", "Empty", "Not Empty", "Not Cancelled"]
            },

            "ship_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Start date for desired shipping date range. Not actual scheduled ship date. Format: YYYY-MM-DD",
                "examples": ["{today}", "{yesterday}", "2024-01-01"]
            },
            "ship_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "End date for desired shipping date range. Not actual scheduled ship date. Format: YYYY-MM-DD",
                "examples": ["{today}", "{week_end}", "2024-12-31"]
            },
            "customer": {
                "type": "string",
                "operator": "LIKE",
                "description": "Customer name or name of account. Matches partial customer names.",
                "examples": ["Amazon", "Walmart", "Target", "TJ Maxx", "Ulta", "Ross"]
            },
            "sap_code": {
                "type": "string",
                "operator": "LIKE",
                "description": "SKU/Product identifying number",
                "examples": ["584512562", "GP013021"]
            },           
          
            # === SHIPPING STATUS ===
            "etd_from": {
                "type": "date",
                "operator": ">=",
                "description": "Estimated Time of Departure start date range for outbound shipment.",
                "examples": ["{today}", "{next_week_start}"]
            },
            "etd_to": {
                "type": "date",
                "operator": "<=",
                "description": "Estimated Time of Departure end date range for outbound shipment.",
                "examples": ["{week_end}", "{month_end}"]
            },
            "shipped": {
                "type": "string",
                "operator": "exact",
                "description": """Whether order has been shipped. Use 'not_empty' for shipped orders or cancelled orders(C), "empty" for not shipped orders,'Y' for confirmed shipped.""",
                "examples": ["not_empty", "Y", "empty"]
            },
            "shipped_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Start date for shipped(past tense, comfirmed) date range. Format: YYYY-MM-DD",
                "examples": ["{today}", "{yesterday}", "2024-01-01"]
            },
            "shipped_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "End date for shipped(past tense, confirmed) date range. Format: YYYY-MM-DD",
                "examples": ["{today}", "{week_end}", "2024-12-31"]
            },
           "carrier": {
                "type": "string",
                "operator": "LIKE",
                "description": "outbound Shipping carrier name.",
                "examples": ["UPS", "FedEx", "USPS"]
            },
            
            # === WAREHOUSE OPERATIONS ===
            "routing_date": {
                "type": "string",
                "operator": "exact",
                "description": """due date for completion of the fulfillment of sales order, work order, purchase order. if the routing date is empty, the work order or sales order has not been
                                   passed to warehouse for fulfillment. if not_empty, the warehouse has received the sales order/ work order as a fulfillment.""",
                "examples": ["empty", "not_empty"]
            },



            "crew": {
                "type": "string",
                "operator": "exact",
                "description": "Crew or team assignment code.",
                "examples": ["T1", "T2", "T3", "T4", "T5", "T6", "LOCAL", "SONG", "CREW-A"]
            },
            "process_start_from": {
                "type": "date",
                "operator": ">=",
                "description": "Start date for fulfillment start date range. Format: YYYY-MM-DD",
                "examples": ["{today}", "{week_end}", "2024-12-31"]
            },
	        "process_start_to": {
                "type": "date",
                "operator": "<=",
                "description": "End date for fulfillment start date range. Format: YYYY-MM-DD",
                "examples": ["{today}", "{week_end}", "2024-12-31"]
            },

            "process_end_from": {
                "type": "date",
                "operator": ">=",
                "description": "Start date for fulfillment finish date range. Format: YYYY-MM-DD",
                "examples": ["{today}", "{week_end}", "2024-12-31"]
            },

            "process_end_to": {
                "type": "date",
                "operator": "<=",
                "description": "End date for fulfillment finish date range. Format: YYYY-MM-DD",
                "examples": ["{today}", "{week_end}", "2024-12-31"]
            },
            
            # === ITEM STATUS ===
            "item_status": {
                "type": "string",
                "operator": "exact",
                "description": """Status of items in the order in relation with pick ticket. Common values: missing, repacking, eta. 'missing' means there has been shortage of units or boxes from the pick ticket's
                                    picked_qty at the location given. 'repacking' means the sku is in the process of repacking of which details can be found in repack. 'eta' means, in relation with pick ticket,
                                    salesorder's order quantity is greater than pick ticket's picked quantity.""",
                "examples": ["missing", "repacking", "eta"]
            },
            
            # === DATE FILTERS ===
            "so_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Start date for order creation date range.",
                "examples": ["{today}", "{week_start}", "2024-01-01"]
            },
            "so_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "End date for order creation date range.",
                "examples": ["{today}", "{month_end}", "2024-12-31"]
            },

            
            # === LOCATION & ROUTING ===

 
        }
    },
    
    "inventory": {
        "url": "/inventory.php",
        "description": """
        Inventory endpoint for querying stock information including:
        - Product quantities and locations
        - Bin locations and warehouse positions
        - Expiration dates and lot tracking
        - Item codes, SKUs, and descriptions
        - Available, assigned, and reserved inventory
        """,
        "keywords": [
            "inventory", "stock", "bin", "location", "warehouse",
            "product", "item", "sku", "quantity", "units",
            "expiration", "expire", "lot", "available", "assigned"
        ],
        "parameters": {
            "item_code": {
                "type": "string",
                "operator": "LIKE",
                "description": "Item code, SAP code, or SKU. Matches partial item codes.",
                "examples": ["584512091", "ITEM-123", "SAP-456"]
            },
            "item_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Item number or product number.",
                "examples": ["PROD-001", "123456"]
            },
            "bin": {
                "type": "string",
                "operator": "LIKE",
                "description": "Warehouse bin or location code. Use wildcards for zone searches.",
                "examples": ["12-X-1", "SCRAP", "SETLOC", "21-A-%"]
            },
            "units_min": {
                "type": "integer",
                "operator": ">=",
                "description": "Minimum units/quantity threshold.",
                "examples": ["1", "10", "100"]
            },
            "units_max": {
                "type": "integer",
                "operator": "<=",
                "description": "Maximum units/quantity threshold.",
                "examples": ["1000", "500"]
            },
            "assigned_status": {
                "type": "string",
                "operator": "exact",
                "description": "Assignment status. Use 'assigned' for assigned inventory, 'available' for unassigned.",
                "examples": ["assigned", "available"]
            },
            "exp_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Expiration date range start.",
                "examples": ["{today}", "{next_week}"]
            },
            "exp_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "Expiration date range end.",
                "examples": ["{next_month}", "{today}"]
            },
        }
    },
    
    "inbound": {
        "url": "/inbound.php",
        "description": """
        Inbound receiving endpoint for tracking incoming shipments:
        - Container receipts and unloading
        - Vendor deliveries and returns
        - Pallet tracking and reference numbers
        - Received quantities and dates
        - Item codes and box counts
        """,
        "keywords": [
            "receive", "received", "receiving", "inbound",
            "container", "vendor", "return", "delivery",
            "pallet", "ref", "reference", "unload"
        ],
        "parameters": {
            "vendor": {
                "type": "string",
                "operator": "LIKE",
                "description": "Vendor name or code. Special values: CONTAINER, RETURN, FNS.",
                "examples": ["FNS", "CONTAINER", "RETURN", "VENDOR-A"]
            },
            "ref_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Reference number, container number, or tracking number.",
                "examples": ["CONT-123", "REF-456"]
            },
            "pallet_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Pallet number or pallet ID.",
                "examples": ["24110800033", "PAL-123"]
            },
            "date_from": {
                "type": "datetime",
                "operator": ">=",
                "description": "Start date for receiving date range.",
                "examples": ["{today}", "{yesterday}", "2024-01-01"]
            },
            "date_to": {
                "type": "datetime",
                "operator": "<=",
                "description": "End date for receiving date range.",
                "examples": ["{today}", "{week_end}"]
            },
            "item_code": {
                "type": "string",
                "operator": "LIKE",
                "description": "Item code or SAP code for received items.",
                "examples": ["584512091", "ITEM-123"]
            },
        }
    },
    "salesorder": {
        "url": "/salesorder.php",
        "description": """
        Simple sales order header endpoint for querying:
        - Order status and dates
        - Customer and PO information
        - Item number and SAP code
        """,
        "keywords": [
            "salesorder", "sales order", "so", "order header",
            "customer", "po", "purchase order", "item", "sap"
        ],
        "parameters": {
            "so_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Sales order number (partial or full).",
                "examples": ["12345", "SO12345"]
            },
            "status": {
                "type": "string",
                "operator": "exact",
                "description": "Sales order status (e.g. C, Ca, open).",
                "examples": ["C", "Ca", "open"]
            },
            "so_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Order date start (header SO date).",
                "examples": ["{today}", "2025-01-01"]
            },
            "so_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "Order date end (header SO date).",
                "examples": ["{today}", "2025-12-31"]
            },
            "ship_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Requested ship date start.",
                "examples": ["{today}", "2025-01-01"]
            },
            "ship_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "Requested ship date end.",
                "examples": ["{today}", "2025-12-31"]
            },
            "customer": {
                "type": "string",
                "operator": "LIKE",
                "description": "Customer name (partial match).",
                "examples": ["Amazon", "Costco"]
            },
            "cust_po_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Customer PO number (partial match).",
                "examples": ["11319997", "PO-2024-001"]
            },
            "item_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Item number / style number.",
                "examples": ["PROD-001", "1234"]
            },
            "sap_code": {
                "type": "string",
                "operator": "LIKE",
                "description": "SAP code / item code/SKU.",
                "examples": ["584512091", "584510753"]
            },
            "order_qty": {
                "type": "integer",
                "operator": "=",
                "description": "Exact order quantity filter.",
                "examples": ["100", "500"]
            },
        }
    },

    "warehouse": {
        "url": "/warehouse.php",
        "description": """
        Warehouse header endpoint for dock/crew scheduling:
        - Receiving date and pallet details
        - Routing date and crew assignment
        - Process start/end timestamps
        - Summary flags (Y/CA/empty)
        """,
        "keywords": [
            "warehouse", "dock", "crew", "routing", "pallet",
            "process", "process_start", "process_end", "summary"
        ],
        "parameters": {
            "rcvd_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "fulfillment Received by warehouse date start.",
                "examples": ["{today}", "2025-01-01"]
            },
            "rcvd_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "fulfillment Received by warehouse Received date end.",
                "examples": ["{today}", "2025-12-31"]
            },
            "so_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Sales order number for the warehouse row.",
                "examples": ["12345", "SO12345"]
            },
            "account": {
                "type": "string",
                "operator": "LIKE",
                "description": "Account/customer on the warehouse record.",
                "examples": ["Costco", "Amazon"]
            },
            "po_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Purchase order number on warehouse record.",
                "examples": ["11319997", "PO-2024-001"]
            },
            "ctn_num": {
                "type": "integer",
                "operator": "=",
                "description": "Carton count on the warehouse record.",
                "examples": ["10", "50"]
            },
            "routing_date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Routing date start.",
                "examples": ["{today}", "2025-01-01"]
            },
            "routing_date_to": {
                "type": "date",
                "operator": "<=",
                "description": "Routing date end.",
                "examples": ["{today}", "2025-12-31"]
            },
            "routing_date_is": {
                "type": "string",
                "operator": "special",
                "description": """Routing date emptiness: 'empty' or 'not_empty'. if 'routing_date_is' is empty, the sales order has not been received by warehouse as fulfillment.
                                    if 'routing_date_is' is not_empty , the sales order has been received as fulfillment.""",
                "examples": ["empty", "not_empty"]
            },
            "crew": {
                "type": "string",
                "operator": "exact",
                "description": "Crew assignment. One of song, local, t1..t6, all, empty, not_empty.",
                "examples": ["t1", "song", "local", "empty", "not_empty","t2","t3","t4","t5","t6"]
            },
            "process_start_from": {
                "type": "date",
                "operator": ">=",
                "description": " order fulfillment Process start timestamp (>=).",
                "examples": ["{today}", "2025-01-01"]
            },
            "process_start_to": {
                "type": "date",
                "operator": "<=",
                "description": "order fulfillment Process start timestamp (<=).",
                "examples": ["{today}", "2025-12-31"]
            },
            "process_end_from": {
                "type": "date",
                "operator": ">=",
                "description": "order fulfillment Process end timestamp (>=).",
                "examples": ["{today}", "2025-01-01"]
            },
            "process_end_to": {
                "type": "date",
                "operator": "<=",
                "description": "order fulfillment Process end timestamp (<=).",
                "examples": ["{today}", "2025-12-31"]
            },
            "summary": {
                "type": "string",
                "operator": "exact",
                "description": "Summary code: 'Y', 'CA', or 'empty' for not completed. status order fulfillment process.",
                "examples": ["y", "ca", "empty"]
            },
            "pallet_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Pallet count for the completed order fulfillment.",
                "examples": ["24110800033", "PAL-123"]
            },
        }
    },

    "outbound": {
        "url": "/outbound.php",
        "description": """
        Outbound/shipping endpoint for:
        - ETD / shipped dates
        - Carrier and BOL
        - Shipped/Not shipped status
        """,
        "keywords": [
            "outbound", "shipping", "ship out", "etd", "carrier",
            "bol", "tracking", "shipped", "not shipped"
        ],
        "parameters": {
            "so_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Sales order number (partial or full).",
                "examples": ["12345", "SO12345"]
            },
            "po_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Customer PO associated to shipment.",
                "examples": ["11319997", "PO-2024-001"]
            },
            "account": {
                "type": "string",
                "operator": "LIKE",
                "description": "Account/customer name.",
                "examples": ["Costco", "Target"]
            },
            "bol": {
                "type": "string",
                "operator": "LIKE",
                "description": "Bill of Lading number.",
                "examples": ["BOL12345"]
            },
            "carrier": {
                "type": "string",
                "operator": "LIKE",
                "description": "Shipping carrier name.",
                "examples": ["UPS", "FedEx", "USPS"]
            },
            "etd_from": {
                "type": "date",
                "operator": ">=",
                "description": "estimated time of departure of the work order start date.",
                "examples": ["{today}", "{week_start}"]
            },
            "etd_to": {
                "type": "date",
                "operator": "<=",
                "description": "estimated time of departure of the work order end date.",
                "examples": ["{today}", "{week_end}"]
            },
            "shipped_date_from": {
                "type": "datetime",
                "operator": ">=",
                "description": "date and time range start when the order has shipped out. only applies to orders that has been shipped out.",
                "examples": ["{today}", "{yesterday}"]
            },
            "shipped_date_to": {
                "type": "datetime",
                "operator": "<=",
                "description": "date and time range end when the order has shipped out. only applies to orders that has been shipped out.",
                "examples": ["{today}", "{yesterday}"]
            },
            "shipped_status": {
                "type": "string",
                "operator": "special",
                "description": "Shipped status filter: 'empty', 'not_empty', 'C'.",
                "examples": ["empty", "not_empty", "C", "1"]
            },
        }
    },

    "picktix": {
        "url": "/picktix.php",
        "description": """
        Pick ticket endpoint:
        - Pick activity by location and PO
        - Missing cartons/units
        - Pick status by date range
        """,
        "keywords": [
            "picktix", "pick ticket", "picking", "location",
            "missing", "picked", "ctn", "units"
        ],
        "parameters": {
            "date_from": {
                "type": "date",
                "operator": ">=",
                "description": "Start date for pick ticket issue date.",
                "examples": ["{today}", "2025-01-01"]
            },
            "date_to": {
                "type": "date",
                "operator": "<=",
                "description": "End date for pick ticket issue date.",
                "examples": ["{today}", "2025-12-31"]
            },
            "so_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Sales order number on pick ticket.",
                "examples": ["12345", "SO12345"]
            },
            "account": {
                "type": "string",
                "operator": "LIKE",
                "description": "Account/customer on the pick ticket.",
                "examples": ["Costco", "Amazon"]
            },
            "po_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Customer PO on the pick ticket.",
                "examples": ["11319997", "PO-2024-001"]
            },
            "item_num": {
                "type": "string",
                "operator": "LIKE",
                "description": "Item number / style number picked.",
                "examples": ["PROD-001", "1234"]
            },
            "sap_code": {
                "type": "string",
                "operator": "LIKE",
                "description": "SKU/SAP code for picked item.",
                "examples": ["584512091", "584510753"]
            },
            "location": {
                "type": "string",
                "operator": "LIKE",
                "description": "Pick location / bin.",
                "examples": ["21-X-1", "12-X-1"]
            },
            "status": {
                "type": "string",
                "operator": "exact",
                "description": "Pick ticket status (e.g. Active, Cancelled).",
                "examples": ["Active", "Cancelled"]
            },
            "missing_ctn_num": {
                "type": "integer",
                "operator": ">=",
                "description": "number of boxes of the SKU missing from pick ticket location and picked quantity.",
                "examples": ["1", "10"]
            },
            "missing_units": {
                "type": "integer",
                "operator": ">=",
                "description": "number of units of the SKU missing from pick ticket location and picked quantity.",
                "examples": ["1", "50"]
            },
            "note": {
                "type": "string",
                "operator": "LIKE",
                "description": "Free text search on pick notes.",
                "examples": ["damage", "short"]
            },
        }
    },


}


# =============================================================================
# DATE PLACEHOLDER DEFINITIONS
# =============================================================================

DATE_PLACEHOLDERS = {
    "today": "Current date",
    "yesterday": "Previous day",
    "tomorrow": "Next day",
    "week_start": "Start of current week (Monday)",
    "week_end": "End of current week (Sunday)",
    "last_week_start": "Start of previous week",
    "last_week_end": "End of previous week",
    "next_week_start": "Start of next week",
    "next_week_end": "End of next week",
    "month_start": "Start of current month",
    "month_end": "End of current month",
    "last_month_start": "Start of previous month",
    "last_month_end": "End of previous month",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_endpoint_by_name(endpoint_name: str):
    """Get endpoint configuration by name"""
    return API_ENDPOINTS.get(endpoint_name)


def get_all_endpoint_names():
    """Get list of all endpoint names"""
    return list(API_ENDPOINTS.keys())


def get_endpoint_keywords():
    """Get mapping of keywords to endpoints for routing"""
    keyword_map = {}
    for endpoint_name, config in API_ENDPOINTS.items():
        for keyword in config.get("keywords", []):
            if keyword not in keyword_map:
                keyword_map[keyword] = []
            keyword_map[keyword].append(endpoint_name)
    return keyword_map


def get_all_parameters_for_endpoint(endpoint_name: str):
    """Get all parameters for a specific endpoint"""
    endpoint = API_ENDPOINTS.get(endpoint_name)
    if endpoint:
        return endpoint.get("parameters", {})
    return {}
