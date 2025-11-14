
# Database Configuration
DB_URL = "postgresql://postgres:@localhost:5432/creme"

# MySQL Configuration
MYSQL_CONFIG = {
    "host": "10.40.96.173",
    "user": "root",
    "password": "mandoo69!",
    "database": "creme"
}

# Ollama Configuration
OLLAMA_OPTIONS = {
    "num_ctx": 4096,
}
ROUTING_MODEL = 'llama3.1:latest'
#ROUTING_MODEL = 'mistral-nemo:latest'
#LLM_MODEL = 'llama3.1:latest'
LLM_MODEL = 'llama3.1:latest'
#LLM_MODEL = 'mistral-small:24b'
#LLM_MODEL = 'gpt-oss:20b'
#LLM_MODEL = 'phi4:latest'
#LLM_MODEL = 'mistral-nemo:latest'
#LLM_MODEL = 'gemma3:12b'
#LLM_MODEL = 'phi4-mini:3.8b'
# Table Configuration
TABLE_NAMES = [
    'outbound', 'picktix', 'products', 'salesorder', 'warehouse',
    'inbound', 'repack', 'sto', 'sto_inbound', 'errand', 'inventory','delivery_order'
]

# Keyword to Table Mapping
KEYWORD_TABLE_MAP = {
    "sales": ["salesorder"],
    "crew": ["warehouse"],
    "master qty": ["products"],
    "picked": ["picktix"],
    "shipping": ["outbound"],
    "ship": ["outbound"],
    "shipped": ["outbound"],
    "pick ticket": ["picktix"],
    "work order": ["warehouse"],
    "shipment": ["outbound","inbound"],
    "warehouse": ["warehouse"],
    "routing date":["warehouse"],
    "sales order": ["salesorder"],
    "to local":["sto","inbound"],
    "to FNS":["sto"],
    "to KCC":["sto"],
    "to e-commerce":["sto"],
    "from KCC":["inbound"],
    "from FNS":["inbound"],
    "air":["inbound"],
    "container":["inbound"],
    "repack":["repack","delivery_order","salesorder","warehouse"],
    "repacking":["repack","delivery_order","salesorder","warehouse"],
    "ticketing":["repack","delivery_order","salesorder","warehouse"],
    "receive":["inbound"],
    "received":["inbound"],
    "pick up":["outbound"],
    "carrier":["outbound"],
    "sto":["sto"],
    "ETA item":["salesorder","warehouse","picktix","repack"],
    "T1":["warehouse"],
    "T2":["warehouse"],
    "T3":["warehouse"],
    "missing":["picktix", "salesorder"],
}

# Column Equivalence Mapping
COLUMN_EQUIVALENCE = {
    "custoemr": ["account","customer"],
    "po number": ["cust_po_num","po_num"],
    "po": ["cust_po_num","po_num"],
    "so": ["so_num"],
    "new code": ["item_code", "SAP","sap_code"],
    "sap code": ["sap_code", "SAP","item_code"],
    "shipped":["shipped", "shipped_date"],
    "ship":["shipped",'ETD'],
}

# Follow-up Detection Keywords
FOLLOW_UP_KEYWORDS = [
    "this", "that", "previous", "last", "these",
    "those", "above table", "same", "last list", "now", "then",
    "result", "results", "data", "output"
]

# Coreference Phrases
COREFERENCE_PHRASES = [
    "those po", "these po", "those order", "these order",
    "those shipment", "these shipment",
    "the same po", "the prior po", "the above po", "the previous po",
    "last result", "previous result", "that result", "those results",
    "the above results", "previous data", "last data", "that data",
    "last query", "previous query", "last search"
]

# RAG Configuration
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 5
RAG_MIN_SIMILARITY = 0.1

# Supported document formats
SUPPORTED_DOC_FORMATS = {'.pdf', '.docx', '.txt', '.csv', '.xlsx'}

# Document processing settings
MAX_FILE_SIZE_MB = 50
MAX_FILES_PER_UPLOAD = 10

WEB_SEARCH_ENABLED = True
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_PROVIDER = "searxng"  # optional, for clarity
SEARXNG_URL = "http://localhost:8090"  # or LAN IP
  