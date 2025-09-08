# utils/ids.py
import uuid
from datetime import datetime

def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{unique_id}"

def generate_query_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"query_{timestamp}_{unique_id}"

def generate_graph_id() -> str:
    unique_id = str(uuid.uuid4())[:12]
    return f"graph_{unique_id}"# run_id, query_id utilities
