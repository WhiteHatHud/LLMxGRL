# utils/io.py
import os
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Any, filepath: str):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def save_jsonl(data: List[Dict], filepath: str):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item, default=str) + '\n')

def save_csv(df: pd.DataFrame, filepath: str):
    ensure_dir(os.path.dirname(filepath))
    df.to_csv(filepath, index=False)

def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def save_upload(content: bytes, filepath: str):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        f.write(content)# File save/load utilities; uploads/
