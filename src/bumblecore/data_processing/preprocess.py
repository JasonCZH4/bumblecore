from typing import List, Dict, Any
from pathlib import Path
import json


def load_data(data_path: str) -> List[Dict[str, Any]]:
    path = Path(data_path)
    if path.suffix == ".json":
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif path.suffix == ".jsonl":
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")


def load_pretrain_data(data_path: str) -> List[Dict[str, Any]]:
    return load_data(data_path)


def load_sft_data(data_path: str) -> List[Dict[str, Any]]:
    return load_data(data_path)


def load_dpo_data(data_path: str) -> List[Dict[str, Any]]:
    return load_data(data_path)
