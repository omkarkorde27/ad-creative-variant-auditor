import json
from pathlib import Path

REQUIRED_KEYS = {"name", "label", "max_chars", "style"}

def load_platform_rules(path: str = "data/platform_rules.json") -> list:
    data = json.loads(Path(path).read_text())
    platforms = data.get("platforms", [])
    if not platforms:
        raise ValueError("platform_rules.json has no platforms defined")
    for p in platforms:
        missing = REQUIRED_KEYS - p.keys()
        if missing:
            raise ValueError(f"Platform rule {p.get('name', '?')} missing keys: {missing}")
    return platforms