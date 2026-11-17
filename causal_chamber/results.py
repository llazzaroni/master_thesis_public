"""Read completed snapshots and live, append-only experiment results."""

import json
from pathlib import Path


def load_result_document(path: Path) -> dict:
    document = json.loads(path.read_text())
    if document.get("storage") == "jsonl":
        journal = path.with_name("results.jsonl")
        rows = []
        if journal.exists():
            for line in journal.read_bytes().splitlines(keepends=True):
                # An actively appended last row can be incomplete. All earlier
                # rows must be valid; never silently discard a complete bad row.
                if not line.endswith(b'\n'):
                    break
                rows.append(json.loads(line))
        document = {**document, "results": rows}
    return document
