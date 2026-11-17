"""Report CSV sizes and columns without loading the data into memory."""

import argparse
import csv
import json
import sys
from pathlib import Path

from causal_chamber.splits import DEFAULT_DATA_DIR, describe_csv, discover, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, help="Optional CSV report")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text()) if args.manifest else None
        if manifest:
            verify_manifest(manifest, args.data_dir)
        rows = []
        for name, path in discover(args.data_dir).items():
            info = manifest["environments"][name] if manifest else describe_csv(path)
            row = {"environment": name, "rows": info["num_rows"]}
            if manifest:
                row.update({key: len(info["splits"][key]) for key in ("train", "val", "test")})
                row["target_budget_max"] = row["train"] + row["val"]
            row["columns"] = ",".join(info["columns"])
            rows.append(row)
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        counts = [r["rows"] for r in rows]
        print(f"Environments: {len(rows)}; rows: min={min(counts)}, max={max(counts)}, total={sum(counts)}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
