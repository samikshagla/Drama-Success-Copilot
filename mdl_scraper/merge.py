"""
Merge dramas_balanced.jsonl and dramas_recent.jsonl into one deduplicated dataset.
Usage: python merge_dramas.py
Expects both input files in the same directory as this script (or edit paths below).
"""
import json

INPUT_FILES = ["dramas_balanced.jsonl", "dramas_recent.jsonl"]
OUTPUT_FILE = "dramas_final.jsonl"

seen_urls = set()
merged_rows = []
duplicates_skipped = 0
bad_lines = 0

for path in INPUT_FILES:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            url = row.get("url")
            if not url:
                # keep rows with no url rather than silently dropping them
                merged_rows.append(row)
                continue

            if url in seen_urls:
                duplicates_skipped += 1
                continue

            seen_urls.add(url)
            merged_rows.append(row)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for row in merged_rows:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Merged {len(merged_rows)} unique dramas into {OUTPUT_FILE}")
print(f"Skipped {duplicates_skipped} duplicate URLs")
print(f"Skipped {bad_lines} malformed lines")