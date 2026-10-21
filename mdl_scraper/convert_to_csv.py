import json
import csv
 
INPUT_FILE = "dramas_final.jsonl"
OUTPUT_FILE = "dramas_final.csv"
 
 
def flatten_list(value):
    if not value:
        return ""
    return " | ".join(str(v) for v in value)
 
 
def flatten_cast(cast_list):
    if not cast_list:
        return ""
    parts = []
    for c in cast_list:
        actor = c.get("actor") or ""
        character = c.get("character")
        role = c.get("role_type") or ""
        if character:
            parts.append(f"{actor} ({character}) [{role}]")
        else:
            parts.append(f"{actor} [{role}]")
    return "; ".join(parts)
 
 
rows = []
with open(INPUT_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
 
        flat_row = dict(row)  # copy
        flat_row["genres"] = flatten_list(row.get("genres"))
        flat_row["tags"] = flatten_list(row.get("tags"))
        flat_row["also_known_as"] = flatten_list(row.get("also_known_as"))
        flat_row["cast_summary"] = flatten_cast(row.get("cast"))
        flat_row["cast_json"] = json.dumps(row.get("cast") or [], ensure_ascii=False)
        flat_row.pop("cast", None)  # remove original nested field
 
        rows.append(flat_row)
 
# Collect all possible column names across rows (in case some rows have extra/missing keys)
fieldnames = []
seen = set()
for r in rows:
    for k in r.keys():
        if k not in seen:
            seen.add(k)
            fieldnames.append(k)
 
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
 
print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")
print(f"Columns: {fieldnames}")
 