import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PATTERNS = [
    r"sk-[a-zA-Z0-9]{20,}",
    r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
    r"bearer\s+[a-zA-Z0-9_\-\.]{20,}",
]

clean = True
for ext in ["*.py", "*.json", "*.md"]:
    for p in REPO_ROOT.rglob(ext):
        if "venv" in p.parts or ".git" in p.parts or "models" in p.parts:
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
            for pat in PATTERNS:
                if re.search(pat, content, re.IGNORECASE):
                    print(f"Warning: suspicious pattern {pat} in {p}")
                    clean = False
        except Exception:
            pass

if clean:
    print("Safety scan PASSED: Zero credentials, private API keys, or bearer tokens detected.")
else:
    print("Safety scan FAILED!")
