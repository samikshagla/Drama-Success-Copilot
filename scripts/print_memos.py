"""
scripts/print_memos.py
Renders executive decision memorandums for fixtures 01, 05, 08, 14, and 16
from committed fixture files, saves them as golden reference files in tests/golden/,
and prints them to stdout with fixture ID headers (outside memo text).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent import AgentOrchestrator, MockLLM
from src.agent_tools import load_explain_artifacts, load_retriever_artifacts
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
FIXTURES_FILE = REPO_ROOT / "tests" / "fixtures" / "pitches_eval.json"

CONFIGS = [
    {
        "id": "pitch_01_complete_kr_thriller",
        "file_name": "memo_01.md",
        "changes": None,
    },
    {
        "id": "pitch_05_blank_network_and_rating",
        "file_name": "memo_05.md",
        "changes": {"network": "JTBC", "content_rating": "15+"},
    },
    {
        "id": "pitch_08_blank_everything_except_country_and_start",
        "file_name": "memo_08.md",
        "changes": None,
    },
    {
        "id": "pitch_14_adversarial_disallowed_lever",
        "file_name": "memo_14.md",
        "changes": {"episodes": 100},
    },
    {
        "id": "pitch_16_all_blank_optional_cn",
        "file_name": "memo_16.md",
        "changes": None,
    },
]


def render_and_save_golden_memos(overwrite: bool = True) -> Dict[str, str]:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIXTURES_FILE, "r", encoding="utf-8") as f:
        pitches = json.load(f)
    pitch_map = {p["id"]: p for p in pitches}

    explain_artifacts = load_explain_artifacts()
    retriever_artifacts = load_retriever_artifacts()
    orchestrator = AgentOrchestrator(
        llm_client=MockLLM(),
        explain_artifacts=explain_artifacts,
        retriever_artifacts=retriever_artifacts,
    )

    rendered: Dict[str, str] = {}

    for cfg in CONFIGS:
        pid = cfg["id"]
        fname = cfg["file_name"]
        changes = cfg["changes"]
        pitch = pitch_map[pid]

        req: Dict[str, Any] = {"pitch": pitch}
        if changes:
            req["changes"] = changes

        result = orchestrator.run(req)
        memo = result["memo"]
        rendered[fname] = memo

        if overwrite:
            out_path = GOLDEN_DIR / fname
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(memo)

        # Print fixture ID in header outside of memo text
        print(f"\n{'=' * 78}")
        print(f"FIXTURE: {pid} -> {fname}")
        print(f"{'=' * 78}\n")
        print(memo)

    return rendered


if __name__ == "__main__":
    render_and_save_golden_memos(overwrite=True)
