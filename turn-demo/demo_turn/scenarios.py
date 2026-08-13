"""Preset scenarios: prefer evaluated samples marked category_ok for a reliable demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_TEST_JSONL = ""
DEFAULT_PREDS_JSONL = ""
BUILTIN_SAMPLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "sample_en.wav")
)
BUILTIN_TEXT = "Hello, could you tell me what the weather is like today?"


@dataclass
class Scenario:
    key: str
    title: str
    category: str
    wav: str
    text: str
    tip: str


def _cat_of(wav: str) -> str:
    if "/testset/" in wav:
        return wav.split("/testset/")[1].split("/")[0]
    return "unknown"


def _load_from_preds(preds_jsonl: str, per_cat: int = 8) -> Dict[str, List[dict]]:
    """Use only samples with category_ok=True for a more reliable demo."""
    by_cat: Dict[str, List[dict]] = {
        "complete": [],
        "incomplete": [],
        "backchannel": [],
        "wait": [],
    }
    if not os.path.isfile(preds_jsonl):
        return by_cat
    with open(preds_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            cat = rec.get("category") or _cat_of(rec.get("wav") or "")
            if cat not in by_cat or len(by_cat[cat]) >= per_cat:
                continue
            if not rec.get("category_ok"):
                continue
            wav = rec.get("wav") or ""
            if not wav or not os.path.isfile(wav):
                continue
            by_cat[cat].append(
                {
                    "wav": wav,
                    "text": rec.get("text") or "",
                    "last_char_pred": rec.get("last_char_pred"),
                    "asr_text": rec.get("asr_text") or "",
                }
            )
            if all(len(v) >= per_cat for v in by_cat.values()):
                break
    return by_cat


def _load_from_test(test_jsonl: str, per_cat: int = 5) -> Dict[str, List[dict]]:
    by_cat: Dict[str, List[dict]] = {
        "complete": [],
        "incomplete": [],
        "backchannel": [],
        "wait": [],
    }
    if not os.path.isfile(test_jsonl):
        return by_cat
    with open(test_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            wav = obj.get("wav") or ""
            cat = _cat_of(wav)
            if cat not in by_cat or len(by_cat[cat]) >= per_cat:
                continue
            if not os.path.isfile(wav):
                continue
            by_cat[cat].append({"wav": wav, "text": obj.get("text") or ""})
            if all(len(v) >= per_cat for v in by_cat.values()):
                break
    return by_cat


def load_scenario_pool(
    test_jsonl: str = DEFAULT_TEST_JSONL,
    preds_jsonl: str = DEFAULT_PREDS_JSONL,
    per_cat: int = 8,
) -> Dict[str, List[dict]]:
    pool = _load_from_preds(preds_jsonl, per_cat=per_cat)
    # Fill missing categories from the test set.
    if any(len(v) == 0 for v in pool.values()):
        fallback = _load_from_test(test_jsonl, per_cat=per_cat)
        for cat, xs in fallback.items():
            if not pool[cat]:
                pool[cat] = xs
    return pool


def build_scenarios(
    test_jsonl: str = DEFAULT_TEST_JSONL,
    preds_jsonl: str = DEFAULT_PREDS_JSONL,
) -> List[Scenario]:
    pool = load_scenario_pool(test_jsonl, preds_jsonl, per_cat=8)
    out: List[Scenario] = []
    if os.path.isfile(BUILTIN_SAMPLE):
        out.append(
            Scenario(
                key="builtin_sample_en",
                title="[built-in] English question",
                category="complete",
                wav=BUILTIN_SAMPLE,
                text=BUILTIN_TEXT,
                tip="Bundled synthetic Quickstart audio",
            )
        )

    def pick(cat: str, i: int = 0) -> Optional[dict]:
        xs = pool.get(cat) or []
        return xs[i] if i < len(xs) else None

    specs = [
        ("complete", "Frame-level Turn states for a complete sentence"),
        ("wait", "Frame-level Turn states for a wait instruction"),
        ("backchannel", "Frame-level Turn states for a brief acknowledgment"),
        ("incomplete", "Frame-level Turn states for an incomplete utterance"),
    ]
    used = {c: 0 for c in ("complete", "wait", "backchannel", "incomplete")}
    for cat, tip in specs:
        i = used[cat]
        item = pick(cat, i)
        used[cat] = i + 1
        if item is None:
            # Search later entries in the same category.
            for j in range(i + 1, len(pool.get(cat) or [])):
                item = pick(cat, j)
                if item is not None:
                    used[cat] = j + 1
                    break
        if item is None:
            continue
        key = f"{cat}_{i}"
        title = f"[{cat}] {item['text'][:24]}"
        out.append(
            Scenario(
                key=key,
                title=title,
                category=cat,
                wav=item["wav"],
                text=item["text"],
                tip=tip,
            )
        )
    return out


def scenario_choices(scenarios: List[Scenario]) -> Dict[str, Scenario]:
    return {s.title: s for s in scenarios}
