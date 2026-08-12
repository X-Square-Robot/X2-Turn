"""预设剧本: 优先从已评估 preds 里挑 category_ok 的样本, 保证 demo 可演示。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_TEST_JSONL = ""
DEFAULT_PREDS_JSONL = ""


@dataclass
class Scenario:
    key: str
    title: str
    category: str
    mode: str          # listen | barge_in
    expect: str        # ACCEPT / REJECT / HOLD / BARGE_THEN_*
    wav: str
    text: str
    tip: str


def _cat_of(wav: str) -> str:
    if "/testset/" in wav:
        return wav.split("/testset/")[1].split("/")[0]
    return "unknown"


def _load_from_preds(preds_jsonl: str, per_cat: int = 8) -> Dict[str, List[dict]]:
    """只取 category_ok=True 的样本, demo 体验更稳。"""
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
    # 缺类时用 testset 补齐
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

    def pick(cat: str, i: int = 0) -> Optional[dict]:
        xs = pool.get(cat) or []
        return xs[i] if i < len(xs) else None

    specs = [
        ("complete", "listen", "ACCEPT", "完整一句 → 期望 turn_end → 绿灯接话"),
        ("wait", "listen", "ACCEPT", "结束指令/wait → 期望 turn_end → 接话停播"),
        ("backchannel", "listen", "REJECT", "嗯/哦对 → 期望 backchannel → 拒识不回复"),
        ("incomplete", "listen", "HOLD", "半句话 → 期望 speaking/uncertain → 继续听"),
        ("complete", "barge_in", "BARGE_THEN_ACCEPT", "Bot 播报中用户插完整句 → 先打断再接话"),
        ("backchannel", "barge_in", "BARGE_OR_IGNORE", "Bot 播报中用户只嗯一声 → 尽量不打断或打断后拒识"),
        ("incomplete", "barge_in", "BARGE_THEN_HOLD", "Bot 播报中用户半句插入 → 打断后继续听"),
    ]
    used = {c: 0 for c in ("complete", "wait", "backchannel", "incomplete")}
    for cat, mode, expect, tip in specs:
        i = used[cat]
        item = pick(cat, i)
        used[cat] = i + 1
        if item is None:
            # 同类别再往后找
            for j in range(i + 1, len(pool.get(cat) or [])):
                item = pick(cat, j)
                if item is not None:
                    used[cat] = j + 1
                    break
        if item is None:
            continue
        key = f"{mode}_{cat}_{i}"
        title = {
            "listen": f"[听] {cat}",
            "barge_in": f"[打断] {cat}",
        }[mode] + f": {item['text'][:24]}"
        out.append(
            Scenario(
                key=key,
                title=title,
                category=cat,
                mode=mode,
                expect=expect,
                wav=item["wav"],
                text=item["text"],
                tip=tip,
            )
        )
    return out


def scenario_choices(scenarios: List[Scenario]) -> Dict[str, Scenario]:
    return {s.title: s for s in scenarios}
