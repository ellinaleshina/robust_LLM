from __future__ import annotations
import re, random
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

import pandas as pd

ZW = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
NBSP = "\u00A0"
VS16 = "\ufe0f"
OPT_PATTERNS = [
    r"(['\"])Option\s*1\1",
    r"(['\"])Option\s*2\1",
    r"\bOption\s*1\b",
    r"\bOption\s*2\b",
]

def _freeze_patterns(text: str, patterns: List[str]) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    idx = 0
    def repl(m):
        nonlocal idx
        original = m.group(0)
        token = f"¤{idx}¤"
        mapping[token] = original
        idx += 1
        return token
    for pat in patterns:
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text, mapping

def _unfreeze(text: str, mapping: Dict[str, str]) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text

def _inject_zw(s: str, p: float) -> Tuple[str, int]:
    count = 0
    out = []
    for ch in s:
        out.append(ch)
        if ch.isalpha() and random.random() < p:
            out.append(random.choice(ZW))
            count += 1
    return "".join(out), count

def _space_noise(s: str, p_nbsp: float, p_nl: float) -> Tuple[str, Dict[str, int]]:
    stats = {"nbsp": 0, "nl": 0}
    def repl_sp(m):
        if random.random() < p_nbsp:
            stats["nbsp"] += 1
            return NBSP
        return m.group(0)
    s = re.sub(r" ", repl_sp, s)
    def repl_ws(m):
        if random.random() < p_nl:
            stats["nl"] += 1
            return "\n"
        return m.group(0)
    s = re.sub(r"[ \t]{2,}", repl_ws, s)
    return s, stats

@dataclass
class PerturbConfig:
    level: str = "H"
    seed: int = 42
    p_zw: float = 0.03
    p_nbsp: float = 0.06
    p_nl: float = 0.05
    wrappers_min: int = 1
    wrappers_max: int = 2
    enable_comments: bool = True
    enable_codefence: bool = True
    enable_blockquote: bool = True
    enable_header: bool = True

    @staticmethod
    def for_level(level: str, seed: int) -> "PerturbConfig":
        L = level.upper()
        if L == "L":
            return PerturbConfig(level="L", seed=seed, p_zw=0.01, p_nbsp=0.03, p_nl=0.02,
                                 wrappers_min=0, wrappers_max=1)
        if L == "H":
            return PerturbConfig(level="H", seed=seed, p_zw=0.07, p_nbsp=0.12, p_nl=0.10,
                                 wrappers_min=2, wrappers_max=3)
        return PerturbConfig(level="M", seed=seed)

def perturb_instruction(instr_text: str, level: str, seed: int) -> Tuple[str, Dict]:
    cfg = PerturbConfig.for_level(level, seed)
    random.seed(cfg.seed)
    meta = {"level": cfg.level, "seed": cfg.seed, "steps": []}

    frozen_text, mapping = _freeze_patterns(instr_text or "", OPT_PATTERNS)
    meta["frozen_count"] = len(mapping)

    tmp, space_stats = _space_noise(frozen_text, cfg.p_nbsp, cfg.p_nl)
    meta["steps"].append({"op": "space_noise", **space_stats})
    frozen_text = tmp

    frozen_text, zw_count = _inject_zw(frozen_text, cfg.p_zw)
    meta["steps"].append({"op": "insert_zw", "count": zw_count})

    def add_vs16(m): return m.group(0) + VS16
    prob = 0.3 if cfg.level == "L" else 0.6 if cfg.level == "M" else 0.85
    import re as _re
    if random.random() < prob:
        frozen_text = _re.sub(r"[:;,.!?()]", add_vs16, frozen_text, count=3)
        meta["steps"].append({"op": "vs16_punct", "count": 3})

    wrappers = []
    if random.random() < 0.5:
        frozen_text = "```\n" + frozen_text + "\n```"
        wrappers.append("codefence")
    meta["steps"].append({"op": "wrappers", "used": wrappers})

    out = _unfreeze(frozen_text, mapping)
    meta["summary"] = asdict(cfg)
    return out, meta
