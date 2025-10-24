from __future__ import annotations
import os, re, json, argparse, random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    set_seed as hf_set_seed,
)


try:
    from peft import PeftModel
    PEFT_OK = True
except Exception:
    PEFT_OK = False

def canonical_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = s.replace("_", " ")
    return s.strip()

def choose_label_from_generation(gen_text: str, labels: List[str]) -> str:
    if not labels:
        return gen_text.strip()
    canon2label: Dict[str, str] = {canonical_text(lbl): lbl for lbl in labels}
    canon_labels = list(canon2label.keys())
    gen_low = gen_text.lower()
    m = re.search(r"(option\s*([12])\b)|(^|\b)([12])($|\b)", gen_low)
    if m and any("option" in cl for cl in canon_labels):
        digit = m.group(2) or m.group(4)
        if digit in ("1","2"):
            target = f"option {digit}"
            if target in canon2label:
                return canon2label[target]
    for lbl in labels:
        if lbl.lower() in gen_low:
            return lbl
    gen_can = canonical_text(gen_text)
    for cl in canon_labels:
        if cl and cl in gen_can:
            return canon2label[cl]
    gen_tokens = set(gen_can.split())
    best_lbl, best_score = labels[0], -1.0
    for cl in canon_labels:
        ltoks = set(cl.split())
        if not ltoks:
            continue
        score = len(gen_tokens & ltoks) / len(ltoks)
        if score > best_score:
            best_score = score
            best_lbl = canon2label[cl]
    return best_lbl

def metrics_from_preds(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    assert len(y_true) == len(y_pred)
    labels_sorted = sorted(list(set(y_true) | set(y_pred)))
    acc = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    idx_map = {lbl: i for i, lbl in enumerate(labels_sorted)}
    n = len(labels_sorted)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[idx_map[t], idx_map[p]] += 1
    f1s, precs, recs = [], [], []
    for i in range(n):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1); precs.append(prec); recs.append(rec)
    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "labels": labels_sorted,
        "confusion": conf.tolist(),
        "per_label": [
            {"label": labels_sorted[i], "precision": float(precs[i]),
             "recall": float(recs[i]), "f1": float(f1s[i])}
            for i in range(n)
        ],
    }

@dataclass
class Args:
    clean_dir: str
    pert_dir: str
    out_dir: str
    base_model: str
    peft_path: Optional[str]
    arch: str
    batch_size: int
    max_input_tokens: int
    gen_max_new_tokens: int
    seed: int
    use_chat_template: bool
    prompt_format: str
    splits: List[str]
    id_col: str
    instruction_col: str
    text_col_clean: str
    text_col_pert: str
    label_col: str
    join_format: str

def parse_args() -> Args:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True, help="Folder with {split}.csv")
    ap.add_argument("--pert_dir", required=True, help="Folder with {split}_llm_pert_w_input_instr.csv")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--peft_path", default=None)
    ap.add_argument("--arch", default="seqcls", choices=["causallm","seqcls"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_input_tokens", type=int, default=1024)
    ap.add_argument("--gen_max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--prompt_format", default="User: {text}\nAssistant:")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--instruction_col", default="instruction")
    ap.add_argument("--text_col_clean", default="input")
    ap.add_argument("--text_col_pert", default="text_pert")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--join_format", default="{instruction}\n\n{text}")
    a = ap.parse_args()
    return Args(
        clean_dir=a.clean_dir,
        pert_dir=a.pert_dir,
        out_dir=a.out_dir,
        base_model=a.base_model,
        peft_path=a.peft_path,
        arch=a.arch,
        batch_size=a.batch_size,
        max_input_tokens=a.max_input_tokens,
        gen_max_new_tokens=a.gen_max_new_tokens,
        seed=a.seed,
        use_chat_template=a.use_chat_template,
        prompt_format=a.prompt_format,
        splits=[s.strip() for s in a.splits.split(",") if s.strip()],
        id_col=a.id_col,
        instruction_col=a.instruction_col,
        text_col_clean=a.text_col_clean,
        text_col_pert=a.text_col_pert,
        label_col=a.label_col,
        join_format=a.join_format,
    )

def _model_device(m: torch.nn.Module) -> torch.device:
    return next(m.parameters()).device

def load_model_and_tokenizer(args: Args):
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = getattr(tok, "eos_token", None) or getattr(tok, "unk_token", None)

    LABELS   = ["Option 1", "Option 2"]
    LABEL2ID = {l:i for i,l in enumerate(LABELS)}
    ID2LABEL = {i:l for l,i in LABEL2ID.items()}

    if args.arch == "causallm":
        mdl = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            device_map="cuda" if use_cuda else None,
        )
        if args.peft_path:
            if not PEFT_OK: raise RuntimeError("peft is not installed but --peft_path was provided")
            mdl = PeftModel.from_pretrained(mdl, args.peft_path)
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            torch_dtype=dtype,
            device_map="cuda" if use_cuda else None,
        )
        mdl = base
        if args.peft_path:
            if not PEFT_OK: raise RuntimeError("peft is not installed but --peft_path was provided")
            mdl = PeftModel.from_pretrained(base, args.peft_path)

    if hasattr(mdl.config, "pad_token_id") and tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id

    mdl.eval()
    return mdl, tok

def _pick_id_col(cols: List[str], pref: str) -> Optional[str]:
    for c in [pref, "id", "id_raw", "example_id"]:
        if c in cols: return c
    return None

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

def _wrap_candidates(cfg: "PerturbConfig"):
    return [
        ("codefence", lambda t: "```\n" + t + "\n```") if cfg.enable_codefence else None,
        ("blockquote", lambda t: "> " + t.replace("\n", "\n> ")) if cfg.enable_blockquote else None,
        ("html_comment", lambda t: "<!-- instr -->\n" + t + "\n<!-- /instr -->") if cfg.enable_comments else None,
        ("md_header", lambda t: "# Task\n\n" + t) if cfg.enable_header else None,
    ]

def _apply_wrappers(instr: str, cfg: "PerturbConfig") -> Tuple[str, List[str]]:
    cand = [c for c in _wrap_candidates(cfg) if c is not None]
    k = random.randint(cfg.wrappers_min, cfg.wrappers_max) if cand else 0
    chosen = random.sample(cand, k) if k > 0 else []
    out = instr
    names = []
    for name, fn in chosen:
        out = fn(out)
        names.append(name)
    return out, names

@dataclass
class PerturbConfig:
    level: str = "M"
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
    def for_level(level: str, seed: int = 42) -> "PerturbConfig":
        level = level.upper()
        if level == "L":
            return PerturbConfig(level="L", seed=seed, p_zw=0.01, p_nbsp=0.03, p_nl=0.02,
                                 wrappers_min=0, wrappers_max=1)
        if level == "H":
            return PerturbConfig(level="H", seed=seed, p_zw=0.07, p_nbsp=0.12, p_nl=0.10,
                                 wrappers_min=2, wrappers_max=3)
        return PerturbConfig(level="M", seed=seed)

def perturb_instruction(instr_text: str, level: str = "", seed: int = 42) -> Tuple[str, Dict]:
    cfg = PerturbConfig.for_level(level, seed)
    random.seed(cfg.seed)
    meta = {"level": cfg.level, "seed": cfg.seed, "steps": []}

    frozen_text, mapping = _freeze_patterns(instr_text, OPT_PATTERNS)
    meta["frozen_count"] = len(mapping)

    tmp, space_stats = _space_noise(frozen_text, cfg.p_nbsp, cfg.p_nl)
    meta["steps"].append({"op": "space_noise", **space_stats})
    frozen_text = tmp

    frozen_text, zw_count = _inject_zw(frozen_text, cfg.p_zw)
    meta["steps"].append({"op": "insert_zw", "count": zw_count})

    def add_vs16(m): return m.group(0) + VS16
    prob = 0.3 if cfg.level == "L" else 0.6 if cfg.level == "M" else 0.85
    if random.random() < prob:
        frozen_text = re.sub(r"[:;,.!?()]", add_vs16, frozen_text, count=3)
        meta["steps"].append({"op": "vs16_punct", "count": 3})

    wrapped, used = _apply_wrappers(frozen_text, cfg)
    meta["steps"].append({"op": "wrappers", "used": used})

    out = _unfreeze(wrapped, mapping)

    meta["summary"] = asdict(cfg)
    return out, meta

def load_and_merge_split(clean_dir: str, pert_dir: str, split: str, id_pref: str,
                         instruction_col: str, text_col_clean: str, text_col_pert: str,
                         label_col: str, base_seed: int = 1337) -> pd.DataFrame:
    clean_path = os.path.join(clean_dir, f"{split}.csv")
    pert_path  = os.path.join(pert_dir,  f"{split}_llm_pert_w_input_instr.csv")
    if not os.path.exists(clean_path): raise FileNotFoundError(clean_path)
    if not os.path.exists(pert_path):  raise FileNotFoundError(pert_path)

    df_c = pd.read_csv(clean_path)
    df_p = pd.read_csv(pert_path)

    if "instruction" not in df_p.columns:
        raise KeyError(f"'instruction' column not found in {pert_path}")

    instr_list = df_p["instruction"].fillna("").astype(str).tolist()
    res = [perturb_instruction(s, level="H", seed=base_seed + i) for i, s in enumerate(instr_list)]
    df_p["instruction_pert"] = [r[0] for r in res]
    df_p["instr_meta"]       = [json.dumps(r[1], ensure_ascii=False) for r in res]
    df_p.to_csv(f"{split}_instr_perts.csv")


    id_c = _pick_id_col(df_c.columns.tolist(), id_pref)
    id_p = _pick_id_col(df_p.columns.tolist(), id_pref)
    if id_c is None or id_p is None:
        raise KeyError(f"ID column not found. Clean={list(df_c.columns)}, Pert={list(df_p.columns)}")

    need_clean = [col for col in [id_c, instruction_col, text_col_clean, label_col] if col in df_c.columns]
    df_c = df_c[need_clean].drop_duplicates(subset=[id_c])

    keep_pert = [id_p, text_col_pert, "instruction_pert", "instr_meta"]
    for extra in ["level", "aug_meta", "split"]:
        if extra in df_p.columns: keep_pert.append(extra)
    df_p = df_p[keep_pert].drop_duplicates(subset=[id_p])

    df = df_c.merge(df_p, left_on=id_c, right_on=id_p, how="inner", suffixes=("","_p"))
    df = df.rename(columns={id_c: "id"})
    if "id_p" in df.columns: df = df.drop(columns=["id_p"])

    renames = {}
    if instruction_col in df.columns:
        renames[instruction_col] = "instruction_clean"
    if text_col_clean in df.columns:
        renames[text_col_clean] = "text_clean"
    if text_col_pert in df.columns:
        renames[text_col_pert] = "text_pert"
    df = df.rename(columns=renames)

    # проверка
    required = ["id", "text_clean", "text_pert", "instruction_pert", label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after merge for split={split}: {missing}")

    return df

def build_joined_texts(df: pd.DataFrame, which: str, args: Args) -> List[str]:
    instr_col = "instruction_clean" if which == "clean" else "instruction_pert"
    if instr_col not in df.columns:
        raise KeyError(f"Column '{instr_col}' not in dataframe for which={which}")
    instr = df[instr_col].astype(str).fillna("")
    body  = (df["text_clean"] if which=="clean" else df["text_pert"]).astype(str).fillna("")
    return [args.join_format.format(instruction=i, text=t) for i, t in zip(instr, body)]

def make_prompts(texts: List[str], tok: AutoTokenizer, args: Args) -> List[str]:
    if args.use_chat_template and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return [tok.apply_chat_template([{"role":"user","content":str(t)}], tokenize=False, add_generation_prompt=True) for t in texts]
    return [args.prompt_format.format(text=str(t)) for t in texts]

@torch.no_grad()
def infer_causallm_texts(model, tok, prompts: List[str], args: Args) -> List[str]:
    out_texts: List[str] = []
    bs = args.batch_size
    dev = _model_device(model)
    for i in range(0, len(prompts), bs):
        batch_prompts = prompts[i:i+bs]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_input_tokens).to(dev)
        input_lens = (enc["input_ids"] != tok.pad_token_id).sum(dim=1)
        gen = model.generate(
            **enc,
            max_new_tokens=args.gen_max_new_tokens,
            do_sample=False,
            eos_token_id=getattr(tok, "eos_token_id", None),
            pad_token_id=tok.pad_token_id,
            num_beams=1,
        )
        for j in range(gen.shape[0]):
            gen_ids = gen[j, input_lens[j]:]
            txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
            out_texts.append(txt)
    return out_texts

@torch.no_grad()
def infer_seqcls_labels(model, tok, texts: List[str], id2label: Dict[int,str], args: Args) -> List[str]:
    out_labels: List[str] = []
    bs = args.batch_size
    dev = _model_device(model)
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_input_tokens).to(dev)
        logits = model(**enc).logits
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        out_labels.extend([id2label.get(pid, id2label.get(pid % len(id2label), str(pid))) for pid in pred_ids])
    return out_labels

def evaluate_on_dataframe(df: pd.DataFrame, model, tok, args: Args, which: str, label_universe: List[str]) -> Tuple[pd.DataFrame, Dict[str,Any]]:
    texts = build_joined_texts(df, which=which, args=args)
    golds = df[args.label_col].astype(str).tolist()
    if args.arch == "causallm":
        prompts = make_prompts(texts, tok, args)
        gens = infer_causallm_texts(model, tok, prompts, args)
        preds = [choose_label_from_generation(g, label_universe) for g in gens]
        df_out = df.copy()
        df_out["joined_text"] = texts
        df_out["pred_text"] = gens
        df_out["pred_label"] = preds
    else:
        if hasattr(model.config, "id2label") and model.config.id2label:
            if isinstance(model.config.id2label, dict):
                id2label = {int(k): v for k, v in model.config.id2label.items()}
            else:
                id2label = dict(enumerate(model.config.id2label))
        else:
            id2label = {i: l for i, l in enumerate(label_universe)}
        preds = infer_seqcls_labels(model, tok, texts, id2label, args)
        df_out = df.copy()
        df_out["joined_text"] = texts
        df_out["pred_label"] = preds
    m = metrics_from_preds(golds, preds)
    return df_out, m

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_bar(title: str, items: List[Tuple[str, float]], out_path: str, ylim01: bool=True):
    labels = [k for k,_ in items]; values = [v for _,v in items]
    plt.figure(); plt.bar(labels, values); plt.title(title); plt.ylabel("score")
    if ylim01: plt.ylim(0.0, 1.0)
    for i, v in enumerate(values): plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def plot_confusion(conf: List[List[int]], labels: List[str], title: str, out_path: str):
    arr = np.array(conf, dtype=float)
    plt.figure(); plt.imshow(arr); plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right"); plt.yticks(np.arange(len(labels)), labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]): plt.text(j, i, f"{int(arr[i,j])}", ha="center", va="center", fontsize=7)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def make_plots(all_metrics: Dict[str,Any], out_dir: str):
    plots_dir = os.path.join(out_dir, "plots"); os.makedirs(plots_dir, exist_ok=True)
    for metric_name in ["accuracy", "macro_f1"]:
        items = []
        for split in ["val","test","train"]:
            for which in ["clean","pert"]:
                key = f"{split}_{which}"
                if key in all_metrics:
                    items.append((f"{split}-{which}", float(all_metrics[key][metric_name])))
        if items:
            plot_bar(f"{metric_name} (clean vs pert)", items, os.path.join(plots_dir, f"{metric_name}_bars.png"))
    for split in ["val","test","train"]:
        for which in ["clean","pert"]:
            key = f"{split}_{which}"
            if key in all_metrics:
                m = all_metrics[key]
                plot_confusion(m["confusion"], m["labels"], f"{split} {which} confusion",
                               os.path.join(plots_dir, f"{split}_{which}_confusion.png"))

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    hf_set_seed(args.seed)

    dfs: Dict[str, pd.DataFrame] = {}
    for s in args.splits:
        df = load_and_merge_split(
            args.clean_dir, args.pert_dir, s, args.id_col,
            args.instruction_col, args.text_col_clean, args.text_col_pert, args.label_col,
            base_seed=args.seed,
        )
        # сохраняем сравнение инструкций для анализа
        save_cols = [c for c in ["id", "instruction_clean", "instruction_pert", "instr_meta"] if c in df.columns]
        if save_cols:
            df[save_cols].to_csv(os.path.join(args.out_dir, f"instruction_perts_{s}.csv"), index=False)
        dfs[s] = df

    labels_universe = sorted(list(set(pd.concat(dfs.values())[args.label_col].astype(str).tolist())))

    model, tok = load_model_and_tokenizer(args)
    all_metrics: Dict[str,Any] = {}

    for s, df in dfs.items():
        print(f"[{s}] rows={len(df)}")

        clean_df, clean_m = evaluate_on_dataframe(df, model, tok, args, which="clean", label_universe=labels_universe)
        clean_path = os.path.join(args.out_dir, f"{s}_clean_preds.csv")
        clean_df.to_csv(clean_path, index=False)
        all_metrics[f"{s}_clean"] = clean_m
        print(f"[OK] {s} clean -> acc={clean_m['accuracy']:.4f} macro_f1={clean_m['macro_f1']:.4f}  saved={clean_path}")

        pert_df, pert_m = evaluate_on_dataframe(df, model, tok, args, which="pert", label_universe=labels_universe)
        pert_path = os.path.join(args.out_dir, f"{s}_pert_preds.csv")
        pert_df.to_csv(pert_path, index=False)
        all_metrics[f"{s}_pert"] = pert_m
        print(f"[OK] {s} pert  -> acc={pert_m['accuracy']:.4f} macro_f1={pert_m['macro_f1']:.4f}  saved={pert_path}")

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"[METRICS] saved -> {metrics_path}")

    make_plots(all_metrics, args.out_dir)
    print(f"[PLOTS] saved under {os.path.join(args.out_dir, 'plots')}")

if __name__ == "__main__":
    main()
