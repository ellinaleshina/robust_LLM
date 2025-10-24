# launch example
#  CUDA_VISIBLE_DEVICES=1 python /prepare_data/llm_pert.py
# --csv_dir /data   --out_dir /llm_pert_train
# --text_col input --id_col id_raw --label_col output
# --caps 5288,0,0   --levels H   --llm_repo_id bartowski/Ministral-8B-Instruct-2410-GGUF
# --llm_filename Ministral-8B-Instruct-2410-Q6_K.gguf   --n_ctx 4096
# --n_gpu_layers -1   --llm_temperature 0.7   --seed 4

import os, json, argparse, random, sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from llama_cpp import Llama


def parse_caps(s: str) -> Dict[str, int]:
    caps = {"train": 0, "val": 0, "test": 0}
    if "=" in s:
        for part in s.split(","):
            if not part.strip():
                continue
            k, v = part.split("=")
            caps[k.strip()] = int(v)
    else:
        a, b, c = [int(x) for x in s.split(",")]
        caps["train"], caps["val"], caps["test"] = a, b, c
    return caps

def load_split(csv_dir: Path, name: str) -> Optional[pd.DataFrame]:
    p = csv_dir / f"{'validation' if name=='val' and not (csv_dir/'val.csv').exists() else name}.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, on_bad_lines="skip", encoding_errors="replace")
    except Exception:
        df = pd.read_csv(p, on_bad_lines="skip", encoding_errors="replace", engine="python")
    return df

def choose_rows(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    if k <= 0:
        return df.iloc[[]]
    k = min(k, len(df))
    return df.sample(n=k, random_state=seed)

def severity_from_level(level: str) -> str:
    level = level.upper()
    if level == "L": return "very mild"
    if level == "H": return "aggressive but still readable"
    return "moderate"

def _ensure_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s

PROMPT_SYS = (
    "You are a careful text editor. You make ONLY tiny, non-semantic, character-level typos "
    "(missing letters, swapped neighbors, doubled letters, homoglyphs, zero-width joiners, thin spaces) "
    "BUT ONLY IN LINES THAT START WITH 'Sentence '. "
    "You MUST NOT modify any line that starts with 'Option '. "
    "Keep labels like 'Sentence 1:' and 'Option 1:' unchanged; keep line count and order identical. "
    "Do NOT add or remove lines. Do NOT add new options. "
    "Output ONLY the modified text with the SAME structure. No explanations, no quotes, no lists."
)

FS_EXAMPLES = [
    {
        "severity": "very mild",
        "before": (
            "Sentence 1: Alice was extremely shy and never dated anyone.\n"
            " Sentence 3: He approached her with a confident swag\n"
            " Sentence 4:  But Alice recoiled afraid by his outburst\n"
            " Sentence 5:  Xavier changed tactic and tamed the shy Alice overtime\n"
            " Option 1: She really just needed to get fresh air.\n"
            " Option 2: She really just needed to sit down."
        ),
        "after": (
            "Sentence 1: Alice was extemely shy and never dated anyone.\n"
            " Sentence 3: He aproached her with a confident swag\n"
            " Sentence 4:  But Alice recoiled, afriad by his outburst\n"
            " Sentence 5:  Xavier changed tactic and tamed the shy Alice over­time\n"
            " Option 1: She really just needed to get fresh air.\n"
            " Option 2: She really just needed to sit down."
        ),
    },
    {
        "severity": "moderate",
        "before": (
            "Sentence 1: Jordan felt anxious before the presentation.\n"
            " Sentence 3: He checked his notes once more\n"
            " Sentence 4:  The projector flickered for a second\n"
            " Sentence 5:  He took a breath and began speaking calmly\n"
            " Option 1: He forgot the entire speech.\n"
            " Option 2: He spoke clearly and stayed on track."
        ),
        "after": (
            "Sentence 1: Jordan felt an‍xious before the presentatiоn.\n"
            " Sentence 3: He chekced his notes once more\n"
            " Sentence 4:  The projector f‍lickered for a secоnd\n"
            " Sentence 5:  He took a breath and began spea­king calmy\n"
            " Option 1: He forgot the entire speech.\n"
            " Option 2: He spoke clearly and stayed on track."
        ),
    },
    {
        "severity": "aggressive but still readable",
        "before": (
            "Sentence 1: The hallway lights were buzzing faintly.\n"
            " Sentence 3: She tightened her grip on the notebook\n"
            " Sentence 4:  A door clicked somewhere down the corridor\n"
            " Sentence 5:  She hurried along trying not to make noise\n"
            " Option 1: She relaxed and stopped worrying.\n"
            " Option 2: She kept moving carefully, alert to sounds."
        ),
        "after": (
            "Sentence 1: The hall­way ligths were buzzng faintly.\n"
            " Sentence 3: She tightend her grip on the notebоok\n"
            " Sentence 4:  A door clikced some­where down the cor­ridor\n"
            " Sentence 5:  She hurried along tryin‍g not to make noize\n"
            " Option 1: She relaxed and stopped worrying.\n"
            " Option 2: She kept moving carefully, alert to sounds."
        ),
    },
]

def _few_shot_messages(k: int) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    k = max(0, min(k, len(FS_EXAMPLES)))
    for i in range(k):
        ex = FS_EXAMPLES[i]
        msgs.append({
            "role": "user",
            "content": (
                f"Severity: {ex['severity']}.\n\n"
                "Edit the following text in-place with tiny non-semantic typos, preserving the exact line structure:\n\n"
                f"{ex['before']}"
            )
        })
        msgs.append({"role": "assistant", "content": ex["after"]})
    return msgs

def _build_messages(inp: str, level: str, few_shot_k: int) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": PROMPT_SYS}]
    msgs += _few_shot_messages(few_shot_k)
    msgs.append({
        "role": "user",
        "content": (
            f"Severity: {severity_from_level(level)}.\n\n"
            "Edit the following text in-place with tiny non-semantic typos, preserving the exact line structure:\n\n"
            f"{inp}"
        ),
    })
    return msgs

def _strip_wrappers(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if lines and lines[0].strip().lower() in {"text", "markdown"}:
            lines = lines[1:]
        s = "\n".join(lines)
    s = s.strip().strip('"').strip("'")
    return s.strip()


def _split_head_body(line: str) -> Tuple[str, str]:
    if ":" in line:
        head, body = line.split(":", 1)
        return head + ":", body
    return line, ""

def _norm_label_prefix(line: str) -> str:
    s = line.strip()
    return s.split(":", 1)[0] + ":" if ":" in s else s

def _is_option_prefix(pref: str) -> bool:
    return pref.lower().startswith("option ") and pref.endswith(":")

def _is_sentence_prefix(pref: str) -> bool:
    return pref.lower().startswith("sentence ") and pref.endswith(":")

def _enforce_structure(src: str, gen: str) -> str:
    src_lines = src.splitlines()
    gen_lines = gen.splitlines()

    gen_body_by_pref: Dict[str, str] = {}
    for gl in gen_lines:
        pref = _norm_label_prefix(gl)
        if pref not in gen_body_by_pref:
            _, gbody = _split_head_body(gl)
            gen_body_by_pref[pref] = gbody

    out_lines: List[str] = []
    for sl in src_lines:
        src_head_exact, _ = _split_head_body(sl)
        pref = _norm_label_prefix(sl)

        if _is_option_prefix(pref):
            out_lines.append(sl)
            continue

        if _is_sentence_prefix(pref):
            gbody = gen_body_by_pref.get(pref, None)
            if gbody is None or not gbody.strip():
                out_lines.append(sl)
            else:
                out_lines.append(src_head_exact + gbody)
            continue

        out_lines.append(sl)

    return "\n".join(out_lines)

def _assert_options_frozen(src: str, out: str):
    s_lines = src.splitlines()
    o_lines = out.splitlines()
    if len(s_lines) != len(o_lines):
        raise ValueError(f"Структура изменилась: число строк не совпало (src={len(s_lines)} vs out={len(o_lines)}).")
    for i, (s_line, o_line) in enumerate(zip(s_lines, o_lines)):
        sp = _norm_label_prefix(s_line)
        if _is_option_prefix(sp) and s_line != o_line:
            raise ValueError(f"Option line was modified at line {i+1}.")

def _source_is_valid(src: str) -> Tuple[bool, str]:
    if not src or not src.strip():
        return False, "empty_text"
    lines = src.splitlines()
    has_sent = any(_is_sentence_prefix(_norm_label_prefix(l)) for l in lines)
    has_opt  = any(_is_option_prefix(_norm_label_prefix(l))   for l in lines)
    if not has_sent:
        return False, "no_sentence_lines"
    if not has_opt:
        return False, "no_option_lines"
    return True, ""


def llm_init(repo_id: str, filename: str, n_ctx: int, n_gpu_layers: int) -> "Llama":
    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
    )

def _call_chat_or_template(llm: "Llama", messages: List[Dict[str, str]],
                           temperature: float, max_tokens: int, stop: List[str]) -> str:
    if hasattr(llm, "create_chat_completion"):
        out = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        return out["choices"][0]["message"]["content"]

    if hasattr(llm, "apply_chat_template"):
        prompt = llm.apply_chat_template(messages, add_generation_prompt=True)
    else:
        prompt = f"[SYSTEM]\n{messages[0]['content']}\n"
        for m in messages[1:]:
            role = m["role"].upper()
            prompt += f"[{role}]\n{m['content']}\n"

    out = llm(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
    )
    return out["choices"][0]["text"]

def llm_perturb(llm: "Llama", text: str, level: str, temperature: float,
                few_shot_k: int, max_tokens: int = 0) -> str:
    if max_tokens <= 0:
        approx = max(64, len(text) // 3 + 64)
        max_tokens = min(2048, approx)

    messages = _build_messages(text, level, few_shot_k)
    stop = [
        "\nOption 3", "Option 3:", "\nOption 4", "Option 4:",
        "\nOption 5", "Option 5:", "\nOption 6", "Option 6:",
        "\nOption 7", "Option 7:", "\nOption 8", "Option 8:",
        "\nOption 9", "Option 9:", "\nChoices:", "\nAnswers:",
        "\nAnswer:", "\nA)", "\nB)", "\nC)", "\nD)",
        "\nSentence 2", "Sentence 2:"
    ]

    raw = _call_chat_or_template(
        llm, messages, temperature=temperature, max_tokens=max_tokens, stop=stop
    )
    s = _strip_wrappers(raw)
    s = _enforce_structure(text, s)
    _assert_options_frozen(text, s)
    return s if s.strip() else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--text_col", required=True)
    ap.add_argument("--id_col", required=True)
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--caps", default="200,20,20")
    ap.add_argument("--levels", nargs="+", default=["M"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_repo_id", required=True)
    ap.add_argument("--llm_filename", required=True)
    ap.add_argument("--n_ctx", type=int, default=4096)
    ap.add_argument("--n_gpu_layers", type=int, default=-1)
    ap.add_argument("--llm_temperature", type=float, default=0.7)
    ap.add_argument("--on_broken", choices=["keep_orig", "skip"], default="keep_orig")
    ap.add_argument("--log_every", type=int, default=20)
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    caps = parse_caps(args.caps)
    random.seed(args.seed)

    llm = llm_init(args.llm_repo_id, args.llm_filename, args.n_ctx, args.n_gpu_layers)

    for split in ["train", "val", "test"]:
        df = load_split(csv_dir, split)
        if df is None:
            print(f"[skip] {split}: файла нет")
            continue

        for col in [args.text_col, args.id_col, args.label_col]:
            if col not in df.columns:
                raise ValueError(f"{split}.csv: нет колонки '{col}'")

        df = df.copy()
        df = df.dropna(subset=[args.text_col, args.id_col, args.label_col])
        df[args.text_col] = df[args.text_col].map(_ensure_text)

        sub = choose_rows(df, caps[split], args.seed)
        if len(sub) == 0:
            print(f"[{split}] квота 0 — пропускаю")
            continue

        broken_rows: List[Dict[str, str]] = []
        rows: List[Dict[str, str]] = []

        for i, (_, row) in enumerate(sub.iterrows(), start=1):
            src = _ensure_text(row[args.text_col])
            ok, why = _source_is_valid(src)

            def make_record(pert_text: str, meta_extra: Dict) -> Dict[str, str]:
                meta = {
                    "op": "llm_typos",
                    "model": f"{args.llm_repo_id}:{args.llm_filename}",
                    "temperature": args.llm_temperature,
                    "level": meta_extra.get("level", "M"),
                    "seed": args.seed,
                    "few_shot_k": args.few_shot_k,
                }
                meta.update(meta_extra)
                return {
                    "orig_split": split,
                    args.id_col: row[args.id_col],
                    args.text_col: src,
                    args.label_col: row[args.label_col],
                    "level": meta_extra.get("level", "M"),
                    "pert_text_llm": pert_text,
                    "pert_meta_llm": json.dumps(meta, ensure_ascii=False),
                }

            handled = False
            for lvl in args.levels:
                lvlU = lvl.upper()
                if not ok:
                    broken_rows.append({
                        "orig_split": split,
                        args.id_col: row[args.id_col],
                        "reason": why,
                        "text": src[:1000],
                    })
                    if args.on_broken == "skip":
                        handled = True
                        continue
                    rows.append(make_record(src, {"level": lvlU, "source_broken": True, "reason": why}))
                    handled = True
                    continue

                try:
                    pert = llm_perturb(
                        llm, src, lvlU,
                        temperature=args.llm_temperature,
                        few_shot_k=args.few_shot_k
                    )
                    rows.append(make_record(pert, {"level": lvlU}))
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    rows.append(make_record(src, {"level": lvlU, "error": err, "fallback": "kept_original"}))
                    broken_rows.append({
                        "orig_split": split,
                        args.id_col: row[args.id_col],
                        "reason": f"runtime_error:{type(e).__name__}",
                        "text": src[:1000],
                        "error": err,
                    })
                    continue

            if (i % args.log_every) == 0:
                print(f"[{split}] обработано {i}/{len(sub)} строк (накоплено {len(rows)} записей)", flush=True)

        if broken_rows:
            bad_path = out_dir / f"{split}_bad.csv"
            pd.DataFrame(broken_rows).to_csv(bad_path, index=False)
            print(f"[warn] {split}: проблемные примеры: {len(broken_rows)} → {bad_path}")

        out_df = pd.DataFrame(rows)
        out_path = out_dir / f"{split}_llm.csv"
        out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
