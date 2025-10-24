"""
prepare_data/collect_delta.py

Съём Δ_lora на заданном модуле LLaMA (обычно mlp.up_proj/down_proj) для пар (clean, pert),
когда clean и pert лежат в РАЗНЫХ CSV.

Clean CSV (по умолчанию): instruction, input, label
Pert  CSV (по умолчанию): instruction_pert, text_pert|pert_text|pert_text_llm

Склейка:
  clean_text = f"{instruction}\n\n{input}"
  pert_text  = f"{instruction_pert}\n\n{text_pert}"

Выходы (в --out_dir):
  - ddelta.npy          — ΔΔ_lora [N, D] float32
  - delta.csv           — тот же массив в CSV c колонками d0..d{D-1}
  - meta.csv            — id_raw, level, orig_split, output, pert_meta_llm (+ порядок 1:1 с ddelta)
  - y_aligned.csv       — (если --write_y_aligned) метки в порядке ddelta, колонка 'label'
  - config.json
  - (если --save_extras): dL_clean.npy, dL_pert.npy, s_base.npy, s_total.npy, residual.npy
  - (если --save_pairs_csv): pairs_debug.csv

Пример:
CUDA_VISIBLE_DEVICES=0 python prepare_data/collect_delta.py \
  --clean_csv data/train.csv \
  --pert_csv  train_pert_full/train_llm_pert_w_instr_preview.csv \
  --pert_instr_col instruction_pert \
  --out_dir   delta_train \
  --base_model llama-3.2-3b-instruct \
  --peft_path out/lora_task065_layer9__new/checkpoint-996 \
  --layer_idx 9 --module mlp.up_proj \
  --batch_size 16 --max_len 512 --load_in_4bit \
  --on_bad_lines skip --write_clean_csv --write_y_aligned \
  --save_extras --save_pairs_csv
"""
import os, json, argparse, warnings, csv
import numpy as np
import pandas as pd
import torch
from contextlib import contextmanager
from typing import Dict, List

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

def safe_read_csv(path: str, on_bad_lines: str = "skip") -> pd.DataFrame:
    return pd.read_csv(
        path,
        engine="python",
        sep=",",
        dtype=str,
        quotechar='"',
        doublequote=True,
        escapechar="\\",
        on_bad_lines=on_bad_lines
    )

def safe_write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8",
              quoting=csv.QUOTE_ALL, escapechar="\\")

def _normalize_base_model(name: str) -> str:
    alias = name.strip()
    if alias.lower() in {"llama-3.2-3b-instruct", "llama3.2-3b-instruct"}:
        return "meta-llama/Llama-3.2-3B-Instruct"
    return name

def rgetattr(obj, attr):
    for name in attr.split('.'):
        obj = getattr(obj, name)
    return obj

@contextmanager
def adapters_off(m):
    ctx = None
    if hasattr(m, "disable_adapter"):
        ctx = m.disable_adapter()
        ctx.__enter__()
    try:
        yield
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

def masked_mean(seq_tensor: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    m = attn_mask.float()
    s = (seq_tensor * m.unsqueeze(-1)).sum(dim=1)
    n = m.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
    return s / n

def build_module_path(layer_idx: int, module: str) -> str:
    return f"model.model.layers.{layer_idx}.{module}"

def load_seq_model_and_tokenizer(base_model: str, peft_path: str, load_in_4bit: bool, device: str):
    base_model = _normalize_base_model(base_model)

    use_bfloat16 = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    quant_cfg = None
    if load_in_4bit and HAS_BNB:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            bnb_4bit_quant_type="nf4",
        )

    kwargs = dict(trust_remote_code=True, torch_dtype=torch_dtype)
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg

    _ = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(base_model, **kwargs)
    except TypeError as e:
        if quant_cfg is not None and "quantization_config" in str(e):
            kwargs_fb = dict(trust_remote_code=True, torch_dtype=torch_dtype, load_in_4bit=True)
            model = AutoModelForSequenceClassification.from_pretrained(base_model, **kwargs_fb)
        else:
            raise

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, peft_path)
    model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        model.to(device)

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id

    return model, tok

def capture_module_pooled(model, module_path: str, enc_inputs: Dict[str, torch.Tensor], with_adapters: bool) -> torch.Tensor:
    buf = {}
    def hook(_m, _inp, out):
        buf["h"] = out.detach()
    try:
        handle = rgetattr(model, module_path).register_forward_hook(hook)
    except AttributeError as e:
        raise RuntimeError(f"Не найден модуль '{module_path}'. Проверь --layer_idx/--module. Исходная ошибка: {e}")
    if with_adapters:
        _ = model(**enc_inputs)
    else:
        with adapters_off(model):
            _ = model(**enc_inputs)
    handle.remove()
    return masked_mean(buf["h"], enc_inputs["attention_mask"])

def _first_present(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def make_text_clean(row, instr_col, input_col, tmpl="{instruction}\n\n{input}"):
    instr = str(row[instr_col]) if instr_col else ""
    inp   = str(row[input_col]) if input_col else ""
    return tmpl.format(instruction=instr, input=inp).strip()

def make_text_pert(row, instrp_col, textp_col, tmpl="{instruction}\n\n{input}"):
    instr = str(row[instrp_col]) if instrp_col else ""
    inp   = str(row[textp_col]) if textp_col else ""
    return tmpl.format(instruction=instr, input=inp).strip()

@torch.no_grad()
def process_csv(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model, tok = load_seq_model_and_tokenizer(args.base_model, args.peft_path, args.load_in_4bit, args.device)
    module_path = build_module_path(args.layer_idx, args.module)

    df_c = safe_read_csv(args.clean_csv, on_bad_lines=args.on_bad_lines)
    if args.write_clean_csv:
        safe_write_csv(df_c, os.path.splitext(args.clean_csv)[0] + "_cleaned.csv")

    instr_col = args.clean_instr_col or _first_present(df_c.columns, ["instruction", "instructions", "prompt"])
    input_col = args.clean_input_col or _first_present(df_c.columns, ["input", "inputs", "text"])
    label_col = args.clean_label_col or _first_present(df_c.columns, ["label", "labels", "target", "targets", "output"])
    if input_col is None or label_col is None:
        raise ValueError("В clean CSV не найдены обязательные столбцы: input и label (можно указать явно флагами).")

    df_p = safe_read_csv(args.pert_csv, on_bad_lines=args.on_bad_lines)
    if args.write_clean_csv:
        safe_write_csv(df_p, os.path.splitext(args.pert_csv)[0] + "_cleaned.csv")

    instrp_col = args.pert_instr_col or _first_present(df_p.columns, ["instruction_pert", "instr_pert"])
    textp_col  = args.pert_text_col  or _first_present(df_p.columns, ["text_pert", "pert_text", "pert_text_llm"])
    if textp_col is None:
        raise ValueError("В pert CSV не найден столбец с пертурбированным текстом (например, text_pert / pert_text_llm).")

    key_clean = _first_present(df_c.columns, ["id_raw", "id"])
    key_pert  = _first_present(df_p.columns, ["id_raw", "id"])
    if key_clean and key_pert:
        df = (df_c.rename(columns={key_clean: "id_raw"})
                 .merge(df_p.rename(columns={key_pert: "id_raw"}), on="id_raw", how="inner", suffixes=("", "_p")))
        if len(df) == 0:
            raise ValueError("После merge по ключу не осталось строк. Проверьте id/id_raw.")
    else:
        n = min(len(df_c), len(df_p))
        if len(df_c) != len(df_p):
            warnings.warn(f"[WARN] Длины clean ({len(df_c)}) и pert ({len(df_p)}) различаются. Беру первые {n} попарно.")
        df = pd.concat([df_c.iloc[:n].reset_index(drop=True),
                        df_p.iloc[:n].reset_index(drop=True)], axis=1)
        if "id_raw" not in df.columns:
            df["id_raw"] = np.arange(n)

    df["clean_text"] = df.apply(lambda r: make_text_clean(r, instr_col, input_col, args.concat_fmt), axis=1)
    df["pert_text"]  = df.apply(lambda r: make_text_pert(r, instrp_col, textp_col, args.concat_fmt), axis=1)

    if "level" not in df.columns:
        df["level"] = "H"
    if "orig_split" not in df.columns:
        guessed = args.orig_split or (
            "val" if "val" in args.clean_csv.lower()
            else "test" if "test" in args.clean_csv.lower()
            else "train" if "train" in args.clean_csv.lower()
            else "unknown"
        )
        df["orig_split"] = guessed
    df["output"] = df[label_col].astype(str)
    if "pert_meta_llm" not in df.columns:
        df["pert_meta_llm"] = df.get("aug_meta", pd.Series(["instr+text_pert"] * len(df)))

    df = df[~df["pert_text"].isna() & (df["pert_text"].astype(str).str.len() > 0)].reset_index(drop=True)
    N = len(df)
    if N == 0:
        raise ValueError("После фильтрации пустых pert_text не осталось строк.")
    print(f"[INFO] rows={N}  module={module_path}")

    ddelta_list: List[torch.Tensor] = []
    dL_clean_list: List[torch.Tensor] = []
    dL_pert_list:  List[torch.Tensor] = []
    s_base_list:   List[torch.Tensor] = []
    s_total_list:  List[torch.Tensor] = []
    residuals:     List[float]        = []

    bs = args.batch_size
    for i in range(0, N, bs):
        sub = df.iloc[i:i+bs]

        enc_clean = tok(sub["clean_text"].astype(str).tolist(),
                        return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        enc_pert  = tok(sub["pert_text"].astype(str).tolist(),
                        return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
        enc_clean = {k:v.to(args.device) for k,v in enc_clean.items()}
        enc_pert  = {k:v.to(args.device) for k,v in enc_pert.items()}

        h0_c = capture_module_pooled(model, module_path, enc_clean, with_adapters=False)
        hP_c = capture_module_pooled(model, module_path, enc_clean, with_adapters=True)
        h0_p = capture_module_pooled(model, module_path, enc_pert,  with_adapters=False)
        hP_p = capture_module_pooled(model, module_path, enc_pert,  with_adapters=True)

        dL_c = hP_c - h0_c
        dL_p = hP_p - h0_p
        ddL  = dL_p - dL_c

        s_base = h0_p - h0_c
        s_tot  = hP_p - hP_c
        resid  = torch.linalg.norm(s_tot - (s_base + ddL), dim=1)

        ddelta_list.append(ddL.cpu())
        if args.save_extras:
            dL_clean_list.append(dL_c.cpu())
            dL_pert_list.append(dL_p.cpu())
            s_base_list.append(s_base.cpu())
            s_total_list.append(s_tot.cpu())
            residuals += resid.cpu().tolist()

        if ((i//bs) % 20) == 0:
            print(f"[BATCH] {i}/{N}  dim={ddL.shape[-1]}")

    ddelta = torch.cat(ddelta_list, dim=0).to(torch.float32).numpy()
    np.save(os.path.join(out_dir, "ddelta.npy"), ddelta)
    print(f"[OK] ddelta.npy saved: shape={ddelta.shape} dtype=float32")

    cols = [f"d{j}" for j in range(ddelta.shape[1])]
    safe_write_csv(pd.DataFrame(ddelta, columns=cols), os.path.join(out_dir, "delta.csv"))
    print(f"[OK] delta.csv saved in {out_dir}  shape={ddelta.shape}")

    if args.save_extras:
        def _save_tensor(name, lst):
            arr = torch.cat(lst, dim=0).to(torch.float32).numpy()
            np.save(os.path.join(out_dir, f"{name}.npy"), arr)
            print(f"[OK] {name}.npy saved: shape={arr.shape}")
        _save_tensor("dL_clean", dL_clean_list)
        _save_tensor("dL_pert",  dL_pert_list)
        _save_tensor("s_base",   s_base_list)
        _save_tensor("s_total",  s_total_list)
        np.save(os.path.join(out_dir, "residual.npy"), np.array(residuals, dtype=np.float32))

    meta_cols = ["id_raw","level","orig_split","output","pert_meta_llm"]
    safe_write_csv(df[meta_cols], os.path.join(out_dir, "meta.csv"))

    if args.write_y_aligned:
        ya = pd.DataFrame({"label": df["output"].astype(str).values})
        safe_write_csv(ya, os.path.join(out_dir, "y_aligned.csv"))

    if args.save_pairs_csv:
        cols2 = ["id_raw","clean_text","pert_text","output","level","orig_split","pert_meta_llm"]
        df_to_save = df[cols2] if all(c in df.columns for c in cols2) else df
        safe_write_csv(df_to_save, os.path.join(out_dir, "pairs_debug.csv"))
        print("[OK] pairs_debug.csv saved")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "clean_csv": args.clean_csv,
            "pert_csv": args.pert_csv,
            "out_dir": out_dir,
            "base_model": _normalize_base_model(args.base_model),
            "peft_path": args.peft_path,
            "layer_idx": args.layer_idx,
            "module": args.module,
            "max_len": args.max_len,
            "batch_size": args.batch_size,
            "load_in_4bit": bool(args.load_in_4bit),
            "save_extras": bool(args.save_extras),
            "concat_fmt": args.concat_fmt,
            "on_bad_lines": args.on_bad_lines,
            "write_clean_csv": bool(args.write_clean_csv),
            "write_y_aligned": bool(args.write_y_aligned),
        }, f, ensure_ascii=False, indent=2)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_csv", required=True, type=str)
    ap.add_argument("--pert_csv",  required=True, type=str)
    ap.add_argument("--out_dir",   required=True, type=str)

    ap.add_argument("--clean_instr_col", type=str, default=None)
    ap.add_argument("--clean_input_col", type=str, default=None)
    ap.add_argument("--clean_label_col", type=str, default=None)
    ap.add_argument("--pert_instr_col",  type=str, default=None)
    ap.add_argument("--pert_text_col",   type=str, default=None)

    ap.add_argument("--concat_fmt", type=str, default="{instruction}\n\n{input}")
    ap.add_argument("--orig_split", type=str, default=None)

    ap.add_argument("--base_model", required=True, type=str)
    ap.add_argument("--peft_path",  required=True, type=str)
    ap.add_argument("--layer_idx",  type=int, default=9)
    ap.add_argument("--module",     type=str, default="mlp.up_proj")

    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--max_len",     type=int, default=512)
    ap.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--save_extras",  action="store_true")
    ap.add_argument("--save_pairs_csv", action="store_true")

    ap.add_argument("--on_bad_lines", choices=["skip","warn","error"], default="skip",
                    help="Поведение pandas при битых строках в CSV")
    ap.add_argument("--write_clean_csv", action="store_true",
                    help="Сохранить *_cleaned.csv версии входных CSV рядом")
    ap.add_argument("--write_y_aligned", action="store_true",
                    help="Записать y_aligned.csv (колонка 'label') в порядке ddelta")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_csv(args)
