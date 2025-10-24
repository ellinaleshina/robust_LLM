# example of my launch
# for f in 0.002 0.005 0.01 0.02 0.05; do
#   CUDA_VISIBLE_DEVICES=0 python eval_lora_shiftaware.py \
#     --pairs_csv pert/test_pert.csv \
#     --calib_csv pert/train_pert.csv \
#     --calib_limit 2048 \
#     --base_model llama-3.2-3b-instruct \
#     --peft_path out/lora_task065_layer9_new/checkpoint-996 \
#     --layer_idx 9 --module mlp.up_proj \
#     --pca_r 2 \
#     --alpha_list 0.05 0.08 0.10 0.20 \
#     --mask_frac $f \
#     --mask_modes none ucb tucb \
#     --batch_size 16 --max_len 512
# done

# columns in test_pert.csv
# id,instruction,input,split,text_clean,text_pert,label,aug_meta,level,instruction_pert,instr_meta
# pca/eval_lora_shiftaware.py

import json, argparse, random
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from peft import PeftModel
    PEFT_OK = True
except Exception:
    PEFT_OK = False


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def _resolve_by_path(root: nn.Module, dotted: str) -> nn.Module:
    node = root
    for p in dotted.split("."):
        if p.isdigit():
            node = node[int(p)]
        else:
            node = getattr(node, p)
    return node

def _first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _merge_keys(cli_keys: Optional[List[str]], fallbacks: List[str]) -> List[str]:
    return list(dict.fromkeys((cli_keys or []) + fallbacks))

def text_from_row(row: dict, cols_instr: List[str], cols_input: List[str],
                  cols_clean_text: List[str], cols_pert_text: List[str],
                  cols_label: List[str], cols_instr_pert: List[str]) -> Tuple[str, str, str]:
    instr = _first(row, cols_instr, "")
    inp   = _first(row, cols_input, "")
    clean_text = f"{instr}\n\n{inp}" if (instr and inp) else (instr or inp or _first(row, cols_clean_text, ""))

    pert = _first(row, cols_pert_text, "")
    instr_pert = _first(row, cols_instr_pert, None)
    pert_text = f"{instr_pert}\n\n{pert}" if instr_pert is not None else (pert if pert else clean_text)

    label_str = _first(row, cols_label, None)
    if label_str is None:
        raise RuntimeError("label column is missing")
    return clean_text, pert_text, str(label_str)


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, limit: Optional[int],
                 cols_instr: List[str], cols_input: List[str],
                 cols_clean_text: List[str], cols_pert_text: List[str],
                 cols_label: List[str], cols_instr_pert: List[str]):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.rows = df.to_dict(orient="records")
        if limit is not None:
            self.rows = self.rows[:limit]
        self.cols = (cols_instr, cols_input, cols_clean_text, cols_pert_text, cols_label, cols_instr_pert)

        labels = []
        for r in self.rows:
            _, _, y = text_from_row(r, *self.cols); labels.append(y)
        uniq, seen = [], set()
        for y in labels:
            if y not in seen:
                uniq.append(y); seen.add(y)
        self.label2id = {y:i for i,y in enumerate(uniq)}
        self.id2label = {i:y for y,i in self.label2id.items()}

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        clean_text, pert_text, y = text_from_row(r, *self.cols)
        yid = self.label2id[y]
        meta = {
            "level": r.get("level", None),
            "pert_meta": r.get("pert_flags", r.get("pert_meta_llm", None)),
            "id_raw": r.get("id_raw", r.get("id", idx)),
        }
        return {"clean": clean_text, "pert": pert_text, "label": yid, "meta": meta}


def collect_pooled_activations(model, tok, texts, hook_path, batch_size=16, max_len=512, device="cuda", pool="mean"):
    module = _resolve_by_path(model, hook_path)
    storage = []

    def _hook_fn(m, i, o):
        if torch.is_tensor(o):
            storage.append(o.detach())
        return o

    handle = module.register_forward_hook(_hook_fn)
    pooled = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding="longest", truncation=True, max_length=max_len, return_tensors="pt")
            enc = to_device(enc, device)
            storage.clear()
            out = model(**enc)
            _ = out.logits
            if not storage:
                handle.remove()
                raise RuntimeError("forward hook captured nothing")
            act = storage[-1]
            if act.ndim == 2 or pool == "none":
                pooled_batch = act
            else:
                if pool == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).type_as(act)
                    sum_act = (act * mask).sum(dim=1)
                    lengths = mask.sum(dim=1).clamp_min(1.0)
                    pooled_batch = sum_act / lengths
                elif pool == "first":
                    pooled_batch = act[:, 0, :]
                else:
                    raise ValueError("pool must be in {'mean','first','none'}")
            pooled.append(pooled_batch)
    handle.remove()
    return torch.cat(pooled, dim=0)


class ShiftAwarePCA:
    def __init__(self, pca_r: int = 2, alpha: float = 0.1, mask: Optional[torch.Tensor]=None):
        self.mu = None
        self.U = None
        self.pca_r = int(pca_r)
        self.alpha = float(alpha)
        self.mask = mask

    @torch.no_grad()
    def fit(self, deltas: torch.Tensor):
        device = deltas.device
        orig_dtype = deltas.dtype
        mu32 = deltas.float().mean(dim=0)
        Xc = (deltas.float() - mu32)
        U_feat = None
        if self.pca_r > 0:
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            U_feat = Vh[:self.pca_r, :].T.contiguous()
        self.mu = mu32.to(device=device, dtype=orig_dtype)
        self.U  = U_feat.to(device=device, dtype=orig_dtype) if U_feat is not None else None
        if self.mask is not None:
            self.mask = self.mask.to(device=device, dtype=orig_dtype)

    @torch.no_grad()
    def transform(self, h: torch.Tensor) -> torch.Tensor:
        if self.mu is None:
            return h
        if self.mu.dtype != h.dtype: self.mu = self.mu.to(h.dtype)
        if self.U is not None and self.U.dtype != h.dtype: self.U = self.U.to(h.dtype)
        if self.mask is not None and self.mask.dtype != h.dtype: self.mask = self.mask.to(h.dtype)
        x = h - self.mu
        if self.U is None or self.alpha <= 0:
            return x
        Ut_x = torch.matmul(x, self.U)
        proj = torch.matmul(Ut_x, self.U.T)
        x_supp = x - self.alpha * proj
        if self.mask is None:
            return x_supp
        m = self.mask
        while m.ndim < x_supp.ndim:
            m = m.unsqueeze(0)
        return (1 - m) * x + m * x_supp


class ModulePatch:
    def __init__(self, root: nn.Module, module_path: str, suppressor: ShiftAwarePCA):
        self.root = root
        self.path = module_path
        self.suppressor = suppressor
        self.hook = None
        self._module = _resolve_by_path(root, module_path)

    def enable(self):
        if self.hook is not None:
            return
        def _fn(module, inp, out):
            if not torch.is_tensor(out):
                return out
            return self.suppressor.transform(out)
        self.hook = self._module.register_forward_hook(lambda m, i, o: _fn(m, i, o))

    def disable(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


@torch.no_grad()
def run_logits(model, tok, texts: List[str], batch_size=16, max_len=512, device="cuda"):
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding="longest", truncation=True, max_length=max_len, return_tensors="pt")
        enc = to_device(enc, device)
        out = model(**enc)
        outs.append(out.logits.detach())
    return torch.cat(outs, dim=0)

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def build_coord_mask_from_deltas(deltas: torch.Tensor, mode: str = "tucb", frac: float = 0.01) -> torch.Tensor:
    N, D = deltas.shape
    mu = deltas.mean(dim=0)
    if mode.lower() == "ucb":
        score = mu.abs()
    elif mode.lower() == "tucb":
        sd = deltas.std(dim=0, unbiased=True).clamp_min(1e-12)
        score = (mu + 1.96 * sd / np.sqrt(max(N,1))).abs()
    else:
        raise ValueError("mode must be in {'ucb','tucb'}")
    k = max(1, int(round(frac * D)))
    topk = torch.topk(score, k=k, largest=True).indices
    m = torch.zeros(D, device=deltas.device, dtype=torch.float32)
    m[topk] = 1.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--calib_csv", type=str, default=None)
    ap.add_argument("--calib_limit", type=int, default=None)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--peft_path", type=str, default=None)
    ap.add_argument("--layer_idx", type=int, default=9)
    ap.add_argument("--module", type=str, default="mlp.up_proj")
    ap.add_argument("--hook_path", type=str, default=None)
    ap.add_argument("--pool", type=str, default="mean", choices=["mean","first","none"])
    ap.add_argument("--calib_size", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pca_r", type=int, default=2)
    ap.add_argument("--alpha_list", type=float, nargs="+", default=[0.05, 0.08, 0.10, 0.20])
    ap.add_argument("--mask_frac", type=float, default=0.01)
    ap.add_argument("--mask_compare_alpha", type=float, default=0.08)
    ap.add_argument("--mask_modes", type=str, nargs="+", default=["none","ucb","tucb"])
    ap.add_argument("--save_prefix", type=str, default=None)

    ap.add_argument("--cols_instr", nargs="+", default=[])
    ap.add_argument("--cols_input", nargs="+", default=[])
    ap.add_argument("--cols_clean_text", nargs="+", default=[])
    ap.add_argument("--cols_pert_text", nargs="+", default=[])
    ap.add_argument("--cols_label", nargs="+", default=[])
    ap.add_argument("--cols_instr_pert", nargs="+", default=[])

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fall_instr = ["instruction","instruction_clean","instr","system_instruction"]
    fall_input = ["input","text_clean","clean_text","prompt"]
    fall_clean = ["clean_text","text_clean"]
    fall_pert  = ["pert_text","text_pert","perturbation","pert"]
    fall_label = ["output","label","y","target"]
    fall_instrp= ["instruction_pert"]

    cols_instr = _merge_keys(args.cols_instr, fall_instr)
    cols_input = _merge_keys(args.cols_input, fall_input)
    cols_clean = _merge_keys(args.cols_clean_text, fall_clean)
    cols_pert  = _merge_keys(args.cols_pert_text, fall_pert)
    cols_label = _merge_keys(args.cols_label, fall_label)
    cols_instrp= _merge_keys(args.cols_instr_pert, fall_instrp)

    ds_eval = PairsDataset(args.pairs_csv, args.limit,
                           cols_instr, cols_input, cols_clean, cols_pert, cols_label, cols_instrp)
    id2label = ds_eval.id2label
    num_labels = len(id2label)

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    if args.peft_path:
        if not PEFT_OK:
            raise RuntimeError("install peft")
        model = PeftModel.from_pretrained(model, args.peft_path)
        model = model.merge_and_unload()

    model.config.pad_token_id = tok.pad_token_id
    model.to(device); model.eval()

    if args.hook_path:
        hook_path = args.hook_path
        _ = _resolve_by_path(model, hook_path)
    else:
        llama_path = f"model.layers.{args.layer_idx}.{args.module}"
        candidates = [
            llama_path,
            f"model.model.layers.{args.layer_idx}.{args.module}",
            f"roberta.encoder.layer.{args.layer_idx}.{args.module}",
        ]
        hook_path = None
        for p in candidates:
            try:
                _ = _resolve_by_path(model, p)
                hook_path = p; break
            except Exception:
                continue
        if hook_path is None:
            raise RuntimeError("module path not found")

    clean_texts_eval, pert_texts_eval, ys_eval = [], [], []
    for i in range(len(ds_eval)):
        it = ds_eval[i]
        clean_texts_eval.append(it["clean"])
        pert_texts_eval.append(it["pert"])
        ys_eval.append(it["label"])
    y = torch.tensor(ys_eval, dtype=torch.long, device=device)

    with torch.no_grad():
        logits_clean = run_logits(model, tok, clean_texts_eval, batch_size=args.batch_size, max_len=args.max_len, device=device)
        logits_pert  = run_logits(model, tok,  pert_texts_eval,  batch_size=args.batch_size, max_len=args.max_len, device=device)
    acc_clean_base = accuracy_from_logits(logits_clean.to(device), y)
    acc_pert_base  = accuracy_from_logits(logits_pert.to(device),  y)
    print(f"BASE clean={acc_clean_base:.4f} pert={acc_pert_base:.4f}")

    calib_from_eval_fallback = False
    clean_texts_calib, pert_texts_calib = [], []
    if args.calib_csv and Path(args.calib_csv).exists():
        ds_calib = PairsDataset(args.calib_csv, args.calib_limit,
                                cols_instr, cols_input, cols_clean, cols_pert, cols_label, cols_instrp)
        calib_N = len(ds_calib)
        for i in range(calib_N):
            it = ds_calib[i]
            clean_texts_calib.append(it["clean"])
            pert_texts_calib.append(it["pert"])
        print(f"CALIB csv={args.calib_csv} N={calib_N}")
    else:
        calib_from_eval_fallback = True
        calib_n = min(args.calib_size, len(ds_eval))
        clean_texts_calib = clean_texts_eval[:calib_n]
        pert_texts_calib  = pert_texts_eval[:calib_n]
        print(f"CALIB fallback first {calib_n}")

    Hc = collect_pooled_activations(model, tok, clean_texts_calib, hook_path,
                                    batch_size=args.batch_size, max_len=args.max_len, device=device, pool=args.pool)
    Hp = collect_pooled_activations(model, tok, pert_texts_calib, hook_path,
                                    batch_size=args.batch_size, max_len=args.max_len, device=device, pool=args.pool)
    if Hc.shape != Hp.shape:
        raise RuntimeError(f"shape mismatch {tuple(Hc.shape)} vs {tuple(Hp.shape)}")
    deltas = (Hp - Hc).to(device)

    def eval_variant(alpha: float, mask: Optional[torch.Tensor]) -> float:
        sup = ShiftAwarePCA(pca_r=args.pca_r, alpha=float(alpha), mask=mask)
        sup.fit(deltas)
        patch = ModulePatch(model, hook_path, sup); patch.enable()
        with torch.no_grad():
            logits = run_logits(model, tok, pert_texts_eval, batch_size=args.batch_size, max_len=args.max_len, device=device)
        patch.disable()
        return accuracy_from_logits(logits.to(device), y)

    sweep_rows = []
    for a in args.alpha_list:
        accp = eval_variant(a, mask=None)
        sweep_rows.append({"alpha": float(a), "pca_r": int(args.pca_r),
                           "acc_clean_base": acc_clean_base, "acc_pert_base": acc_pert_base,
                           "acc_pert": accp, "impr": accp - acc_pert_base})
        print(f"SWEEP r={args.pca_r} a={a:.3f} pert={accp:.4f} Δ={accp-acc_pert_base:+.4f}")

    out_prefix = args.save_prefix if args.save_prefix else str(Path(args.pairs_csv).with_suffix(""))
    sweep_csv = f"{out_prefix}.r{args.pca_r}.alpha_sweep.csv"
    import pandas as pd
    pd.DataFrame(sweep_rows).to_csv(sweep_csv, index=False)
    print(f"SAVE {sweep_csv}")

    alphas = [r["alpha"] for r in sweep_rows]
    accs   = [r["acc_pert"] for r in sweep_rows]
    plt.figure(figsize=(6,4))
    plt.plot(alphas, accs, marker="o")
    plt.axhline(acc_pert_base, linestyle="--")
    plt.title(f"PCA r={args.pca_r}: pert-acc vs alpha")
    plt.xlabel("alpha"); plt.ylabel("pert accuracy")
    plt.tight_layout()
    fig1 = f"{out_prefix}.r{args.pca_r}.alpha_sweep.png"
    plt.savefig(fig1, dpi=160)
    plt.close()
    print(f"SAVE {fig1}")

    mask_rows = []
    for mode in args.mask_modes:
        mode_l = mode.lower()
        if mode_l == "none":
            m = None
        else:
            m = build_coord_mask_from_deltas(deltas, mode=mode_l, frac=args.mask_frac)
        accp = eval_variant(args.mask_compare_alpha, mask=m)
        mask_rows.append({"mode": mode_l, "alpha": float(args.mask_compare_alpha),
                          "mask_frac": args.mask_frac, "acc_pert": accp,
                          "impr": accp - acc_pert_base})
        print(f"MASK mode={mode_l} a={args.mask_compare_alpha:.3f} frac={args.mask_frac:.3f} pert={accp:.4f} Δ={accp-acc_pert_base:+.4f}")

    mask_csv = f"{out_prefix}.r{args.pca_r}.mask_compare_a{args.mask_compare_alpha:.2f}.csv"
    pd.DataFrame(mask_rows).to_csv(mask_csv, index=False)
    print(f"SAVE {mask_csv}")

    labels = [r["mode"].upper() for r in mask_rows]
    accs2  = [r["acc_pert"] for r in mask_rows]
    plt.figure(figsize=(6,4))
    plt.bar(labels, accs2)
    plt.axhline(acc_pert_base, linestyle="--")
    plt.title(f"PCA r={args.pca_r}: coord masks @ alpha={args.mask_compare_alpha}")
    plt.ylabel("pert accuracy")
    plt.tight_layout()
    fig2 = f"{out_prefix}.r{args.pca_r}.mask_compare_a{args.mask_compare_alpha:.2f}.png"
    plt.savefig(fig2, dpi=160)
    plt.close()
    print(f"SAVE {fig2}")

    report = {
        "acc_clean_base": acc_clean_base,
        "acc_pert_base": acc_pert_base,
        "layer_idx": args.layer_idx,
        "module": args.module,
        "hook_path": hook_path,
        "pool": args.pool,
        "num_labels": num_labels,
        "label_mapping": ds_eval.id2label,
        "alpha_sweep_csv": sweep_csv,
        "mask_compare_csv": mask_csv,
        "alpha_sweep_png": fig1,
        "mask_compare_png": fig2,
        "pca_r": args.pca_r,
        "alpha_list": args.alpha_list,
        "mask_compare_alpha": args.mask_compare_alpha,
        "mask_frac": args.mask_frac,
        "mask_modes": args.mask_modes,
        "calib_csv": args.calib_csv if args.calib_csv else None,
        "calib_limit": args.calib_limit,
        "calib_from_eval_fallback": calib_from_eval_fallback,
        "calib_size_used_if_fallback": None if not calib_from_eval_fallback else min(args.calib_size, len(ds_eval)),
    }
    out_json = f"{out_prefix}.r{args.pca_r}.summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"SAVE {out_json}")


if __name__ == "__main__":
    main()
