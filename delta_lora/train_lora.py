import os
import re
import random
import argparse
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def normalize_label(raw) -> str:
    """
    Приводит метку к строго 'Option 1' или 'Option 2'.
    Допускает варианты: 'option 1', 'Option1', '1', '2', и точки в конце.
    """
    s = str(raw).strip().strip('"').strip("'")
    s = re.sub(r"\s+\.", "", s)
    s = s.rstrip(".")
    low = s.lower().replace(" ", "")
    if low in {"option1", "1"} or "option1" in low:
        return "Option 1"
    if low in {"option2", "2"} or "option2" in low:
        return "Option 2"
    if "option 1" in s.lower():
        return "Option 1"
    if "option 2" in s.lower():
        return "Option 2"
    raise ValueError(f"Не удалось нормализовать метку: {raw!r}")


def compose_text(df: pd.DataFrame, instr_col: str, input_col: str, concat_instruction: bool) -> pd.Series:
    if concat_instruction and instr_col in df.columns:
        return (df[instr_col].fillna("").astype(str).str.strip() + "\n\n" +
                df[input_col].fillna("").astype(str).str.strip())
    else:
        return df[input_col].fillna("").astype(str).str.strip()


def load_split_csv(
    path: str,
    instr_col: str,
    input_col: str,
    label_col: str,
    label2id: dict,
    concat_instruction: bool,
) -> Dataset:
    df = pd.read_csv(path)
    for col in [input_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Columns: {list(df.columns)}")

    df["text"] = compose_text(df, instr_col, input_col, concat_instruction)

    df["label_norm"] = df[label_col].map(normalize_label)
    df["labels"] = df["label_norm"].map(label2id).astype("int64")

    if df["labels"].isna().any():
        bad = df[df["labels"].isna()][[label_col, "label_norm"]].head(5)
        raise ValueError(f"NaN в labels после нормализации. Примеры:\n{bad}")

    keep = ["text", "labels"]
    if "id_raw" in df.columns:
        keep.append("id_raw")
    return Dataset.from_pandas(df[keep], preserve_index=False)


def find_layer_mlp_targets(model, layer_idx: int):
    """
    Возвращает подстроки имён модулей для точечного матчинга up/down_proj ровно на слое `layer_idx`.
    PEFT фильтрует по `if any(substr in name for substr in target_modules)`.
    Покрываем разные схемы именования (HF/Meta/Llama variants).
    """
    candidates = [
        f"model.layers.{layer_idx}.mlp.up_proj",
        f"model.layers.{layer_idx}.mlp.down_proj",
        f"model.model.layers.{layer_idx}.mlp.up_proj",
        f"model.model.layers.{layer_idx}.mlp.down_proj",
        f"layers.{layer_idx}.mlp.up_proj",
        f"layers.{layer_idx}.mlp.down_proj",
    ]
    names = [n for n, _ in model.named_modules()]
    selected = sorted({c for c in candidates if any(c in n for n in names)})

    if not selected:
        hint = [n for n in names if f".layers.{layer_idx}.mlp." in n][-15:]
        raise RuntimeError(
            f"Не нашёл up/down_proj на слое {layer_idx}. "
            f"Примеры имён модулей рядом со слоем:\n" + "\n".join(hint)
        )
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct"))
    ap.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--train_file", default=os.environ.get("TRAIN_FILE", "train.csv"))
    ap.add_argument("--val_file",   default=os.environ.get("VAL_FILE",   "val.csv"))
    ap.add_argument("--test_file",  default=os.environ.get("TEST_FILE",  "test.csv"))

    ap.add_argument("--instr_col",  default=os.environ.get("INSTR_COL",  "instruction"))
    ap.add_argument("--input_col",  default=os.environ.get("INPUT_COL",  "input"))
    ap.add_argument("--label_col",  default=os.environ.get("LABEL_COL",  "output"))

    ap.add_argument("--concat_instruction", type=int, default=int(os.environ.get("CONCAT_INSTRUCTION", "1")),
                    help="1 = prepend instruction к input; 0 = использовать только input")
    ap.add_argument("--out_dir",    default=os.environ.get("OUT_DIR",    "out/lora_task065_layer9"))
    ap.add_argument("--layer_idx",  type=int, default=int(os.environ.get("LAYER_IDX", "9")))
    ap.add_argument("--max_len",    type=int, default=int(os.environ.get("MAX_LEN", "512")))
    ap.add_argument("--lr",         type=float, default=float(os.environ.get("LR", "1e-4")))
    ap.add_argument("--epochs",     type=float, default=float(os.environ.get("EPOCHS", "3")))
    ap.add_argument("--tr_bs",      type=int, default=int(os.environ.get("TR_BS", "16")))
    ap.add_argument("--ev_bs",      type=int, default=int(os.environ.get("EV_BS", "16")))
    ap.add_argument("--seed",       type=int, default=int(os.environ.get("SEED", "42")))
    ap.add_argument("--save_preds", type=int, default=int(os.environ.get("SAVE_PREDS", "1")))
    args = ap.parse_args()

    LABEL2ID = {"Option 1": 0, "Option 2": 1}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    train = load_split_csv(
        os.path.join(args.data_dir, args.train_file),
        args.instr_col, args.input_col, args.label_col, LABEL2ID, bool(args.concat_instruction)
    )
    val   = load_split_csv(
        os.path.join(args.data_dir, args.val_file),
        args.instr_col, args.input_col, args.label_col, LABEL2ID, bool(args.concat_instruction)
    )
    test  = load_split_csv(
        os.path.join(args.data_dir, args.test_file),
        args.instr_col, args.input_col, args.label_col, LABEL2ID, bool(args.concat_instruction)
    )
    ds = DatasetDict({"train": train, "validation": val, "test": test})

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=args.max_len)

    remove_cols = ["text"]
    if "id_raw" in ds["train"].column_names:
        remove_cols = ["text"]
    ds = ds.map(tok, batched=True, remove_columns=remove_cols)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=len(LABEL2ID),
        torch_dtype=dtype,
        device_map="auto",
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    base.config.problem_type = "single_label_classification"
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()

    targets = find_layer_mlp_targets(base, args.layer_idx)
    print("[INFO] Will attach LoRA to:", targets)

    peft_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["score"],
    )
    model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    fp16 = not torch.cuda.is_bf16_supported()
    args_hf = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",   
        learning_rate=args.lr,
        weight_decay=0.0,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.tr_bs,
        per_device_eval_batch_size=args.ev_bs,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to=[],
        fp16=fp16,
        bf16=not fp16,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
        }

    trainer = Trainer(
        model=model,
        args=args_hf,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("[INFO] Evaluating…")
    val_metrics  = trainer.evaluate(ds["validation"])
    test_metrics = trainer.evaluate(ds["test"])

    print("\n=== Validation ===")
    for k, v in val_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:>15}: {v:.6f}")

    print("\n=== Test ===")
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k:>15}: {v:.6f}")

    if args.save_preds:
        print("[INFO] Predicting test split and saving to CSV…")
        preds = trainer.predict(ds["test"])
        pred_ids = preds.predictions.argmax(axis=-1)
        pred_labels = [ID2LABEL[int(i)] for i in pred_ids]

        id_raw = None
        if "id_raw" in ds["test"].column_names:
            id_raw = ds["test"]["id_raw"]

        out_df = pd.DataFrame({
            "pred_id": pred_ids,
            "pred_label": pred_labels,
        })
        if id_raw is not None:
            out_df.insert(0, "id_raw", id_raw)

        out_csv = os.path.join(args.out_dir, "test_predictions.csv")
        out_df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[INFO] Saved: {out_csv}")


if __name__ == "__main__":
    main()
