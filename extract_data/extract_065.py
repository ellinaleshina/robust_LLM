from datasets import load_from_disk, DatasetDict
import re, pathlib

SRC_DIR = "natural-instructions"
OUT_DIR = pathlib.Path("natural-instructions_task065_strict")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ds_all = load_from_disk(SRC_DIR)

def is_task065(ex):
    for k in ("task_id","ni_task_id","id","source_task_id"):
        if k in ex and ex[k] is not None:
            v = ex[k]
            if isinstance(v, (int, float)) and int(v) == 65:
                return True
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit() and int(s) == 65:
                    return True
    name = str(ex.get("task_name", "")).strip().lower()
    if re.match(r"^task0*65(_|\b)", name):
        return True
    return False

def filter_split(d):
    return d.filter(is_task065, load_from_cache_file=True)


filtered = {}
for split, d in ds_all.items():
    sub = filter_split(d)
    if len(sub):
        filtered[split] = sub
        print(f"[{split}] kept {len(sub)} / {len(d)}")

# если нет validation/test — создадим из train
if "validation" not in filtered and "train" in filtered:
    tmp = filtered["train"].train_test_split(test_size=0.1, seed=42)
    filtered["train"], filtered["validation"] = tmp["train"], tmp["test"]
if "test" not in filtered and "train" in filtered and len(filtered["train"]) > 100:
    tmp = filtered["train"].train_test_split(test_size=0.1, seed=42)
    filtered["train"], filtered["test"] = tmp["train"], tmp["test"]

ds065 = DatasetDict(filtered)

def normalize_example(ex):
    definition = ex.get("definition", "")
    if isinstance(definition, list):
        instruction = "\n".join(str(x) for x in definition)
    else:
        instruction = "" if definition is None else str(definition)


    input_txt = ex.get("inputs", "")
    if input_txt is None:
        input_txt = ""
    else:
        input_txt = str(input_txt)

    targets = ex.get("targets", [])
    if isinstance(targets, list):
        output_txt = targets[0] if len(targets) > 0 else ""
    else:
        output_txt = str(targets) if targets is not None else ""


    id_raw = ex.get("id") or ex.get("task_id") or ex.get("ni_task_id") or ex.get("source_task_id")

    return {
        "instruction": instruction,
        "input": input_txt,
        "output": output_txt,
        "id_raw": str(id_raw) if id_raw is not None else "",
        "task_name": str(ex.get("task_name", "")),
    }

for split in list(ds065.keys()):
    ds065[split] = ds065[split].map(
        normalize_example,
        remove_columns=[c for c in ds065[split].column_names
                        if c not in ("instruction","input","output","id_raw","task_name")],
        desc=f"Normalize {split}"
    )

ds065.save_to_disk(str(OUT_DIR))

for split, d in ds065.items():
    path = OUT_DIR / f"{split}.csv"
    d.select_columns(["instruction","input","output","id_raw","task_name"]).to_csv(path, index=False)
    print(f"[OK] wrote {path}")
