from datasets import load_dataset, DatasetDict

train = load_dataset("Muennighoff/natural-instructions", split="train", verification_mode="no_checks")
test  = load_dataset("Muennighoff/natural-instructions", split="test",  verification_mode="no_checks")

# вручную сделать validation
tmp = train.train_test_split(test_size=0.1, seed=42)
ds = DatasetDict({"train": tmp["train"], "validation": tmp["test"], "test": test})
ds.save_to_disk("natural-instructions")
