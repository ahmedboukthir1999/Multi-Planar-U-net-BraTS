import json, random
from pathlib import Path


INPUT_ROOT  = Path("BraTS2021")
SPLIT_DIR   = Path("splits")     # destination folder for split lists
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15               # test is whatever remains
RANDOM_SEED = 42


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    SPLIT_DIR.mkdir(exist_ok=True)

    patient_ids = sorted([p.name for p in INPUT_ROOT.iterdir() if p.is_dir()])
    print(f"Found {len(patient_ids)} patient folders.")

    random.seed(RANDOM_SEED)
    random.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * TRAIN_FRAC)
    n_val   = int(n_total * VAL_FRAC)
    n_test  = n_total - n_train - n_val

    splits = {
        "train": patient_ids[:n_train],
        "val"  : patient_ids[n_train : n_train + n_val],
        "test" : patient_ids[n_train + n_val :],
    }

    # write txt files
    for split_name, case_list in splits.items():
        txt_path = SPLIT_DIR / f"{split_name}_cases.txt"
        with open(txt_path, "w") as f:
            for case in case_list:
                f.write(case + "\n")
        print(f"✅ {split_name:<5}  {len(case_list):3d} cases  → {txt_path}")

    # write JSON
    with open(SPLIT_DIR / "cases_split.json", "w") as jf:
        json.dump(splits, jf, indent=2)
    print("✅ cases_split.json written.")

    print("Done!")

if __name__ == "__main__":
    main()
