"""
Brain Tumor MRI dataset prep (Kaggle: sartajbhuvaji/brain-tumor-classification-mri)

This script performs the following tasks:
1. Downloads the dataset (if missing)
2. Extracts zip contents into raw_data/
3. Merges and splits data into train/val/test sets based on user configuration
4. Preprocesses all images (resizes to 224x224 and saves as NumPy arrays)
5. Logs random seeds and experiment settings to JSON for reproducibility

Usage:
    python preprocess.py --force-resplit --regenerate-seed --use-kaggle-test

Output:
    - train_X.npy, train_y.npy
    - val_X.npy, val_y.npy
    - test_X.npy, test_y.npy
    - split_seed.json (all seeds used)
    - experiment_config.json (each run's configuration)
    - test_source.txt (notes how test set was generated)

Folder structure after running this script:
├── raw_data/
│   ├── Training/
│   ├── Testing/
│   └── Combined/             created if --use-kaggle-test is False
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── train_X.npy, train_y.npy ...
├── split_seed.json
└── experiment_config.json

"""


import os
import zipfile
import shutil
import random
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime


KAGGLE_DATASET = "sartajbhuvaji/brain-tumor-classification-mri"
ZIP_NAME = "brain-tumor-classification-mri.zip"
EXTRACT_FOLDER = "raw_data"
FINAL_FOLDER = "data"
SEED_FILE = "split_seed.json"

# === CLI Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--force-resplit', action='store_true', help='Force re-download and resplit of dataset')
parser.add_argument('--regenerate-seed', action='store_true', help='Generate a new random seed and save it')
parser.add_argument('--use-kaggle-test', action='store_true', help='Use original Testing folder as final test set')
args = parser.parse_args()

#to adjust the split ratio according to the technique used either kaggle's pure testing folder or all shuffeled just for variability
# and in case i compare it to existing resuklts using the kaggle dataset then i should just use the testing folder already there
# let me know if you think i should delete it and just use one
if args.use_kaggle_test:
    SPLIT_RATIO = (0.85, 0.15, 0.0)
else:
    SPLIT_RATIO = (0.7, 0.15, 0.15)


def download_dataset():
    if not os.path.exists(ZIP_NAME):
        print("Downloading dataset from Kaggle")
        os.system(f"kaggle datasets download -d {KAGGLE_DATASET}")
    else:
        print("Dataset already downloaded")

def extract_dataset():
    if not os.path.exists(EXTRACT_FOLDER):
        print("Extracting dataset")
        with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_FOLDER)
    else:
        print("Dataset already extracted")


#keeps track of used seeds to be able to reproduce it
def get_or_generate_seed():
    history = []

    if os.path.exists(SEED_FILE):
        try:
            with open(SEED_FILE, 'r') as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except json.JSONDecodeError:
            history = []

    if args.regenerate_seed or not history:
        seed = random.randint(0, 999999)
        timestamp = datetime.now().isoformat()
        entry = {"seed": seed, "timestamp": timestamp}
        history.append(entry)
        with open(SEED_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Generated new seed: {seed} at {timestamp}")
    else:
        seed = history[-1]["seed"]
        print(f"Using saved seed: {seed} from {history[-1]['timestamp']}")

    return seed


def split_dataset():
    print("Splitting into train/val/test")

    train_path = os.path.join(EXTRACT_FOLDER, "Training")
    test_path = os.path.join(EXTRACT_FOLDER, "Testing")
    combined_path = os.path.join(EXTRACT_FOLDER, "Combined")

    if args.use_kaggle_test:
        source_path = train_path
    else:
        #merges both training and testing folders in case of complete shuffling
        print("Merging Training and Testing folders into Combined")
        if os.path.exists(combined_path):
            shutil.rmtree(combined_path)
        os.makedirs(combined_path)

        # --- begin: robust merge with unique filenames + audit ---
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        report = {
            "train_counts": {},
            "test_counts": {},
            "combined_counts": {},
            "collisions_renamed": 0,
            "skipped": []
        }

        def list_images(d):
            out = []
            for p in os.listdir(d):
                fp = os.path.join(d, p)
                if os.path.isfile(fp):
                    ext = os.path.splitext(fp)[1].lower()
                    if ext in exts:
                        out.append(fp)
            return out

        def copy_split(base_dir, split_tag):
            nonlocal report
            if not os.path.exists(base_dir):
                print(f"Warning: {base_dir} not found. Skipping.")
                return
            for category in os.listdir(base_dir):
                src_dir = os.path.join(base_dir, category)
                if not os.path.isdir(src_dir):
                    continue
                dst_dir = os.path.join(combined_path, category)
                os.makedirs(dst_dir, exist_ok=True)

                files = list_images(src_dir)
                report_key = "train_counts" if split_tag == "train" else "test_counts"
                report.setdefault(report_key, {})
                report[report_key][category] = len(files)

                for src_file in files:
                    base = os.path.basename(src_file)
                    stem, ext = os.path.splitext(base)
                    # prefix with split to avoid train/test name collisions
                    cand_name = f"{split_tag}_{stem}{ext}"
                    dst_file = os.path.join(dst_dir, cand_name)
                    # ensure uniqueness if even that collides
                    k = 1
                    while os.path.exists(dst_file):
                        cand_name = f"{split_tag}_{stem}_{k}{ext}"
                        dst_file = os.path.join(dst_dir, cand_name)
                        k += 1
                        report["collisions_renamed"] += 1
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        report["skipped"].append({"file": src_file, "reason": f"copy_error:{e}"})

        copy_split(train_path, "train")
        copy_split(test_path, "test")

        # post-merge counts
        for category in os.listdir(combined_path):
            cdir = os.path.join(combined_path, category)
            if os.path.isdir(cdir):
                report["combined_counts"][category] = len([f for f in os.listdir(cdir)
                                                           if os.path.isfile(os.path.join(cdir, f)) and
                                                           os.path.splitext(f)[1].lower() in exts])

        with open(os.path.join(combined_path, "combine_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("[combine] wrote", os.path.join(combined_path, "combine_report.json"))
        # merge with uniqe filenames to not overwrite files

        source_path = combined_path

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Expected path {source_path} not found.")

    categories = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, d))]
    seed = get_or_generate_seed()
    random.seed(seed)

    for split in ["train", "val", "test"]:
        for cat in categories:
            os.makedirs(os.path.join(FINAL_FOLDER, split, cat), exist_ok=True)

    #data splitting
    for cat in categories:
        all_files = [f for f in os.listdir(os.path.join(source_path, cat)) if os.path.isfile(os.path.join(source_path, cat, f))]
        random.shuffle(all_files)

        n = len(all_files)
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])

        if args.use_kaggle_test:
            train_files = all_files[:n_train]
            val_files = all_files[n_train:]
            test_files = []
        else:
            train_files = all_files[:n_train]
            val_files = all_files[n_train:n_train + n_val]
            test_files = all_files[n_train + n_val:]

        for split, split_files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            for file in split_files:
                src = os.path.join(source_path, cat, file)
                dst = os.path.join(FINAL_FOLDER, split, cat, file)
                shutil.copy(src, dst)

    #if --use-kaggle-test, copy that test set
    if args.use_kaggle_test:
        print("Copying Kaggle Testing set to test split")
        for cat in categories:
            src_dir = os.path.join(test_path, cat)
            dst_dir = os.path.join(FINAL_FOLDER, "test", cat)
            if not os.path.exists(src_dir):
                print(f"Skipping {cat}: not found in Testing/")
                continue
            for file in os.listdir(src_dir):
                src_file = os.path.join(src_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy(src_file, os.path.join(dst_dir, file))

        with open("test_source.txt", "w") as f:
            f.write("source: kaggle Testing\n")
    else:
        with open("test_source.txt", "w") as f:
            f.write("source: split from Combined Training + Testing\n")


def preprocess(split="train", img_size=224):
    print(f"Preprocessing {split} data...")
    split_path = os.path.join(FINAL_FOLDER, split)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_path} not found.")

    categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    X, y = [], []

    for idx, category in enumerate(categories):
        path = os.path.join(split_path, category)
        if not os.path.exists(path):
            print(f"Skipping {category} — folder not found.")
            continue

        for img_name in tqdm(os.listdir(path), desc=f"{split.upper()} - {category}"):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(idx)
            except Exception as e:
                print(f"Failed to process {img_name}: {e}")
                continue

    X = np.array(X)
    y = np.array(y)
    np.save(f"{split}_X.npy", X)
    np.save(f"{split}_y.npy", y)
    print(f"Saved {split}_X.npy and {split}_y.npy")

if __name__ == "__main__":
    if (
        not args.force_resplit and
        os.path.exists("train_X.npy") and
        os.path.exists("train_y.npy") and
        os.path.exists("val_X.npy") and
        os.path.exists("val_y.npy") and
        os.path.exists("test_X.npy") and
        os.path.exists("test_y.npy")
    ):
        print("Preprocessed data already exists. Use --force-resplit to regenerate.")
    else:
        if args.force_resplit:
            print("Force resplit enabled. Cleaning previous files")
            for fname in ["train_X.npy", "train_y.npy", "val_X.npy", "val_y.npy", "test_X.npy", "test_y.npy"]:
                fpath = os.path.join(FINAL_FOLDER, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            if os.path.exists(FINAL_FOLDER):
                shutil.rmtree(FINAL_FOLDER)

        download_dataset()
        extract_dataset()
        split_dataset()
        preprocess("train")
        preprocess("val")
        preprocess("test")



    # === Save experiment configuration ===
    experiment_entry = {
        "timestamp": datetime.now().isoformat(),
        "used_kaggle_test": args.use_kaggle_test,
        "force_resplit": args.force_resplit,
        "regenerated_seed": args.regenerate_seed,
        "image_size": 224,
        "split_ratio": SPLIT_RATIO,
    }

    if os.path.exists(SEED_FILE):
        with open(SEED_FILE, "r") as f:
            try:
                seed_history = json.load(f)
                if isinstance(seed_history, list) and seed_history:
                    experiment_entry["random_seed"] = seed_history[-1]["seed"]
                    experiment_entry["seed_timestamp"] = seed_history[-1]["timestamp"]
            except json.JSONDecodeError:
                pass

    config_file = "experiment_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                existing_config = json.load(f)
            if not isinstance(existing_config, list):
                existing_config = []
        except json.JSONDecodeError:
            existing_config = []
    else:
        existing_config = []

    existing_config.append(experiment_entry)

    with open(config_file, "w") as f:
        json.dump(existing_config, f, indent=4)

    print("Saved experiment configuration to experiment_config.json")
