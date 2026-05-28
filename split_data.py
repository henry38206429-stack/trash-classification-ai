import os
import random
import shutil

# 原始資料夾
SOURCE_DIR = "dataset_raw"

# 輸出資料夾
TARGET_DIR = "dataset"

TRAIN_RATIO = 0.8

classes = ["glass", "paper", "metal", "plastic", "cardboard"]

for cls in classes:

    src_path = os.path.join(SOURCE_DIR, cls)

    train_path = os.path.join(TARGET_DIR, "train", cls)
    test_path = os.path.join(TARGET_DIR, "test", cls)

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    files = os.listdir(src_path)
    random.shuffle(files)

    split_idx = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # copy train
    for f in train_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(train_path, f)
        )

    # copy test
    for f in test_files:
        shutil.copy(
            os.path.join(src_path, f),
            os.path.join(test_path, f)
        )

    print(f"{cls} done: train={len(train_files)}, test={len(test_files)}")

print("All done!")