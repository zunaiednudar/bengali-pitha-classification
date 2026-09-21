"""
Dataset Stratified Splitting & Selective Augmentation (Pipeline V2.1)
---------------------------------------------------------------------
Splits original curated pitha images into Train and Validation (and, optionally,
Test) sets BEFORE applying data augmentation.

Two modes:

  A. Fixed test set (--test-input given)      <- used by the current notebook
       * The test set is a separate, pre-sorted folder (<class>/<images>).
       * clean_data/ is split into Train and Validation only
         (default 85% / 15%; the two ratios must sum to 1.0).
       * Every clean_data image that is byte-identical to a test image is REMOVED
         from the train/val pool, so the test set cannot leak into training.
       * Test images are only standardized (EXIF orientation + 224x224 padding),
         never augmented.

  B. Three-way split (no --test-input)         <- original behaviour
       * clean_data/ is split into Train / Validation / Test (default 70/15/15).

Common to both modes:
  1. The split is performed on pure original images, then ONLY the training
     originals are augmented (num-augments copies each, named <stem>_aug<k>.jpg).
  2. Validation and test images stay unaugmented.
  3. Byte-identical duplicates inside the pool are collapsed to one image before
     splitting, so a duplicate cannot land in both train and validation.
  4. Output stems are made safe for the notebook's group logic: '_aug' is
     reserved for augmented copies (an original containing it is renamed to
     '-aug'), and stems are unique within a class even when two source files
     differ only by extension (every output is written as .jpg).
  5. All images are standardized to target-size x target-size using
     aspect-ratio preserving reflection padding.

The script refuses to write into an output folder that already contains a split,
so a new run can never silently mix files with an older one.

Outputs:
  output_dir/
  |-- train/             (originals + augmented copies)
  |-- val/               (pure originals)
  |-- test/              (pure originals)
  `-- split_manifest.json (audit log: assignments, removed duplicates, config)
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from pitha_preprocessing import (
    VALID_EXT,
    load_and_orient_image,
    resize_with_pad,
)


def random_horizontal_flip(img, p=0.5):
    """Flip image horizontally with probability p."""
    if np.random.random() < p:
        return cv2.flip(img, 1)
    return img.copy()


def random_rotation(img, max_angle=15):
    """Rotate image by random angle in [-max_angle, +max_angle]."""
    h, w = img.shape[:2]
    angle = np.random.uniform(-max_angle, max_angle)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101
    )
    return rotated


def random_color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2):
    """Randomly adjust brightness, contrast, and saturation."""
    result = img.astype(np.float32)

    # Brightness
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    result = result * b_factor

    # Contrast
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    mean = np.mean(result)
    result = (result - mean) * c_factor + mean

    # Saturation
    hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV).astype(np.float32)
    s_factor = 1.0 + np.random.uniform(-saturation, saturation)
    hsv[:, :, 1] = hsv[:, :, 1] * s_factor
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def random_crop_resize(img, min_area_ratio=0.85):
    """Random crop between min_area_ratio and 100%, then resize back."""
    h, w = img.shape[:2]
    area_ratio = np.random.uniform(min_area_ratio, 1.0)
    scale = np.sqrt(area_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    top = np.random.randint(0, h - new_h + 1)
    left = np.random.randint(0, w - new_w + 1)

    cropped = img[top:top + new_h, left:left + new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized


def add_gaussian_noise(img, sigma=0.01):
    """Add small Gaussian noise."""
    noise = np.random.normal(0, sigma * 255, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_image(img):
    """Apply stochastic augmentation pipeline."""
    aug = random_horizontal_flip(img)
    aug = random_rotation(aug)
    aug = random_color_jitter(aug)
    aug = random_crop_resize(aug)
    aug = add_gaussian_noise(aug)
    return aug


def process_image_file(img_path, target_size=224):
    """
    Load with EXIF orientation and pad to a target_size square.
    Returns a BGR numpy array, or None if the file cannot be processed.

    Failures are swallowed deliberately: this runs over thousands of files and a
    single corrupt or degenerate photo should not abort a 40-minute job.
    """
    try:
        pil_img = load_and_orient_image(img_path)
        if pil_img is None:
            return None
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return resize_with_pad(img_bgr, size=target_size)
    except Exception as exc:
        print(f"  [warn] Skipping unreadable image {img_path.name}: {exc}")
        return None


def write_jpg(path, img_bgr):
    """Write a BGR image as a quality-95 JPEG."""
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def file_md5(path):
    """MD5 of the raw file bytes; equal hashes mean byte-identical images."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def list_images(class_dir):
    """Sorted image files directly inside a class folder."""
    return sorted(f for f in class_dir.iterdir() if f.suffix.lower() in VALID_EXT)


def unique_stem(stem, digest, used):
    """
    Output stem that is safe for the notebook's group logic.

    '_aug' is reserved for augmented copies, so it is replaced in original names.
    Stems are kept unique (case-insensitively) within a class: every output is a
    .jpg, so '12.jpg' and '12.png' would otherwise overwrite each other and
    share one group id.
    """
    stem = stem.replace("_aug", "-aug")
    if stem.lower() in used:
        stem = f"{stem}-{digest[:6]}"
    used.add(stem.lower())
    return stem


def die(message):
    """Report a fatal problem and stop with a non-zero exit code."""
    print(f"Error: {message}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Stratified Train/Val(/Test) Split with Selective Augmentation"
    )
    parser.add_argument("--input", required=True,
                        help="Path to clean_data directory containing curated class folders")
    parser.add_argument("--output", required=True,
                        help="Path to output split_data directory (will contain train/, val/, test/)")
    parser.add_argument("--test-input", default=None,
                        help="Path to a pre-sorted test folder (<class>/<images>). When given, "
                             "--input is split into train/val only and this folder becomes test/")
    parser.add_argument("--train-ratio", type=float, default=None,
                        help="Proportion of originals for training "
                             "(default 0.85 with --test-input, otherwise 0.70)")
    parser.add_argument("--val-ratio", type=float, default=None,
                        help="Proportion of originals for validation (default 0.15)")
    parser.add_argument("--test-ratio", type=float, default=None,
                        help="Proportion of originals for testing (default 0.15; "
                             "not allowed together with --test-input)")
    parser.add_argument("--num-augments", type=int, default=5,
                        help="Number of augmented copies per training image (default 5)")
    parser.add_argument("--target-size", type=int, default=224,
                        help="Square resolution for padded images (default 224)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split (default 42)")
    args = parser.parse_args()

    # Resolve the ratios for the chosen mode
    fixed_test = args.test_input is not None
    if fixed_test:
        if args.test_ratio is not None:
            die("--test-ratio cannot be combined with --test-input "
                "(the test set is the folder you supplied).")
        train_ratio = 0.85 if args.train_ratio is None else args.train_ratio
        val_ratio = 0.15 if args.val_ratio is None else args.val_ratio
        test_ratio = 0.0
    else:
        train_ratio = 0.70 if args.train_ratio is None else args.train_ratio
        val_ratio = 0.15 if args.val_ratio is None else args.val_ratio
        test_ratio = 0.15 if args.test_ratio is None else args.test_ratio

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-4:
        die(f"Split ratios must sum to 1.0 (got {total_ratio})")

    # Seed the augmentation RNG too, not just the split, so a rerun with the
    # same seed reproduces the exact same augmented training images.
    np.random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    test_input = Path(args.test_input) if fixed_test else None

    if not input_dir.exists():
        die(f"Input directory not found: {input_dir}")
    if fixed_test and not test_input.exists():
        die(f"Test directory not found: {test_input}")

    # Refuse to mix a new split into an old one
    for existing in (output_dir / "train", output_dir / "val", output_dir / "test",
                     output_dir / "split_manifest.json"):
        if existing.exists() and (existing.is_file() or any(existing.iterdir())):
            die(f"{existing} already exists. Choose a new --output folder or "
                f"delete/rename the old split first.")

    # Discover classes
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        die(f"No class subdirectories found in {input_dir}")
    class_names = {d.name for d in class_dirs}

    # Fixed test set: check class names and hash every test image
    test_class_dirs = []
    test_hashes = {}
    if fixed_test:
        test_class_dirs = sorted([d for d in test_input.iterdir() if d.is_dir()])
        if not test_class_dirs:
            die(f"No class subdirectories found in {test_input}")
        unknown = sorted({d.name for d in test_class_dirs} - class_names)
        if unknown:
            die("Test classes not found in the training data (folder names must match "
                f"exactly): {unknown}")
        for d in test_class_dirs:
            for f in list_images(d):
                test_hashes.setdefault(file_md5(f), []).append(f"{d.name}/{f.name}")

    print("=" * 80)
    print("DATASET STRATIFIED SPLIT & SELECTIVE AUGMENTATION (PIPELINE V2.1)")
    print("=" * 80)
    print(f"Input Directory  : {input_dir.resolve()}")
    print(f"Output Directory : {output_dir.resolve()}")
    print(f"Target Classes   : {len(class_dirs)}")
    if fixed_test:
        n_test_found = sum(len(v) for v in test_hashes.values())
        print(f"Test Directory   : {test_input.resolve()}  ({n_test_found} images, fixed)")
        print(f"Split Ratios     : Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}% "
              f"(of clean_data after removing test overlap); Test = supplied folder")
    else:
        print(f"Split Ratios     : Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, "
              f"Test={test_ratio*100:.0f}%")
    print(f"Augmentation     : {args.num_augments}x (TRAIN ONLY)")
    print(f"Image Resolution : {args.target_size}x{args.target_size} (Aspect-preserving pad)")
    print(f"Random Seed      : {args.seed}")
    print("=" * 80)

    within_test_dupes = sum(len(v) - 1 for v in test_hashes.values())
    if within_test_dupes:
        print(f"[warn] {within_test_dupes} test image(s) are byte-identical to another "
              f"test image; the test set is kept as supplied.")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": {
            "mode": "train_val_with_fixed_test" if fixed_test else "train_val_test",
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "test_source": str(test_input) if fixed_test else None,
            "num_augments": args.num_augments,
            "target_size": args.target_size,
            "seed": args.seed,
        },
        "classes": {},
        "removed": {"overlap_with_test": [], "duplicates_in_pool": []},
        "totals": {}
    }

    min_images = 2 if fixed_test else 3
    summary_rows = []
    grand_train_orig, grand_train_aug, grand_val, grand_test = 0, 0, 0, 0
    grand_skipped = 0
    removed_test, removed_dupes = [], []

    for c_dir in class_dirs:
        class_name = c_dir.name

        # Build the pool: drop images that also appear in the fixed test set, and
        # collapse byte-identical duplicates so they cannot straddle train/val.
        img_paths, digests, seen = [], {}, set()
        for f in list_images(c_dir):
            digest = file_md5(f)
            if digest in test_hashes:
                removed_test.append(f"{class_name}/{f.name}")
                continue
            if digest in seen:
                removed_dupes.append(f"{class_name}/{f.name}")
                continue
            seen.add(digest)
            img_paths.append(f)
            digests[f] = digest
        n_total = len(img_paths)

        if n_total < min_images:
            print(f"  [skip] Class '{class_name}' has too few usable images ({n_total})")
            continue

        # Create class folders in splits
        c_train = train_dir / class_name
        c_val = val_dir / class_name
        c_train.mkdir(parents=True, exist_ok=True)
        c_val.mkdir(parents=True, exist_ok=True)
        c_test = test_dir / class_name
        if not fixed_test:
            c_test.mkdir(parents=True, exist_ok=True)

        # Fixed test:  one split  -> Train vs Val
        # Three-way:   first split -> Train vs Temp (Val + Test), then Temp -> Val vs Test
        # train_test_split raises when a requested side rounds down to zero, so
        # a class with very few images is reported and skipped rather than
        # killing the run partway through.
        try:
            if fixed_test:
                train_files, val_files = train_test_split(
                    img_paths,
                    train_size=train_ratio,
                    random_state=args.seed,
                    shuffle=True
                )
                test_files = []
            else:
                val_test_ratio = val_ratio + test_ratio
                val_proportion_of_temp = val_ratio / val_test_ratio
                train_files, temp_files = train_test_split(
                    img_paths,
                    train_size=train_ratio,
                    random_state=args.seed,
                    shuffle=True
                )
                val_files, test_files = train_test_split(
                    temp_files,
                    train_size=val_proportion_of_temp,
                    random_state=args.seed,
                    shuffle=True
                )
        except ValueError as exc:
            print(f"  [skip] Class '{class_name}' ({n_total} images) cannot be split "
                  f"with ratios {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}: {exc}")
            continue

        used_stems = set()      # keeps output stems unique within this class

        # Process Train: Base + Augmented
        train_orig_count = 0
        train_aug_count = 0
        for f in train_files:
            img_bgr = process_image_file(f, args.target_size)
            if img_bgr is None:
                continue
            stem = unique_stem(f.stem, digests[f], used_stems)

            # Base training image
            write_jpg(c_train / f"{stem}.jpg", img_bgr)
            train_orig_count += 1

            # Augmented variants
            for aug_i in range(1, args.num_augments + 1):
                aug_bgr = augment_image(img_bgr)
                write_jpg(c_train / f"{stem}_aug{aug_i}.jpg", aug_bgr)
                train_aug_count += 1

        # Process Val: Pure Originals Only (No augmentation)
        val_count = 0
        for f in val_files:
            img_bgr = process_image_file(f, args.target_size)
            if img_bgr is None:
                continue
            stem = unique_stem(f.stem, digests[f], used_stems)
            write_jpg(c_val / f"{stem}.jpg", img_bgr)
            val_count += 1

        # Process Test (three-way mode only): Pure Originals Only (No augmentation)
        test_count = 0
        for f in test_files:
            img_bgr = process_image_file(f, args.target_size)
            if img_bgr is None:
                continue
            stem = unique_stem(f.stem, digests[f], used_stems)
            write_jpg(c_test / f"{stem}.jpg", img_bgr)
            test_count += 1

        # Record manifest
        manifest["classes"][class_name] = {
            "total_originals": n_total,
            "train_originals": len(train_files),
            "train_augmented": train_aug_count,
            "train_total": train_orig_count + train_aug_count,
            "val_count": val_count,
            "test_count": test_count,
            "train_files": [f.name for f in train_files],
            "val_files": [f.name for f in val_files],
            "test_files": [f.name for f in test_files],
        }

        grand_train_orig += train_orig_count
        grand_train_aug += train_aug_count
        grand_val += val_count
        grand_test += test_count
        grand_skipped += ((len(train_files) - train_orig_count)
                          + (len(val_files) - val_count)
                          + (len(test_files) - test_count))

        summary_rows.append({
            "class": class_name,
            "originals": n_total,
            "train_orig": train_orig_count,
            "train_aug": train_aug_count,
            "train_total": train_orig_count + train_aug_count,
            "val": val_count,
            "test": test_count,
        })

    # Fixed test set: standardize the supplied images (no augmentation)
    if fixed_test:
        missing = sorted({d.name for d in test_class_dirs} - set(manifest["classes"]))
        if missing:
            die(f"These classes have test images but were skipped for training: {missing}")

        print("\nProcessing fixed test set ...")
        for d in test_class_dirs:
            c_test = test_dir / d.name
            c_test.mkdir(parents=True, exist_ok=True)
            used_stems = set()
            names = []
            for f in list_images(d):
                img_bgr = process_image_file(f, args.target_size)
                if img_bgr is None:
                    grand_skipped += 1
                    continue
                stem = unique_stem(f.stem, file_md5(f), used_stems)
                write_jpg(c_test / f"{stem}.jpg", img_bgr)
                names.append(f.name)
            manifest["classes"][d.name]["test_count"] = len(names)
            manifest["classes"][d.name]["test_files"] = names
            grand_test += len(names)
        for row in summary_rows:
            row["test"] = manifest["classes"][row["class"]]["test_count"]

    if removed_test:
        print(f"\n[info] Removed {len(removed_test)} clean_data image(s) that are "
              f"byte-identical to a test image.")
    if removed_dupes:
        print(f"[info] Removed {len(removed_dupes)} byte-identical duplicate(s) from the "
              f"train/val pool.")
    if grand_skipped:
        print(f"\n[warn] {grand_skipped} image(s) could not be processed and were "
              f"left out of the splits.")

    # Save manifest
    manifest["removed"] = {
        "overlap_with_test": removed_test,
        "duplicates_in_pool": removed_dupes,
    }
    manifest["totals"] = {
        "grand_train_originals": grand_train_orig,
        "grand_train_augmented": grand_train_aug,
        "grand_train_total": grand_train_orig + grand_train_aug,
        "grand_val_total": grand_val,
        "grand_test_total": grand_test,
        "grand_skipped": grand_skipped,
        "grand_removed_test_overlap": len(removed_test),
        "grand_removed_duplicates": len(removed_dupes),
    }
    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Print Summary Table
    orig_label = "Pool" if fixed_test else "Originals"
    print("\n" + "=" * 80)
    print(f"{'Class Name':<28} {orig_label:>10} {'Train Base':>12} {'Train Aug':>10} {'Train Total':>12} {'Val (Pure)':>11} {'Test (Pure)':>11}")
    print("-" * 80)
    for r in summary_rows:
        print(f"{r['class']:<28} {r['originals']:>10} {r['train_orig']:>12} {r['train_aug']:>10} {r['train_total']:>12} {r['val']:>11} {r['test']:>11}")
    print("-" * 80)
    print(f"{'TOTAL':<28} {sum(r['originals'] for r in summary_rows):>10} {grand_train_orig:>12} {grand_train_aug:>10} {grand_train_orig + grand_train_aug:>12} {grand_val:>11} {grand_test:>11}")
    print("=" * 80)
    if fixed_test:
        print("Pool = train + val originals left in clean_data after removing test overlap and duplicates.")
    print(f"Manifest written to: {manifest_path.resolve()}")
    print("Data splitting & selective augmentation complete!\n")


if __name__ == "__main__":
    main()
