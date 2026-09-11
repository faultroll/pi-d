"""
SOTS dataset builder -- indoor + outdoor combined training, separate test sets.

Changes from previous version
----------------------------------------------------------------------
- SUBSETS controls which subsets are included in train/val.
- Test sets are written per-subset: indoor_test.txt, outdoor_test.txt
  (predict.py reads one at a time for the indoor / outdoor result chapters).
- Combined train.txt and val.txt are written for train.py (no change needed there).
- Split is still done at the GT level to prevent leakage.
"""

import os
import random
from collections import defaultdict

# ----------------------------------------------------------------------
#  CONFIG
# ----------------------------------------------------------------------

SOTS_ROOT = './SOTS'
OUT_DIR   = './SOTS/split_txt'

# Hazy variants per GT.  None = all available.
#   indoor : up to 10 variants/GT  (500 GT -> max 5 000 pairs)
#   outdoor: up to 35 variants/GT  (500 GT -> max 17 500 pairs)
# Keep the same number for both to avoid imbalance in joint training.
HAZY_PER_GT = 5

# Subsets to include in training.  Both active = joint indoor+outdoor training.
SUBSETS = [
    'indoor',
    'outdoor',
]

VAL_RATIO  = 0.15   # fraction of GT images -> val  (per subset)
TEST_RATIO = 0.15   # fraction of GT images -> test (per subset)
# remaining 70% -> train

SEED = 42           # fixed for reproducibility
# SEED          = None


# ----------------------------------------------------------------------
#  Helpers
# ----------------------------------------------------------------------

def _base_id(hazy_filename: str) -> str:
    """
    Strip hazy-variant suffix -> GT base ID.
    indoor:  '1400_3.png'      -> '1400'
    outdoor: '0001_0.8_1.png'  -> needs GT-index lookup (handled in build_pairs)
    """
    stem = os.path.splitext(hazy_filename)[0]
    return stem.rsplit('_', 1)[0]


def build_pairs(gt_dir: str, hazy_dir: str, hazy_per_gt=None) -> dict:
    """
    Returns {base_id: [(hazy_path, gt_path), ...]}

    For outdoor, hazy filenames look like '0001_0.8_1.png' while GT is '0001.png'.
    We try stripping suffixes from right until we find a matching GT stem.
    """
    gt_index = {}
    for fname in os.listdir(gt_dir):
        if fname.startswith('.'):
            continue
        stem = os.path.splitext(fname)[0]
        gt_index[stem] = os.path.join(gt_dir, fname)

    hazy_by_id = defaultdict(list)
    for fname in sorted(os.listdir(hazy_dir)):
        if fname.startswith('.'):
            continue
        stem  = os.path.splitext(fname)[0]
        parts = stem.split('_')
        bid   = None
        # try from longest possible base downward
        for cut in range(len(parts), 0, -1):
            candidate = '_'.join(parts[:cut])
            if candidate in gt_index:
                bid = candidate
                break
        if bid is None:
            bid = stem.rsplit('_', 1)[0]   # fallback
        if bid in gt_index:
            hazy_by_id[bid].append(os.path.join(hazy_dir, fname))

    pairs_by_id = {}
    for bid, hazy_paths in hazy_by_id.items():
        gt_path = gt_index[bid]
        sorted_hazy = sorted(hazy_paths)
        if hazy_per_gt is not None:
            sorted_hazy = sorted_hazy[:hazy_per_gt]
        pairs_by_id[bid] = [(h, gt_path) for h in sorted_hazy]

    return pairs_by_id


def split_by_gt(pairs_by_id: dict, val_ratio: float, test_ratio: float, seed: int):
    """Split at GT level -> (train, val, test) flat lists."""
    ids = sorted(pairs_by_id.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)

    N       = len(ids)
    n_val   = max(1, int(N * val_ratio))
    n_test  = max(1, int(N * test_ratio))
    n_train = N - n_val - n_test

    def flatten(id_list):
        out = []
        for bid in id_list:
            out.extend(pairs_by_id[bid])
        return out

    return (flatten(ids[:n_train]),
            flatten(ids[n_train:n_train + n_val]),
            flatten(ids[n_train + n_val:]))


def write_txt(pairs, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for hazy, gt in pairs:
            f.write(f"{hazy} {gt}\n")


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    all_train, all_val = [], []

    for subset in SUBSETS:
        gt_dir   = os.path.join(SOTS_ROOT, subset, 'gt')
        hazy_dir = os.path.join(SOTS_ROOT, subset, 'hazy')

        if not (os.path.isdir(gt_dir) and os.path.isdir(hazy_dir)):
            print(f"[SKIP] {subset}: directory not found.")
            continue

        pairs = build_pairs(gt_dir, hazy_dir, hazy_per_gt=HAZY_PER_GT)
        train, val, test = split_by_gt(pairs, VAL_RATIO, TEST_RATIO, seed=SEED)

        n_gt = len(pairs)
        print(f"{subset}: {n_gt} GT x up to {HAZY_PER_GT} hazy  ->  "
              f"train={len(train)}  val={len(val)}  test={len(test)}")

        all_train.extend(train)
        all_val.extend(val)

        # -- per-subset test file (used by predict.py separately) --
        write_txt(test, os.path.join(OUT_DIR, f'{subset}_test.txt'))
        print(f"  wrote {subset}_test.txt  ({len(test)} pairs)")

    # -- combined train / val (used by train.py) --
    # Shuffle combined lists so indoor and outdoor are interleaved.
    rng = random.Random(SEED)
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    write_txt(all_train, os.path.join(OUT_DIR, 'train.txt'))
    write_txt(all_val,   os.path.join(OUT_DIR, 'val.txt'))

    print(f"\nCombined split (seed={SEED}):")
    print(f"  train : {len(all_train):>6} pairs  -> train.txt")
    print(f"  val   : {len(all_val):>6} pairs  -> val.txt")
    print(f"  test  : separate per subset  -> indoor_test.txt / outdoor_test.txt")
    print(f"\nFiles written to: {os.path.abspath(OUT_DIR)}")

    # Data size reference
    total_train = len(all_train)
    print(f"""
Training time estimate (batch=1, mid-range GPU):
  {total_train} train pairs x ~0.5 s/image = ~{total_train*0.5/60:.0f} min/epoch
  60 epochs ~ {total_train*0.5/60*60/60:.1f} hr
""")
