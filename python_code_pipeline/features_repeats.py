"""
features_repeats.py
STEP 7.5 — Parse RepeatMasker .out files into per-sequence TE features

Parses the RepeatMasker .out file (not .gff) because it contains an explicit class/family column (e.g. DNA/hAT, LTR/Copia, LINE/CR1) so no motif name guessing is needed.

Features per sequence:
  te_coverage           total TE bp / sequence length (0.0-1.0)
  dna_transposon_count  number of DNA transposon hits
  ltr_count             number of LTR element hits
  line_count            number of LINE hits
  sine_count            number of SINE hits
  simple_repeat_density simple repeat + satellite bp / sequence length
  is_te_derived         1 if te_coverage > 0.2 else 0

Input:
  RepeatMasker .out files + FASTA files (for sequence names + lengths)

Output:
  features_output/train_repeats.csv
  features_output/test_repeats.csv
"""

import csv
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# paths
RM_DIR    = Path("/lustre/dheeraj.joshi/vishaal/repeatmasker_csv_maker")
FASTA_DIR = Path("/lustre/dheeraj.joshi/vishaal/repeatmasker_csv_maker")
OUT_DIR   = Path("/lustre/dheeraj.joshi/vishaal/repeatmasker_csv_maker")

OUT_FILES = {
    ("inherited",     "train"): RM_DIR / "inherited_train.fasta.out",
    ("inherited",     "test"):  RM_DIR / "inherited_test.fasta.out",
    ("non_inherited", "train"): RM_DIR / "non_inherited_train.fasta.out",
    ("non_inherited", "test"):  RM_DIR / "non_inherited_test.fasta.out",
}

FASTA_FILES = {
    ("inherited",     "train"): FASTA_DIR / "inherited_train.fasta",
    ("inherited",     "test"):  FASTA_DIR / "inherited_test.fasta",
    ("non_inherited", "train"): FASTA_DIR / "non_inherited_train.fasta",
    ("non_inherited", "test"):  FASTA_DIR / "non_inherited_test.fasta",
}

# Known single-word class names (no / separator)
SIMPLE_CLASSES  = {"SIMPLE_REPEAT", "SATELLITE"}
LOW_COMP_CLASSES = {"LOW_COMPLEXITY"}


# Helper: read FASTA → {name: length} 
def read_fasta_lengths(fasta_path):
    """
    Read a FASTA file and return sequence lengths.

    Arguments:
      fasta_path - path to FASTA file

    Returns:
      dict {sequence_name: length}
    """
    lengths = {}
    name    = None
    bases   = 0

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = bases
                name  = line[1:].split()[0]
                bases = 0
            elif line:
                bases += len(line)

    if name is not None:
        lengths[name] = bases

    return lengths


# Helper: merge overlapping intervals
def merge_intervals(intervals):
    """
    Merge overlapping or adjacent intervals to avoid double-counting bp.

    Arguments:
      intervals - list of (start, end) tuples

    Returns:
      list of merged [start, end] lists
    """
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged    = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


# Parse one .out file
def parse_out_file(out_path, fasta_lengths):
    """
    Parse a RepeatMasker .out file and compute TE features per sequence.

    Sequences with zero hits are included with all-zero features —
    they are found from fasta_lengths, not from the .out file.

    Arguments:
      out_path      - path to RepeatMasker .out file
      fasta_lengths - dict {sequence_name: length} from read_fasta_lengths()

    Returns:
      list of feature dicts, one per sequence
    """
    # Initialise storage for every sequence — including zero-hit ones
    data = {}
    for seq, length in fasta_lengths.items():
        data[seq] = {
            "length":           length,
            "all_intervals":    [],   # all TE hits — for te_coverage
            "simple_intervals": [],   # simple repeat hits — for density
            "dna":              0,
            "ltr":              0,
            "line":             0,
            "sine":             0,
        }

    skipped = 0

    with open(out_path) as f:
        for line in f:
            # Skip header lines — RepeatMasker .out has 3 header lines
            # Skip blank lines and header lines
            line = line.rstrip()
            if not line:
                continue
            if not line[0].isdigit() and not line[0] == " ":
                continue
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            if len(parts) < 11:
                continue

            seq   = parts[4]
            start = int(parts[5])
            end   = int(parts[6])

            # Find class/family field — contains '/' or is a known word
            cls = "Unknown"
            KNOWN = {"Simple_repeat", "Satellite", "Low_complexity",
                     "Unknown", "Small_RNA", "rRNA", "snRNA", "srpRNA"}
            for part in parts[10:]:
                if "/" in part or part in KNOWN:
                    cls = part
                    break

            if seq not in data:
                skipped += 1
                continue

            # All TE intervals — used for total te_coverage
            # Excludes simple repeats and low complexity (not TEs)
            if not cls.startswith("Simple_repeat") and \
               cls != "Satellite" and \
               cls != "Low_complexity":
                data[seq]["all_intervals"].append((start, end))

            # Per-family counts using your logic
            if cls.startswith("DNA"):
                data[seq]["dna"] += 1
            elif cls.startswith("LTR"):
                data[seq]["ltr"] += 1
            elif cls.startswith("LINE"):
                data[seq]["line"] += 1
            elif cls.startswith("SINE"):
                data[seq]["sine"] += 1
            elif cls in ("Simple_repeat", "Satellite"):
                data[seq]["simple_intervals"].append((start, end))

    if skipped > 0:
        log.warning(f"  {skipped} hits skipped — sequence not in FASTA")

    # Compute features from intervals
    rows = []
    for seq, info in data.items():
        length = info["length"]

        # Total TE coverage — merge intervals first to avoid double-counting
        merged_te     = merge_intervals(info["all_intervals"])
        te_bp         = sum(e - s + 1 for s, e in merged_te)
        te_coverage   = te_bp / length if length > 0 else 0.0

        # Simple repeat density — same merging logic
        merged_simple = merge_intervals(info["simple_intervals"])
        simple_bp     = sum(e - s + 1 for s, e in merged_simple)
        simple_density = simple_bp / length if length > 0 else 0.0

        rows.append({
            "sequence_name":                 seq,
            "te_coverage":          round(te_coverage, 6),
            "dna_transposon_count": info["dna"],
            "ltr_count":            info["ltr"],
            "line_count":           info["line"],
            "sine_count":           info["sine"],
            "simple_repeat_density": round(simple_density, 6),
            "is_te_derived":        1 if te_coverage > 0.2 else 0,
        })

    return rows


# Main: process all four splits and write CSVs
def extract_repeat_features():
    """
    Process all four splits (inherited/non_inherited x train/test).
    Writes train_repeats.csv and test_repeats.csv to OUT_DIR.
    """
    OUT_DIR.mkdir(exist_ok=True)

    HEADER = [
        "sequence_name", "label",
        "te_coverage",
        "dna_transposon_count",
        "ltr_count",
        "line_count",
        "sine_count",
        "simple_repeat_density",
        "is_te_derived",
    ]

    for split in ("train", "test"):
        out_csv = OUT_DIR / f"{split}_repeats.csv"

        log.info(f"Processing {split}...")

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADER)
            writer.writeheader()

            for label in ("inherited", "non_inherited"):
                out_path   = OUT_FILES[(label, split)]
                fasta_path = FASTA_FILES[(label, split)]

                if not out_path.exists():
                    raise FileNotFoundError(f"Missing .out file: {out_path}")
                if not fasta_path.exists():
                    raise FileNotFoundError(f"Missing FASTA file: {fasta_path}")

                log.info(f"  Reading FASTA lengths: {fasta_path.name}")
                lengths = read_fasta_lengths(fasta_path)
                log.info(f"  {len(lengths)} sequences in FASTA")

                log.info(f"  Parsing .out file: {out_path.name}")
                rows = parse_out_file(out_path, lengths)
                log.info(f"  {len(rows)} rows computed")

                for row in rows:
                    row["label"] = label
                    writer.writerow(row)

        log.info(f"Saved: {out_csv}")

    log.info("Done.")


if __name__ == "__main__":
    import time
    start = time.time()
    extract_repeat_features()
    log.info(f"Total runtime: {time.time() - start:.2f} seconds")
