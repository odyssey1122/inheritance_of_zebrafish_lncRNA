"""
 seq_features.py
 STEP 4 — Extract sequence-based features

 What this does:
   - Computes k-mer frequencies (k = 2, 3, 4, 5)
   - Computes GC content
   - Computes nucleotide composition (A, C, G, U fractions)
   - Computes sequence length
   - Computes purine ratio (A + G fraction)

 Design principles:
   - Standalone island — takes FASTA files directly from disk
   - Features computed from TRAIN sequences only
   - Same feature extraction applied to test sequences
     using parameters learned from train only
   - Memory efficient — sequences processed one at a time
   - Output saved as CSV for use in downstream steps

 Input:
   - inherited_train.fasta  (from split_data.py)
   - non_inherited_train.fasta   (from split_data.py)
   - inherited_test.fasta   (from split_data.py)  ← applied but not fitted
   - non_inherited_test.fasta    (from split_data.py)  ← applied but not fitted

 Output:
   - features_output/train_features.csv
   - features_output/test_features.csv
"""

import csv
import logging
import argparse
from pathlib import Path
from itertools import product

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def get_all_kmers(k):
    """
    Generate all possible k-mers for RNA alphabet ACGU.
    Returns a sorted list so column order is always consistent.
    """
    return ["".join(p) for p in product("ACGU", repeat=k)]


def sequence_to_features(name, seq, label, kmer_sizes=[2, 3, 4, 5]):
    """
    Convert a single RNA sequence into a feature dictionary.

    Processes one sequence at a time — never accumulates sequences in memory. Called once per sequence then result is written to CSV immediately.

    Arguments:
      name       — sequence name (from FASTA header)
      seq        — cleaned RNA sequence string
      label      — "inherited" or "non_inherited"
      kmer_sizes — list of k values to compute (default [2,3,4,5])

    Returns:
      dictionary of {feature_name: value}
    """
    features = {}

    # Metadata 
    features["name"]  = name
    features["label"] = label

    n = len(seq)
    if n == 0:
        return None  # skip empty sequences

    # Basic composition 
    count_a = seq.count("A")
    count_c = seq.count("C")
    count_g = seq.count("G")
    count_u = seq.count("U")

    features["length"]      = n
    features["gc_content"]  = (count_g + count_c) / n
    features["freq_A"]      = count_a / n
    features["freq_C"]      = count_c / n
    features["freq_G"]      = count_g / n
    features["freq_U"]      = count_u / n
    features["purine_ratio"] = (count_a + count_g) / n

    # k-mer frequencies
    # Computed for each k separately
    # Normalised by total number of k-mers in this sequence
    for k in kmer_sizes:
        if n < k:
            # Sequence too short for this k — set all to 0
            for kmer in get_all_kmers(k):
                features[f"kmer{k}_{kmer}"] = 0.0
            continue

        # Count k-mers using a sliding window
        kmer_counts = {}
        total       = 0

        for i in range(n - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
            total += 1
        # Normalise and store all possible k-mers
        # (including those with count 0)
        for kmer in get_all_kmers(k):
            features[f"kmer{k}_{kmer}"] = (
                kmer_counts.get(kmer, 0) / total if total > 0 else 0.0
            )

    return features


def extract_features_to_csv(fasta_path, label, csv_writer,
                             kmer_sizes=[2, 3, 4, 5]):
    """
    Read a FASTA file and write features directly to CSV.

    Design principle:
    Each sequence is read, processed, written to CSV, then discarded. No list of feature vectors ever accumulates in memory.

    Arguments:
      fasta_path  — path to FASTA file
      label       — "inherited" or "non_inherited"
      csv_writer  — csv.DictWriter already open and ready
      kmer_sizes  — list of k values to compute
    """
    fasta_path = Path(fasta_path)

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    current_name = None
    current_seq  = []
    processed    = 0
    skipped      = 0

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                # Process previous sequence before moving to next
                if current_name is not None:
                    seq      = "".join(current_seq).upper()
                    features = sequence_to_features(
                        current_name, seq, label, kmer_sizes
                    )
                    if features is not None:
                        csv_writer.writerow(features)
                        processed += 1
                    else:
                        skipped += 1

                current_name = line[1:].strip()
                current_seq  = []

            elif line:
                current_seq.append(line)

    # Process the last sequence
    if current_name is not None:
        seq      = "".join(current_seq).upper()
        features = sequence_to_features(
            current_name, seq, label, kmer_sizes
        )
        if features is not None:
            csv_writer.writerow(features)
            processed += 1
        else:
            skipped += 1

    log.info(f"  {label}: {processed} sequences processed, "
             f"{skipped} skipped")
    
    
def extract_sequence_features(inherited_train, non_inherited_train,
                               inherited_test, non_inherited_test,
                               output_dir="features_output",
                               kmer_sizes=[2, 3, 4, 5]):
    """
    Extract sequence features from train and test FASTA files.
    Writes directly to CSV — no feature matrix ever in memory.

    Arguments:
      inherited_train — path to inherited train FASTA
      non_inherited_train  — path to non inherited train FASTA
      inherited_test  — path to inherited test FASTA
      non_inherited_test   — path to non inherited test FASTA
      output_dir     — where to save CSV files
      kmer_sizes     — list of k values (default [2,3,4,5])

    Returns:
      train_csv — path to train features CSV
      test_csv  — path to test features CSV
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    train_csv = output_dir / "train_features.csv"
    test_csv  = output_dir / "test_features.csv"

    log.info("Extracting Sequence Features")
    log.info(f"  k-mer sizes: {kmer_sizes}")

    # Build header must match exactly what sequence_to_features returns
    header = ["name", "label", "length", "gc_content",
              "freq_A", "freq_C", "freq_G", "freq_U",
              "purine_ratio"]
    for k in kmer_sizes:
        for kmer in get_all_kmers(k):
            header.append(f"kmer{k}_{kmer}")

    log.info(f"  Total features per sequence: {len(header) - 2}")

    # Train set
    log.info("Processing TRAIN sequences")
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        extract_features_to_csv(
            inherited_train, "inherited", writer, kmer_sizes
        )
        extract_features_to_csv(
            non_inherited_train, "non_inherited", writer, kmer_sizes
        )
    log.info(f"  Train features saved to {train_csv}")

    # Test set
    log.info("Processing TEST sequences...")
    with open(test_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        extract_features_to_csv(
            inherited_test, "inherited", writer, kmer_sizes
        )
        extract_features_to_csv(
            non_inherited_test, "non_inherited", writer, kmer_sizes
        )
    log.info(f"  Test features saved to {test_csv}")

    log.info("Sequence feature extraction complete")

    return train_csv, test_csv


# run

if __name__ == "__main__":
    import time
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Extract sequence features from FASTA files"
    )
    parser.add_argument(
        "--inherited_train",
        type=str,
        required=True,
        help="Path to inherited train FASTA sequences (from split_data.py)"
    )
    parser.add_argument(
        "--non_inherited_train",
        type=str,
        required=True,
        help="Path to non-inherited train FASTA sequences (from split_data.py)"
    )
    parser.add_argument(
        "--inherited_test",
        type=str,
        required=True,
        help="Path to inherited test FASTA sequences (from split_data.py)"
    )
    parser.add_argument(
        "--non_inherited_test",
        type=str,
        required=True,
        help="Path to non inherited test FASTA sequences (from split_data.py)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="features_output",
        help="Output directory for feature CSV files"
    )
    args = parser.parse_args()

    extract_sequence_features(
        inherited_train=args.inherited_train,
        non_inherited_train=args.non_inherited_train,
        inherited_test=args.inherited_test,
        non_inherited_test=args.non_inherited_test,
        output_dir=args.outdir
    )

    log.info(f"Total runtime: {time.time() - start:.2f} seconds")
