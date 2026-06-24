import os
import pandas as pd
from collections import defaultdict

def read_fasta_lengths(fasta_file):
    """
    Reads FASTA and returns dict: {sequence_name: length}
    """
    lengths = {}
    name = None
    seq = []

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    lengths[name] = len("".join(seq))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)

        if name:
            lengths[name] = len("".join(seq))

    return lengths


def merge_intervals(intervals):
    """
    Merge overlapping intervals
    """
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


def parse_repeatmasker_gff(gff_file, seq_lengths):
    """
    Parse RepeatMasker GFF and return TE features per sequence
    """

    # Initialize ALL sequences (fix bug #1)
    data = {}

    for seq in seq_lengths:
        data[seq] = {
            "intervals": [],
            "length": seq_lengths[seq]
        }

    # Parse GFF
    with open(gff_file) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            seqname = parts[0]
            start = int(parts[3])
            end = int(parts[4])

            if seqname not in data:
                continue

            data[seqname]["intervals"].append((start, end))

    # Compute coverage
    results = []

    for seq, info in data.items():
        intervals = info["intervals"]
        merged = merge_intervals(intervals)

        total_te = sum(end - start for start, end in merged)
        length = info["length"]

        coverage = total_te / length if length > 0 else 0

        results.append({
            "sequence_name": seq,
            "te_coverage": coverage,
            "te_count": len(intervals)
        })

    return pd.DataFrame(results)


def build_train_repeats():
    print("Processing TRAIN TE features...")

    inh_lengths = read_fasta_lengths("inherited_train.fasta")
    non_lengths = read_fasta_lengths("non_inherited_train.fasta")

    df_inh = parse_repeatmasker_gff(
        "inherited_train.fasta.out.gff",
        inh_lengths
    )
    df_inh["label"] = "inherited"

    df_non = parse_repeatmasker_gff(
        "non_inherited_train.fasta.out.gff",
        non_lengths
    )
    df_non["label"] = "non_inherited"

    df = pd.concat([df_inh, df_non], ignore_index=True)

    print("Rows:", len(df))
    print(df.head())

    df.to_csv("train_repeats.csv", index=False)
    print("Saved train_repeats.csv")


if __name__ == "__main__":
    build_train_repeats()
