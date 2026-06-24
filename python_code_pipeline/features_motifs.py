"""
 features_motifs.py
 STEP 6 — Extract motif-based features

 Three sources:
   SOURCE 1 — Hand-curated motifs (hardcoded sequences) miR-430 target sites, m6A, polyA signals, PUMILIO

   SOURCE 2 — STREME discovered motifs
     19 motifs enriched in inherited + 19 in non-inherited. Parsed directly from streme.txt output files

   SOURCE 3 — TOMTOM RBP motifs
     Top 10 RBPs from inherited TOMTOM + top 10 from non-inherited TOMTOM. Consensus sequences extracted from TOMTOM tsv output

 All sources:
   - Count total occurrences per sequence
   - Count in 5', middle, 3' thirds (positional)

 Design principles:
   - Standalone island
   - Pure Python scanning — no external tools needed here
   - Sequences processed one at a time
   - Written directly to CSV

 Input:
   - inherited_train_rna.fasta / non_inherited_train_rna.fasta
   - inherited_test_rna.fasta  / non_inherited_test_rna.fasta
   - streme_output_inherited/streme.txt
   - streme_output_noninherited/streme.txt
   - tomtom_output_inherited_v2/tomtom.tsv
   - tomtom_output_noninherited_v2/tomtom.tsv

 Output:
   - features_output/train_motifs.csv
   - features_output/test_motifs.csv
"""

import re
import csv
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


# SOURCE 1 Hand-curated motifs

HYPOTHESIS_MOTIFS = {

    # miR-430 TARGET SITES (complement of miR-430 seed in lncRNA)
    # miR-430 seed: 5'-GCACUU-3' → Target site in lncRNA: 5'-AAGUGC-3'
    "mir430_target_6mer":   "AAGUGC",
    "mir430_target_7mer_m8": "AAGUGCU",

    # m6A sites (direct sequence in lncRNA)
    "m6a_motif_ggacu": "GGACU",
    "m6a_motif_rrach": "[AG][AG]AC[ACU]",

    # PolyA signals (direct sequence in lncRNA)
    "polya_signal":         "AAUAAA",
    "polya_variant_auuaaa": "AUUAAA",
    "polya_variant_aguaaa": "AGUAAA",

    # PUMILIO PUF-box TARGET SITES
    "pumilio_pufbox": "UGU[AG]UAUA",
}

# IUPAC ambiguity code to regex mapping
IUPAC_MAP = {
    'R': '[AG]',  'Y': '[CU]',  'S': '[CG]',  'W': '[AU]',
    'K': '[GU]',  'M': '[AC]',  'B': '[CGU]', 'D': '[AGU]',
    'H': '[ACU]', 'V': '[ACG]', 'N': '[ACGU]'
}


# Scanning functions

def count_motif_occurrences(seq, motif_pattern):
    """
    Count occurrences of a motif in a sequence using regex. Handles both exact strings and IUPAC regex patterns. Uses lookahead to capture overlapping matches.

    Arguments:
      seq           — RNA sequence string (uppercase)
      motif_pattern — exact string or regex pattern

    Returns:
      integer count of matches
    """
    try:
        return len(re.findall(f"(?={motif_pattern})", seq))
    except re.error:
        log.warning(f"Invalid regex pattern: {motif_pattern}")
        return 0


def get_positional_counts(seq, motif_pattern):
    """
    Count motif occurrences in 5', middle, and 3' thirds.

    Arguments:
      seq           — RNA sequence string
      motif_pattern — motif string or regex

    Returns:
      tuple (five_prime_count, middle_count, three_prime_count)
    """
    n     = len(seq)
    third = max(1, n // 3)

    five_prime  = seq[:third]
    middle      = seq[third:third * 2]
    three_prime = seq[third * 2:]

    return (
        count_motif_occurrences(five_prime,  motif_pattern),
        count_motif_occurrences(middle,      motif_pattern),
        count_motif_occurrences(three_prime, motif_pattern),
    )


def scan_motif_group(seq, motifs_dict, prefix=""):
    """
    Scan a sequence for all motifs in a dictionary.
    Returns total + positional counts for each.

    Arguments:
      seq         — RNA sequence string
      motifs_dict — {motif_name: motif_pattern}
      prefix      — optional prefix for feature names

    Returns:
      dictionary of features
    """
    features = {}
    for motif_name, motif_pattern in motifs_dict.items():
        key = f"{prefix}{motif_name}" if prefix else motif_name
        total = count_motif_occurrences(seq, motif_pattern)
        five, mid, three = get_positional_counts(seq, motif_pattern)
        features[f"{key}_total"]  = total
        features[f"{key}_5prime"] = five
        features[f"{key}_middle"] = mid
        features[f"{key}_3prime"] = three
    return features


# STREME motif parser

def parse_streme_motifs(streme_txt_path):
    """
    Parse STREME output file and extract motif consensus sequences.
    Translates IUPAC ambiguity codes into regex patterns.

    Arguments:
      streme_txt_path — path to streme.txt output file

    Returns:
      dictionary {motif_name: regex_pattern}
    """
    streme_txt = Path(streme_txt_path)

    if not streme_txt.exists():
        log.warning(f"STREME output not found: {streme_txt} — skipping")
        return {}

    motifs = {}

    with open(streme_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("MOTIF"):
                parts = line.split()
                
                if len(parts) >= 2:
                    motif_name = parts[1]
                    
                    # Extract consensus from motif name
                    # Example: "1-AAACGCCG" → "AAACGCCG"
                    if "-" in motif_name:
                        consensus = motif_name.split("-", 1)[1]
                    else:
                        consensus = motif_name

                    consensus = consensus.upper().replace("T", "U")
                    # Translate IUPAC to regex
                    regex = ""
                    for char in consensus:
                        regex += IUPAC_MAP.get(char, char)

                    motifs[motif_name] = regex
                    log.info(f"  STREME motif: {motif_name} = {regex}")

    log.info(f"  Parsed {len(motifs)} STREME motifs from {streme_txt.name}")
    return motifs


# TOMTOM RBP motif parser

def parse_tomtom_top_rbps(tomtom_tsv_path, top_n=10):
    """
    Parse TOMTOM output and extract top N RBP consensus sequences.

    Reads tomtom.tsv, counts how many STREME motifs matched each target RBP, takes the top N, and returns their consensus sequences.

    Arguments:
      tomtom_tsv_path — path to tomtom.tsv
      top_n           — number of top RBPs to take (default 10)

    Returns:
      dictionary {rbp_name: consensus_sequence_as_regex}
    """
    tomtom_tsv = Path(tomtom_tsv_path)

    if not tomtom_tsv.exists():
        log.warning(f"TOMTOM output not found: {tomtom_tsv} — skipping")
        return {}

    # Count hits per target RBP and store their consensus sequences
    rbp_counts    = defaultdict(int)
    rbp_consensus = {}

    with open(tomtom_tsv, "r", newline="") as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and header
            if line.startswith("#") or line.startswith("Query_ID"):
                continue
            if not line:
                continue

            parts = line.strip().split()
            
            if len(parts) < 9:
                log.warning(f"Skipping bad line: {parts}")
                continue

            target_id         = parts[1]
            
            #Extract RBP name after underscore
            rbp_name          = target_id.split("_")[-1]
            target_consensus  = parts[8]

            if not target_id or not target_consensus:
                continue

            rbp_counts[rbp_name]    += 1
            # Keep the consensus from the first hit seen
            if rbp_name not in rbp_consensus:
                rbp_consensus[rbp_name] = target_consensus

    if not rbp_counts:
        log.warning(f"  No RBP hits found in {tomtom_tsv.name}")
        return {}

    # Sort by hit count and take top N
    sorted_rbps = sorted(rbp_counts.items(),
                         key=lambda x: x[1], reverse=True)[:top_n]

    log.info(f"  Top {top_n} RBPs from {tomtom_tsv.name}:")
    motifs = {}
    for rbp_name, count in sorted_rbps:
        consensus = rbp_consensus[rbp_name]
        # Translate IUPAC to regex
        regex = ""
        for char in consensus:
            regex += IUPAC_MAP.get(char, char)

        # Clean RBP name for use as feature name
        clean_name = rbp_name
        motifs[clean_name] = regex
        log.info(f"    {rbp_name} (hits={count}): {consensus}")

    return motifs


# FASTA parser

def iter_fasta(fasta_path):
    """
    Generator — yields (name, sequence) tuples one at a time.
    Never accumulates sequences in memory.

    Arguments:
      fasta_path — path to FASTA file

    Yields:
      (name, sequence) tuples
    """
    fasta_path   = Path(fasta_path)
    current_name = None
    current_seq  = []

    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    seq = "".join(current_seq).upper().replace("T", "U")
                    yield current_name, seq
                current_name = line[1:].strip().split()[0]
                current_seq  = []
            elif line:
                current_seq.append(line)

    if current_name is not None:
        seq = "".join(current_seq).upper().replace("T", "U")
        yield current_name, seq


# Feature extraction 

def get_all_features(name, seq, label,
                     streme_inh_motifs, streme_non_motifs,
                     tomtom_inh_motifs, tomtom_non_motifs):
    """
    Compute all motif features for a single sequence.

    Arguments:
      name               — sequence name
      seq                — RNA sequence string
      label              — "inherited" or "non_inherited"
      streme_inh_motifs  — dict of STREME inherited motifs
      streme_non_motifs  — dict of STREME non-inherited motifs
      tomtom_inh_motifs  — dict of TOMTOM top RBPs from inherited
      tomtom_non_motifs  — dict of TOMTOM top RBPs from non-inherited

    Returns:
      dictionary of all features
    """
    features = {"name": name, "label": label, "length": len(seq)}

    # Source 1 — hand-curated
    features.update(scan_motif_group(seq, HYPOTHESIS_MOTIFS,
                                     prefix="hyp_"))

    # Source 2 — STREME inherited motifs
    features.update(scan_motif_group(seq, streme_inh_motifs,
                                     prefix="streme_inh_"))

    # Source 2 — STREME non-inherited motifs
    features.update(scan_motif_group(seq, streme_non_motifs,
                                     prefix="streme_non_"))

    # Source 3 — TOMTOM RBP motifs from inherited direction
    features.update(scan_motif_group(seq, tomtom_inh_motifs,
                                     prefix="rbp_inh_"))

    # Source 3 — TOMTOM RBP motifs from non-inherited direction
    features.update(scan_motif_group(seq, tomtom_non_motifs,
                                     prefix="rbp_non_"))

    return features


def build_header(streme_inh_motifs, streme_non_motifs,
                 tomtom_inh_motifs, tomtom_non_motifs):
    """Build the full CSV header from all motif sources."""
    header = ["name", "label", "length"]

    suffixes = ["_total", "_5prime", "_middle", "_3prime"]

    for motif_name in HYPOTHESIS_MOTIFS:
        for s in suffixes:
            header.append(f"hyp_{motif_name}{s}")

    for motif_name in streme_inh_motifs:
        for s in suffixes:
            header.append(f"streme_inh_{motif_name}{s}")

    for motif_name in streme_non_motifs:
        for s in suffixes:
            header.append(f"streme_non_{motif_name}{s}")

    for motif_name in tomtom_inh_motifs:
        for s in suffixes:
            header.append(f"rbp_inh_{motif_name}{s}")

    for motif_name in tomtom_non_motifs:
        for s in suffixes:
            header.append(f"rbp_non_{motif_name}{s}")

    return header


def extract_to_csv(fasta_path, label, writer, header,
                   streme_inh_motifs, streme_non_motifs,
                   tomtom_inh_motifs, tomtom_non_motifs):
    """
    Read FASTA and write motif features to CSV one sequence at a time.
    """
    processed = 0
    for name, seq in iter_fasta(fasta_path):
        features = get_all_features(
            name, seq, label,
            streme_inh_motifs, streme_non_motifs,
            tomtom_inh_motifs, tomtom_non_motifs
        )
        row = {k: features.get(k, 0) for k in header}
        writer.writerow(row)
        processed += 1

    log.info(f"  {label}: {processed} sequences processed")


# QC check

def qc_check(csv_path, label):
    """Basic QC on output CSV — checks row counts and NaN values."""
    import csv as csv_module
    with open(csv_path, "r") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)

    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    n_empty = sum(
        1 for row in rows
        for v in row.values()
        if v == "" or v is None
    )

    log.info(f"QC — {label}")
    log.info(f"  Rows:        {n_rows}")
    log.info(f"  Columns:     {n_cols}")
    log.info(f"  Empty cells: {n_empty}")

    if n_empty > 0:
        log.warning(f"  {n_empty} empty cells found — check output")


# Main

def extract_motif_features(inherited_train, non_inherited_train,
                            inherited_test, non_inherited_test,
                            streme_inh_dir, streme_non_dir,
                            tomtom_inh_tsv, tomtom_non_tsv,
                            output_dir="features_output",
                            top_n_rbp=10):
    """
    Extract all motif features from train and test FASTA files.

    Arguments:
      inherited_train     — path to inherited train FASTA
      non_inherited_train — path to non-inherited train FASTA
      inherited_test      — path to inherited test FASTA
      non_inherited_test  — path to non-inherited test FASTA
      streme_inh_dir      — path to STREME inherited output dir
      streme_non_dir      — path to STREME non-inherited output dir
      tomtom_inh_tsv      — path to TOMTOM inherited tomtom.tsv
      tomtom_non_tsv      — path to TOMTOM non-inherited tomtom.tsv
      output_dir          — where to save CSV files
      top_n_rbp           — number of top RBPs to use from TOMTOM

    Returns:
      train_csv, test_csv paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    train_csv = output_dir / "train_motifs.csv"
    test_csv  = output_dir / "test_motifs.csv"

    log.info("Extracting Motif Features")
    log.info(f"  Hand-curated motifs: {len(HYPOTHESIS_MOTIFS)}")

    # Parse all motif sources
    log.info("Parsing STREME motifs...")
    streme_inh_motifs = parse_streme_motifs(
        Path(streme_inh_dir) / "streme.txt"
    )
    streme_non_motifs = parse_streme_motifs(
        Path(streme_non_dir) / "streme.txt"
    )

    log.info("Parsing TOMTOM RBP motifs...")
    tomtom_inh_motifs = parse_tomtom_top_rbps(tomtom_inh_tsv, top_n_rbp)
    tomtom_non_motifs = parse_tomtom_top_rbps(tomtom_non_tsv, top_n_rbp)

    # Build header
    header = build_header(
        streme_inh_motifs, streme_non_motifs,
        tomtom_inh_motifs, tomtom_non_motifs
    )
    log.info(f"  Total features: {len(header) - 2}")

    # Train set
    log.info("Processing TRAIN sequences...")
    with open(train_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header,
                                extrasaction="ignore")
        writer.writeheader()
        extract_to_csv(
            inherited_train, "inherited", writer, header,
            streme_inh_motifs, streme_non_motifs,
            tomtom_inh_motifs, tomtom_non_motifs
        )
        extract_to_csv(
            non_inherited_train, "non_inherited", writer, header,
            streme_inh_motifs, streme_non_motifs,
            tomtom_inh_motifs, tomtom_non_motifs
        )
    log.info(f"  Train motifs saved to {train_csv}")
    qc_check(train_csv, "train_motifs")

    # Test set
    log.info("Processing TEST sequences...")
    with open(test_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header,
                                extrasaction="ignore")
        writer.writeheader()
        extract_to_csv(
            inherited_test, "inherited", writer, header,
            streme_inh_motifs, streme_non_motifs,
            tomtom_inh_motifs, tomtom_non_motifs
        )
        extract_to_csv(
            non_inherited_test, "non_inherited", writer, header,
            streme_inh_motifs, streme_non_motifs,
            tomtom_inh_motifs, tomtom_non_motifs
        )
    log.info(f"  Test motifs saved to {test_csv}")
    qc_check(test_csv, "test_motifs")

    log.info("Motif feature extraction complete")
    return train_csv, test_csv


# Run

if __name__ == "__main__":
    import time
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Extract motif features from FASTA files"
    )
    parser.add_argument("--inherited_train",     type=str, required=True)
    parser.add_argument("--non_inherited_train", type=str, required=True)
    parser.add_argument("--inherited_test",      type=str, required=True)
    parser.add_argument("--non_inherited_test",  type=str, required=True)
    parser.add_argument("--streme_inh_dir",      type=str,
                        default="streme_output_inherited",
                        help="STREME output dir for inherited")
    parser.add_argument("--streme_non_dir",      type=str,
                        default="streme_output_noninherited",
                        help="STREME output dir for non-inherited")
    parser.add_argument("--tomtom_inh_tsv",      type=str,
                        default="tomtom_output_inherited_v2/tomtom.tsv",
                        help="TOMTOM tsv for inherited direction")
    parser.add_argument("--tomtom_non_tsv",      type=str,
                        default="tomtom_output_noninherited_v2/tomtom.tsv",
                        help="TOMTOM tsv for non-inherited direction")
    parser.add_argument("--outdir",              type=str,
                        default="features_output")
    parser.add_argument("--top_n_rbp",           type=int, default=10,
                        help="Number of top RBPs from TOMTOM to use")
    args = parser.parse_args()

    extract_motif_features(
        inherited_train=args.inherited_train,
        non_inherited_train=args.non_inherited_train,
        inherited_test=args.inherited_test,
        non_inherited_test=args.non_inherited_test,
        streme_inh_dir=args.streme_inh_dir,
        streme_non_dir=args.streme_non_dir,
        tomtom_inh_tsv=args.tomtom_inh_tsv,
        tomtom_non_tsv=args.tomtom_non_tsv,
        output_dir=args.outdir,
        top_n_rbp=args.top_n_rbp
    )

    log.info(f"Total runtime: {time.time() - start:.2f} seconds")
