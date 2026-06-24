import pandas as pd

def load_sequence_names(fasta_path):
    """
    Extract sequence names from FASTA file
    """
    names = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                names.append(line.strip()[1:])
    return pd.DataFrame({"sequence_name": names})


def process_fimo_file(fimo_path):
    """
    Convert a FIMO TSV file into a sequence × motif count table
    """
    # Read file (skip comment lines starting with '#')
    df = pd.read_csv(fimo_path, sep="\t", comment="#")

    # Count hits per sequence per motif
    counts = (
        df.groupby(["sequence_name", "motif_id"])
        .size()
        .reset_index(name="count")
    )

    # Pivot to wide format
    matrix = counts.pivot(index="sequence_name",
                          columns="motif_id",
                          values="count").fillna(0)

    # Flatten column names
    matrix.columns = [f"fimo_{col}_count" for col in matrix.columns]

    # Reset index
    matrix = matrix.reset_index()

    return matrix

def build_fimo_feature_matrix():

    # ---- Load ALL sequence names (IMPORTANT FIX) ----
    inh_names = load_sequence_names("inherited_train_rna.fasta")
    non_names = load_sequence_names("non_inherited_train_rna.fasta")

    # ---- Process FIMO outputs ----
    inh_on_inh = process_fimo_file("fimo_inh_on_inh/fimo.tsv")
    inh_on_non = process_fimo_file("fimo_inh_on_non/fimo.tsv")
    non_on_inh = process_fimo_file("fimo_non_on_inh/fimo.tsv")
    non_on_non = process_fimo_file("fimo_non_on_non/fimo.tsv")

    # ---- Merge motif sets (horizontal merge) ----
    inherited_df = pd.merge(inh_on_inh, non_on_inh,
                            on="sequence_name", how="outer")

    non_inherited_df = pd.merge(inh_on_non, non_on_non,
                                on="sequence_name", how="outer")

    # ---- FORCE all sequences back in (critical fix) ----
    inherited_df = pd.merge(inh_names, inherited_df,
                            on="sequence_name", how="left")

    non_inherited_df = pd.merge(non_names, non_inherited_df,
                                on="sequence_name", how="left")

    # ---- Fill missing motif counts with 0 ----
    inherited_df = inherited_df.fillna(0)
    non_inherited_df = non_inherited_df.fillna(0)

    # ---- Add labels ----
    inherited_df["label"] = "inherited"
    non_inherited_df["label"] = "non_inherited"

    # ---- Combine ----
    full_df = pd.concat([inherited_df, non_inherited_df],
                        ignore_index=True)

    return full_df


if __name__ == "__main__":

    print("Building FIMO feature matrix...")

    df = build_fimo_feature_matrix()

    # Save output
    output_path = "fimo_features_train.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")

    # -------- QC checks --------

    print("\n=== QC CHECKS ===")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    print("\nAny missing values?")
    print(df.isna().sum().sum())

    print("\nSample rows:")
    print(df.head())

