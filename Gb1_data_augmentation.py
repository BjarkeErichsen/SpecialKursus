"""
ENVIRONMENT = esm_env

Purpose:
Takes the existing csv and reference for Gb1 and produces a new csv containing all the same info as the old one but now with
the full sequence of the Gb1 protein instead of simply the variant.
"""

import pandas as pd
from Bio import SeqIO

def modify_sequence(ref_seq, variant):
    positions = [39, 40, 41, 54]  # 1-indexed positions
    ref_seq_list = list(ref_seq[2:])  # Convert string to list for easier modification, exclude first 2 AAs

    for i, pos in enumerate(positions):
        ref_seq_list[pos - 1] = variant[i]  # Modify the amino acid, adjust for the 2 missing AAs

    return ''.join(ref_seq_list)  # Convert list back to string

# Paths
path_to_reference = "/home/bjarke/Desktop/Data/DMS/2GI9.fasta"
path_to_csv = "/home/bjarke/Desktop/Data/DMS/Gb1Dataset.csv"
out_putdir = "/home/bjarke/Desktop/Data/DMS/project/"

# Read the reference sequence
with open(path_to_reference, "r") as fasta_file:
    ref_seq_record = next(SeqIO.parse(fasta_file, "fasta"))
    ref_seq = str(ref_seq_record.seq)

# Print the reference sequence, excluding the first 2 amino acids
print("Reference Sequence:", ref_seq)

# Read the CSV file
df = pd.read_csv(path_to_csv)

# Initialize a list to store the new rows
new_rows = []

# Modify the sequences based on variants and add them to the list
for index, row in df.iterrows():
    variant = row['Variants']
    new_seq = modify_sequence(ref_seq, variant)
    new_row = row.copy()
    new_row['Variants'] = new_seq
    new_rows.append(new_row)

# Create a new DataFrame using the list of new rows
new_df = pd.DataFrame(new_rows)

# Save the new DataFrame to a new CSV file
new_csv_path = out_putdir + "/ModifiedSequences.csv"
new_df.to_csv(new_csv_path, index=False)


