"""
ENVIRONMENT = esm_env

The dataset produced can be used for
1 The train/test dataset for the GP deep sequence training.
2 The esm + supervised model
Note that deep sequence itself is trained using an MSA not the Gb1 fitness values.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_data_splits(input_csv_path, output_dir, test_size=0.1, val_size=0.1, random_state=None):
    """
    Splits the data from the input CSV file into training, validation, and testing sets,
    and saves them into the specified output directory.

    :param input_csv_path: Path to the input CSV file.
    :param output_dir: Directory to save the output CSV files.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the dataset to include in the validation split.
    :param random_state: Controls the shuffling applied to the data before the split.
    """

    # Check if output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the data
    data = pd.read_csv(input_csv_path)

    # First split to separate out the training set
    initial_train_size = 1 - test_size - val_size
    train_set, temp_test_set = train_test_split(data, test_size=test_size + val_size, random_state=random_state)

    # Adjust validation size for the second split
    adjusted_val_size = val_size / (test_size + val_size)

    # Second split to separate out the test and validation sets
    val_set, test_set = train_test_split(temp_test_set, test_size=1 - adjusted_val_size, random_state=random_state)

    # Save the split data
    train_set.to_csv(os.path.join(output_dir, "train_set.csv"), index=False)
    val_set.to_csv(os.path.join(output_dir, "validation_set.csv"), index=False)
    test_set.to_csv(os.path.join(output_dir, "test_set.csv"), index=False)

    print(f"Train, validation, and test sets saved in {output_dir}")

# Example usage
path_to_csv = "/home/bjarke/Desktop/Data/DMS/project/ModifiedSequences.csv"
output_directory = "/home/bjarke/Desktop/Data/DMS/project/train_test_splits/"

create_data_splits(path_to_csv, output_directory, test_size=0.8, val_size=0.1, random_state=42)  # Adjust the sizes and random state as needed
