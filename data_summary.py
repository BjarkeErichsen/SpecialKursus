"""
ENVIRONMENT = esm_env

Purpose:
Produces some summary plots
"""

import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Create a histogram for 'Fitness'
    plt.figure(figsize=(10, 5))
    plt.hist(df['Fitness'], bins=50, color='green', edgecolor='black')
    plt.xlabel('Fitness')
    plt.ylabel('Number of Variants LOG')
    plt.yscale('log')
    plt.title('Distribution of Fitness')
    plt.tight_layout()
    plt.show()

    # Create a histogram for 'Count input'
    plt.figure(figsize=(10, 5))
    plt.hist(df['Count input'], bins=50, color='blue', edgecolor='black')
    plt.xlabel('Count Input')
    plt.ylabel('Number of Variants LOG')
    plt.yscale('log')
    plt.title('Distribution of Count Input')
    plt.tight_layout()
    plt.show()

    # Create a histogram for 'Count selected'
    plt.figure(figsize=(10, 5))
    plt.hist(df['Count selected'], bins=50, color='red', edgecolor='black')
    plt.xlabel('Count Selected')
    plt.ylabel('Number of Variants LOG')
    plt.yscale('log')
    plt.title('Distribution of Count Selected')
    plt.tight_layout()
    plt.show()


path_gb1 = "/home/bjarke/Desktop/Data/DMS/Gb1Dataset.csv"
plot_histogram(path_gb1)








