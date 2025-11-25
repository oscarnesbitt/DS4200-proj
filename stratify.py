import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# usage for this file: python/python3 stratify.py input_file.csv output_file.csv

# assign arguments for files
inp, outp = sys.argv[1], sys.argv[2]

# load df with cleaned csv data
df = pd.read_csv(inp)

# check for 'income' column
if 'income' not in df.columns:
    print("Error: no 'income' column found.")
    sys.exit(1)

# create stratified sample dataset w/ 10,000 values
n = min(10000, len(df))
sample, _ = train_test_split(
    df,
    train_size=n,
    stratify=df['income'],
    random_state=42
)

# save sample to new csv
sample.to_csv(outp, index=False)
print(f"✔ Stratified sample saved to: {outp}")

# verify stratfication worked (proportion of datapoints stayed the same based on 'income' feature)
print("\nOriginal dataset distribution:")
print(df['income'].value_counts(normalize=True).round(3))
print("\nUpdated dataset distribution:")
print(sample['income'].value_counts(normalize=True).round(3))
