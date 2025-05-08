# Step 1: Imports
from datasets import load_dataset
import pandas as pd
import duckdb

# Step 2: Load raw dataset from Hugging Face
dataset = load_dataset("knowledgator/Scientific-text-classification", split="train")

# Step 3: Convert to pandas DataFrame
df_raw = dataset.to_pandas()

# Step 4: Save raw dataset
df_raw.to_csv("data/raw/raw_dataset.csv", index=False)
print("Raw dataset saved as 'raw_dataset.csv'")

# Step 5: Use SQL to filter rows where label appears more than 4000 times
query = """
WITH label_counts AS (
    SELECT label, COUNT(*) AS label_count
    FROM df_raw
    GROUP BY label
    HAVING COUNT(*) > 4000
)
SELECT r.*
FROM df_raw r
JOIN label_counts lc
ON r.label = lc.label
ORDER BY lc.label_count DESC;
"""

df_filtered = duckdb.query(query).to_df()

# Step 6: Save processed (filtered) dataset
df_filtered.to_csv("data/processed/processed_dataset.csv", index=False)
print("Processed dataset with full rows saved as 'processed_dataset.csv'")

