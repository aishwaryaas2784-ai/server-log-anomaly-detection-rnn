
import pandas as pd
import re

logs = pd.read_csv("data/HDFS_100k.log_structured.csv")
labels = pd.read_csv("data/HDFS.anomaly_label.csv")

print("LOG COLUMNS:", logs.columns.tolist())
print("LABEL COLUMNS:", labels.columns.tolist())

print("\nSample log content:")
print(logs["Content"].head(10).to_string(index=False))

print("\nSample labels:")
print(labels.head(10).to_string(index=False))

# Check how many log rows contain block IDs.
block_pattern = r"blk_-?\d+"
matched = logs["Content"].astype(str).str.contains(
    block_pattern, regex=True
)

print("\nLog rows containing block IDs:", int(matched.sum()))
print("Total log rows:", len(logs))

print("\nSample extracted block IDs:")
for content in logs.loc[matched, "Content"].head(5):
    print(re.findall(block_pattern, str(content)))
