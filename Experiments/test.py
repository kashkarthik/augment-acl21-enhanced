import pandas as pd
from mr_generic_scripts import load_combined_data
# data = load_combined_data()


original_data = pd.read_csv("../Data/original_dataset.csv")
augmented_data = pd.read_csv("../Data/augmented_dataset.csv")
combined_data = load_combined_data()

print("Original Data Rows:", len(original_data))
print("Augmented Data Rows:", len(augmented_data))
print("Combined Data Rows:", len(combined_data))
# print(data.head(10))  # Display the first 10 rows
# print(data.tail(10))  # Display the last 10 rows