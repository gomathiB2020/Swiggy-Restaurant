import pandas as pd
import pickle

# Load encoded data
df = pd.read_csv("encoded_data.csv")

# Ensure all values are numeric (extra safety)
df = df.apply(pd.to_numeric, errors='coerce')

# Replace any remaining NaN with 0
df.fillna(0, inplace=True)

# Save as pickle
with open("encoded_data.pkl", "wb") as f:
    pickle.dump(df, f)

print("encoded_data.pkl saved successfully")
