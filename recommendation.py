import pandas as pd
import pickle

# Just save encoded data, DO NOT compute similarity matrix

df = pd.read_csv("encoded_data.csv")

pickle.dump(df, open("encoded_data.pkl", "wb"))

print("encoded_data.pkl saved successfully")
