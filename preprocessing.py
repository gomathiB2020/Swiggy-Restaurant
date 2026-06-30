import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("swiggy.csv")

# Replace invalid strings with NaN
df.replace(['--', 'NEW', '-'], pd.NA, inplace=True)

# Convert numeric columns
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

# Fill missing numeric values
if df['rating'].notna().sum() > 0:
    df['rating'].fillna(df['rating'].mean(), inplace=True)
else:
    df['rating'].fillna(0, inplace=True)
df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
df['cost'].fillna(df['cost'].median(), inplace=True)

# Drop rows only if city or cuisine missing
df.dropna(subset=['city', 'cuisine'], inplace=True)

print("After Cleaning:", df.shape)

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

# One Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Reduce cuisine complexity (ONLY FIRST CUISINE)
df['cuisine'] = df['cuisine'].apply(lambda x: str(x).split(',')[0])

encoded = encoder.fit_transform(df[['city', 'cuisine']])

encoded_df = pd.DataFrame(encoded, index=df.index)

# Combine all features
final_df = pd.concat([
    df[['rating', 'rating_count', 'cost']],
    encoded_df
], axis=1)

# FINAL SAFETY STEP (IMPORTANT)
final_df.fillna(0, inplace=True)

# Save encoded data
final_df.to_csv("encoded_data.csv", index=False)

# Save encoder
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("encoder.pkl created")
print("encoded_data.csv created")
print("Preprocessing completed successfully")
