import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- LOAD ---------------- #

cleaned_df = pd.read_csv("cleaned_data.csv")
encoded_df = pickle.load(open("encoded_data.pkl","rb"))

# ---------------- BASIC CLEAN ---------------- #

cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
cleaned_df['cuisine'] = cleaned_df['cuisine'].astype(str)

# ---------------- UI ---------------- #

st.title("🍽 Swiggy Restaurant Recommendation System")

# ---------------- CITY ---------------- #

city = st.selectbox(
    "Select City",
    sorted(cleaned_df['city'].dropna().unique())
)

filtered_df = cleaned_df[cleaned_df['city'] == city]

# ---------------- CUISINE ---------------- #

cuisine = st.selectbox(
    "Select Cuisine",
    sorted(set(
        item.strip().lower()
        for c in filtered_df['cuisine']
        for item in str(c).split(',')
    ))
)

# Apply cuisine filter safely
temp_df = filtered_df[
    filtered_df['cuisine'].str.lower().str.contains(cuisine, na=False)
]

if not temp_df.empty:
    filtered_df = temp_df

# ---------------- COST (FINAL FIX) ---------------- #

valid_costs = cleaned_df['cost'].dropna()

# If dataset cost is bad → fallback values
if valid_costs.empty:
    min_cost = 100
    max_cost = 1000
else:
    min_cost = int(valid_costs.min())
    max_cost = int(valid_costs.max())

# Prevent slider crash
if min_cost == max_cost:
    max_cost = min_cost + 1

cost_range = st.slider(
    "Select Cost Range",
    min_cost,
    max_cost,
    (min_cost, max_cost)
)

# Apply cost filter safely
temp_df = filtered_df[
    (filtered_df['cost'] >= cost_range[0]) &
    (filtered_df['cost'] <= cost_range[1])
]

# If empty → ignore cost filter
if not temp_df.empty:
    filtered_df = temp_df

# ---------------- RESTAURANT ---------------- #

restaurant = st.selectbox(
    "Select Restaurant",
    filtered_df['name'].dropna().unique()
)

# ---------------- RECOMMEND ---------------- #

def recommend(name):

    index = cleaned_df[cleaned_df['name'] == name].index[0]

    similarity = cosine_similarity(
        [encoded_df.iloc[index]],
        encoded_df
    )

    distances = list(enumerate(similarity[0]))

    restaurants = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    return [cleaned_df.iloc[i[0]] for i in restaurants]

# ---------------- BUTTON ---------------- #

# ---------------- BUTTON ---------------- #

if st.button("Recommend"):

    results = recommend(restaurant)

    st.subheader("Top 5 Recommendations")

    for r in results:
        st.write("🍴", r['name'])
        st.write("📍", r['city'])
        st.write("🍽", r['cuisine'])
        st.write("⭐", round(r['rating'], 2))

        cost = r['cost']
        if pd.isna(cost):
            cost = "Not Available"

        st.write("💰", cost)
        st.write("---")
