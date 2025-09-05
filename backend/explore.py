import pandas as pd
import streamlit as st

def load_data(file_path):
    """Load data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

googlereview_df = load_data('data/GoogleReview_data_cleaned.csv')
tripadvisor_df = load_data('data/TripAdvisor_data_cleaned.csv')

st.write(googlereview_df.head())
st.write(tripadvisor_df.head()) 

st.write(googlereview_df.columns)
st.write(tripadvisor_df.columns)

#concatenate the df on Review, Rating, Restaurant, Location, drop other columns
combined_df = pd.concat([
    googlereview_df[['Review', 'Rating', 'Restaurant', 'Location']],
    tripadvisor_df[['Review', 'Rating', 'Restaurant', 'Location']]
], ignore_index=True)

st.write(combined_df.head(50))
st.write(combined_df.columns)
st.write(combined_df.shape)

#group by Restaurant and Location, concat the reviews, count the number of reviews and average the rating
#KeyError: "Column(s) ['Count'] do not exist"
grouped_df = combined_df.groupby(['Restaurant', 'Location']).agg({
    'Review': ' '.join,
    'Rating': 'mean'
}).reset_index()
grouped_df['Count'] = combined_df.groupby(['Restaurant', 'Location']).size().values

st.write(grouped_df.head(50))
st.write(grouped_df.columns)
st.write(grouped_df.shape)

###The weighted rating is calculated by taking into account both the average rating and the number of reviews. It helps to balance the influence of highly rated restaurants with few reviews and lower rated restaurants with many reviews.
C = grouped_df['Rating'].mean()
m = grouped_df['Count'].quantile(0.90)
grouped_df['weightedRating'] = (grouped_df['Rating'] * grouped_df['Count'] + C * m) / (grouped_df['Count'] + m)

#save to csv
grouped_df.to_csv('data/combined_restaurant_reviews.csv', index=False)

#save a smaller version to csv with only 5 rows
grouped_df.head(5).to_csv('data/combined_restaurant_reviews_sample.csv', index=False)


#show all data in Penang
penang_df = grouped_df[grouped_df['Location'].str.contains('KL', case=False, na=False)]
#sort by weightedRating descending
penang_df = penang_df.sort_values(by='weightedRating', ascending=False).reset_index(drop=True)
st.write(penang_df.head(50))
st.write(penang_df.shape)

