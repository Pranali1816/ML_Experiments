import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Sample user-item ratings
# --------------------------
data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
    'Movie': ['M1', 'M2', 'M3', 'M1', 'M3', 'M2', 'M3', 'M4'],
    'Rating': [5, 3, 4, 4, 5, 2, 5, 4]
}

df = pd.DataFrame(data)

# --------------------------
# Create User-Item matrix
# --------------------------
matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

print("\nUser-Item Matrix:\n")
print(matrix)

# --------------------------
# Similarity between users
# --------------------------
similarity = cosine_similarity(matrix)
similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)

print("\nUser Similarity:\n")
print(similarity_df)

# --------------------------
# Recommend for User A
# --------------------------
user = 'A'
scores = similarity_df[user].dot(matrix) / similarity_df[user].sum()

# remove already rated items
already_rated = matrix.loc[user]
recommendations = scores[already_rated == 0]

print("\nRecommendations for User A:\n")
print(recommendations.sort_values(ascending=False))