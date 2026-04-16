import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Movie dataset
# --------------------------
movies = pd.DataFrame({
    'Movie': ['M1', 'M2', 'M3', 'M4'],
    'Description': [
        'action thriller war hero mission',
        'romantic love drama school life',
        'action spy mission secret agent',
        'comedy family fun school friends'
    ]
})

# --------------------------
# TF-IDF Vectorization
# --------------------------
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['Description'])

# --------------------------
# Cosine similarity
# --------------------------
similarity = cosine_similarity(tfidf_matrix)

similarity_df = pd.DataFrame(similarity, index=movies['Movie'], columns=movies['Movie'])

print("\nMovie Similarity Matrix:\n")
print(similarity_df)

# --------------------------
# Recommend similar movies to M1
# --------------------------
movie = 'M1'
scores = similarity_df[movie].sort_values(ascending=False)

print(f"\nMovies similar to {movie}:\n")
print(scores)