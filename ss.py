import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url = "https://github.com/Francis-Robina/Recommendation_system/raw/refs/heads/main/novels.csv"
books = pd.read_csv(url)


books['combined'] = books['genre'] + " " + books['description']

# Converting Text into TF-IDF Vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined'])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend(book_title, num_recommendations=5):
    if book_title not in books['title'].values:
        return "Book not found in the database."

    idx = books[books['title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][['title', 'author', 'genre']]



# Ask user for a book title and recommend similar books
user_input = input("\nEnter a book title from the list: ")
recommendations = recommend(user_input, num_recommendations=5)
print("\nRecommended Books:\n", recommendations)
