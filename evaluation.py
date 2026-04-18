import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from src.models.existing_user_recommender import ExistingUserRecommender


# -------------------------
# CONFIG
# -------------------------
NUM_USERS = 30              # more users = better evaluation
TOP_K = 10
MIN_HISTORY = 6
SAMPLE_SIZE = 25000         # larger subset (balanced approach)


# -------------------------
# LOAD FULL DATASET
# -------------------------
movie_features_full = pd.read_csv("data/processed/movie_features.csv")

print(f"Total movies in dataset: {len(movie_features_full)}")

# Use larger sample for realistic evaluation
movie_features = movie_features_full.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

model = ExistingUserRecommender(movie_features)


# -------------------------
# HELPERS
# -------------------------
def get_genres(title, df):
    row = df[df["title"] == title]
    if row.empty:
        return set()
    genres = row.iloc[0]["genres"]
    if pd.isna(genres):
        return set()
    return set(genres.split("|"))


def is_relevant(rec_movie, test_movies, df):
    rec_genres = get_genres(rec_movie, df)

    for t in test_movies:
        test_genres = get_genres(t, df)

        if rec_genres.intersection(test_genres):
            return True

    return False


def precision_at_k(recommended, test_movies, df, k):
    recommended = recommended[:k]

    hits = sum(1 for m in recommended if is_relevant(m, test_movies, df))
    return hits / k


def recall_at_k(recommended, test_movies, df, k):
    found = 0

    for t in test_movies:
        t_genres = get_genres(t, df)

        for m in recommended:
            if t_genres.intersection(get_genres(m, df)):
                found += 1
                break

    return found / len(test_movies)


def genre_coverage(df_rec):
    genres = []
    for g in df_rec["genres"]:
        genres.extend(g.split("|"))
    return len(set(genres))


# -------------------------
# CREATE USERS FROM FULL DATASET
# -------------------------
all_movies = movie_features_full["title"].dropna().tolist()

users = []

for _ in range(NUM_USERS):
    history = random.sample(all_movies, MIN_HISTORY)
    users.append(history)


# -------------------------
# EVALUATION
# -------------------------
precision_scores = []
recall_scores = []
coverage_scores = []

start = time.time()

for i, history in enumerate(users):
    print(f"Processing user {i+1}/{NUM_USERS}")

    train = history[:3]
    test = history[3:]

    recs = model.recommend(train, mood="happy", top_n=TOP_K)

    recommended_titles = recs["title"].tolist()

    p = precision_at_k(recommended_titles, test, movie_features, TOP_K)
    r = recall_at_k(recommended_titles, test, movie_features, TOP_K)
    c = genre_coverage(recs)

    precision_scores.append(p)
    recall_scores.append(r)
    coverage_scores.append(c)


# -------------------------
# FINAL RESULTS
# -------------------------
print("\n========== FINAL RESULTS ==========")
print(f"Average Precision@10: {round(sum(precision_scores)/len(precision_scores), 3)}")
print(f"Average Recall@10: {round(sum(recall_scores)/len(recall_scores), 3)}")
print(f"Average Genre Coverage: {round(sum(coverage_scores)/len(coverage_scores), 2)}")

print(f"\nTime taken: {round(time.time() - start, 2)} seconds")


# -------------------------
# PLOTS
# -------------------------
plt.figure()
plt.plot(precision_scores, label="Precision@10")
plt.plot(recall_scores, label="Recall@10")
plt.legend()
plt.title("Model Performance (Full Dataset Evaluation)")
plt.xlabel("Users")
plt.ylabel("Score")
plt.grid()

plt.savefig("full_precision_recall.png")
plt.show()


plt.figure()
plt.plot(coverage_scores, label="Genre Coverage", color="green")
plt.legend()
plt.title("Genre Diversity")
plt.xlabel("Users")
plt.ylabel("Unique Genres")
plt.grid()

plt.savefig("full_coverage.png")
plt.show()