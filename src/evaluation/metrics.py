import numpy as np
import pandas as pd


# -------------------------
# PRECISION@K
# -------------------------
def precision_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(relevant))

    return hits / k


# -------------------------
# RECALL@K
# -------------------------
def recall_at_k(recommended, relevant, k=10):
    recommended_k = recommended[:k]

    hits = len(set(recommended_k) & set(relevant))

    return hits / len(relevant) if relevant else 0


# -------------------------
# COVERAGE (DIVERSITY)
# -------------------------
def genre_coverage(recommended_df):
    genres = []

    for g in recommended_df["genres"]:
        genres.extend(g.split("|"))

    unique_genres = set(genres)

    return len(unique_genres)