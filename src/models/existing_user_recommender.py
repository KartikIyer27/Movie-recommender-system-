import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ExistingUserRecommender:
    def __init__(self, movie_features: pd.DataFrame):
        self.movie_features = movie_features.copy()

        self.movie_features["combined"] = (
            self.movie_features["title"].fillna("") + " " +
            self.movie_features["genres"].fillna("")
        )

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movie_features["combined"])

        self.title_to_index = pd.Series(
            self.movie_features.index,
            index=self.movie_features["title"]
        ).drop_duplicates()

    def _get_recent_genres(self, recent_movies):
        genres = []

        for m in recent_movies:
            if m in self.title_to_index:
                idx = self.title_to_index[m]
                genres.extend(self.movie_features.iloc[idx]["genres"].split("|"))

        return [g for g, _ in Counter(genres).most_common(3)]

    def _similarity(self, idx, recent_indices):
        if not recent_indices:
            return 0

        return max([
            linear_kernel(
                self.tfidf_matrix[idx:idx+1],
                self.tfidf_matrix[r:r+1]
            )[0][0]
            for r in recent_indices
        ])

    def recommend(self, recent_movies, mood, top_n=10):
        df = self.movie_features.copy()

        recent_indices = [
            self.title_to_index[m]
            for m in recent_movies
            if m in self.title_to_index
        ]

        preferred_genres = self._get_recent_genres(recent_movies)

        scores = []

        for idx, row in df.iterrows():
            genre_match = 1 if any(g in row["genres"] for g in preferred_genres) else 0
            user_preference_weight = len(preferred_genres) / 5 if preferred_genres else 1

            score = (
                0.35 * self._similarity(idx, recent_indices) +
                0.25 * genre_match +
                0.25 * row["popularity_score"]
            )

            scores.append(score)

        df["final_score"] = scores
        df = df[~df["title"].isin(recent_movies)]

        # -------------------------
        # DIVERSITY PENALTY
        # -------------------------
        df = df.sort_values("final_score", ascending=False).head(100)

        temp_rows = []
        genre_count = {}

        for _, row in df.iterrows():
            genres = row["genres"].split("|")

            penalty = sum(genre_count.get(g, 0) for g in genres)
            adjusted_score = row["final_score"] - (0.05 * penalty * user_preference_weight)

            row_copy = row.copy()
            row_copy["adjusted_score"] = adjusted_score

            temp_rows.append(row_copy)

        df_diverse = pd.DataFrame(temp_rows)
        df_diverse = df_diverse.sort_values("adjusted_score", ascending=False)

        final_selection = []
        genre_count = {}

        for _, row in df_diverse.iterrows():
            if len(final_selection) >= top_n:
                break

            genres = row["genres"].split("|")

            for g in genres:
                genre_count[g] = genre_count.get(g, 0) + 1

            final_selection.append(row["title"])

        return df_diverse[df_diverse["title"].isin(final_selection)].head(top_n)