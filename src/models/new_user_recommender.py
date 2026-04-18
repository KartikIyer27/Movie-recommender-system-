import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class NewUserRecommender:
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

    def _genre_score(self, genres, preferred_genres):
        if not genres or not preferred_genres:
            return 0
        genre_list = genres.split("|")
        match = sum(1 for g in genre_list if g in preferred_genres)
        return match / len(preferred_genres)

    def _liked_movie_score(self, idx, liked_indices):
        if not liked_indices:
            return 0
        return max([
            linear_kernel(
                self.tfidf_matrix[idx:idx+1],
                self.tfidf_matrix[i:i+1]
            )[0][0]
            for i in liked_indices
        ])

    def _mood_score(self, genres, mood):
        mood_map = {
            "happy": ["Comedy", "Animation", "Family"],
            "serious": ["Drama", "Crime"],
            "excited": ["Action", "Thriller", "Adventure"]
        }

        if mood not in mood_map:
            return 0

        genre_list = genres.split("|") if genres else []
        return 1 if any(g in genre_list for g in mood_map[mood]) else 0

    def recommend(self, preferred_genres, liked_movies, mood, top_n=10):
        df = self.movie_features.copy()

        liked_indices = [
            self.title_to_index[m]
            for m in liked_movies
            if m in self.title_to_index
        ]

        scores = []
        user_preference_weight = len(preferred_genres) / 10 if preferred_genres else 1

        for idx, row in df.iterrows():
            score = (
                0.4 * self._genre_score(row["genres"], preferred_genres) +
                0.3 * self._liked_movie_score(idx, liked_indices) +
                0.2 * row["popularity_score"] +
                0.1 * self._mood_score(row["genres"], mood)
            )
            scores.append(score)

        df["final_score"] = scores
        df = df[~df["title"].isin(liked_movies)]

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