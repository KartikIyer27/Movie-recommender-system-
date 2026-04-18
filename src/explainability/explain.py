def generate_explanation(
    movie_genres: str,
    mood: str,
    recent_movies: list,
    recommended_title: str,
    movie_features_df,
    preferred_genres: list = None
):
    reasons = []

    movie_genres_lower = movie_genres.lower() if movie_genres else ""

    # -------------------------
    # GET MOVIE ROW
    # -------------------------
    movie_row = movie_features_df[
        movie_features_df["title"] == recommended_title
    ]

    popularity_score = 0
    if not movie_row.empty:
        popularity_score = movie_row.iloc[0]["popularity_score"]

    # -------------------------
    # ⭐ MUST WATCH LOGIC
    # -------------------------
    if preferred_genres:
        if not any(g.lower() in movie_genres_lower for g in preferred_genres):
            if popularity_score > 5:
                return "⭐ Must Watch: Highly rated and popular movie"

    # -------------------------
    # HISTORY-BASED
    # -------------------------
    if recent_movies:
        last_movie = recent_movies[-1]

        last_row = movie_features_df[
            movie_features_df["title"] == last_movie
        ]

        if not last_row.empty:
            last_genres = last_row.iloc[0]["genres"].lower()

            if any(g in movie_genres_lower for g in last_genres.split("|")):
                reasons.append(f"similar to '{last_movie}'")

    # -------------------------
    # MOOD-BASED
    # -------------------------
    mood_map = {
        "happy": ["comedy", "animation", "family"],
        "serious": ["drama", "crime"],
        "excited": ["action", "thriller", "adventure"]
    }

    if mood in mood_map:
        if any(g in movie_genres_lower for g in mood_map[mood]):
            reasons.append(f"matches your {mood} mood")

    # -------------------------
    # GENRE FALLBACK
    # -------------------------
    if not reasons:
        if movie_genres:
            main_genre = movie_genres.split("|")[0]
            reasons.append(f"belongs to {main_genre} genre")

    return "Recommended because it " + ", ".join(reasons) + "."