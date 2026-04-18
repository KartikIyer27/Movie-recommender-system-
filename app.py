import time
import pandas as pd
import streamlit as st

from src.models.new_user_recommender import NewUserRecommender
from src.models.existing_user_recommender import ExistingUserRecommender
from src.realtime.update_model import get_user_preferences

from src.utils.time_utils import get_time_of_day
from src.explainability.explain import generate_explanation
from src.realtime.update_model import (
    add_clicked_movie,
    get_recent_history,
    user_exists,
    create_user,
)


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# -------------------------
# LOAD DATA (CACHED)
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/movie_features.csv")


# -------------------------
# LOAD MODELS (CACHED)
# -------------------------
@st.cache_resource
def load_models(movie_features):
    return (
        NewUserRecommender(movie_features),
        ExistingUserRecommender(movie_features)
    )


movie_features = load_data()
new_user_model, existing_user_model = load_models(movie_features)


def build_explanations(recs, mood, recent_movies, preferred_genres=None):
    """Build a stable explanation map for the current recommendation list."""
    explanations = {}
    for row in recs.head(10).itertuples():
        explanations[row.title] = generate_explanation(
            movie_genres=row.genres,
            mood=mood,
            recent_movies=recent_movies,
            recommended_title=row.title,
            movie_features_df=movie_features,
            preferred_genres=preferred_genres
        )
    return explanations


# -------------------------
# HELPERS
# -------------------------
def animated_progress(message: str):
    with st.spinner(message):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.003)
            progress.progress(i + 1)


def show_history_section(user_id: str):
    recent_movies = get_recent_history(user_id)

    st.subheader("📺 Previously Watched")

    if recent_movies:
        for movie in reversed(recent_movies):
            st.markdown(
                f"""
                <div style="padding:10px; margin:5px 0; border-radius:10px; background-color:#262730;">
                    🎬 {movie}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No watch history yet. Start watching to build your profile!")

    return recent_movies


# -------------------------
# WELCOME SECTION
# -------------------------
st.markdown("""
# 🎬 Welcome to Movie Recommendation System

### 👨‍💻 Developed by:
- S Kartik Iyer
- Ravi Sharma
- Satyam Pal

---
""")

st.success("🚀 Smart Recommendation System Ready")

mood = st.selectbox("Select your mood", ["happy", "serious", "excited"])
time_of_day = get_time_of_day()
st.write(f"🕒 Detected Time: **{time_of_day}**")

user_type = st.radio("Are you a new user?", ["Yes", "No"], horizontal=True)

all_genres = sorted(list(set("|".join(movie_features["genres"].fillna("")).split("|"))))
all_genres = [g for g in all_genres if g.strip()]


# -------------------------
# NEW USER FLOW
# -------------------------
if user_type == "Yes":
    st.subheader("🆕 New User Setup")

    new_user_id = st.text_input("Create a user ID for future use")

    selected_genre = st.selectbox("Select a genre", all_genres)

    filtered_movies = movie_features[
        movie_features["genres"].str.contains(selected_genre, na=False)
    ]["title"].dropna().tolist()

    liked_movies = st.multiselect(
        "Select movies you like",
        options=filtered_movies[:200]
    )

    preferred_genres = st.multiselect(
        "Select additional preferred genres",
        options=all_genres
    )

    if st.button("Get Recommendations", key="new_user_recs"):
        if not new_user_id.strip():
            st.error("❌ Please create a valid user ID.")
            st.stop()

        if user_exists(new_user_id.strip()):
            st.error("❌ This user ID already exists. Please choose a different one.")
            st.stop()

        create_user(
            new_user_id.strip(),
            liked_movies=liked_movies,
            preferred_genres=preferred_genres
        )
        animated_progress("🎬 Generating recommendations...")

        recs = new_user_model.recommend(
            preferred_genres=preferred_genres,
            liked_movies=liked_movies,
            mood=mood,
            top_n=10
        )

        st.session_state["recs"] = recs
        st.session_state["user_id"] = new_user_id.strip()
        st.session_state["explanations"] = build_explanations(
            recs=recs,
            mood=mood,
            recent_movies=[],
            preferred_genres=preferred_genres
        )

        st.success(f"✅ User '{new_user_id.strip()}' created successfully.")


# -------------------------
# EXISTING USER FLOW
# -------------------------
else:
    st.subheader("👤 Existing User")

    existing_user_id = st.text_input("Enter your user ID")

    if existing_user_id:
        existing_user_id = existing_user_id.strip()

        if not user_exists(existing_user_id):
            st.error("❌ User not registered. Please enter a valid user ID.")
            st.stop()

        recent_movies = show_history_section(existing_user_id)

        if st.button("Get Recommendations", key="existing_user_recs"):
            if not recent_movies:
                liked_movies, preferred_genres = get_user_preferences(existing_user_id)

                if not liked_movies and not preferred_genres:
                    st.warning("No history or preferences found.")
                    st.stop()

                st.info("Using your saved preferences...")

                recs = new_user_model.recommend(
                    preferred_genres=preferred_genres,
                    liked_movies=liked_movies,
                    mood=mood,
                    top_n=10
                )

                st.session_state["recs"] = recs
                st.session_state["user_id"] = existing_user_id
            else:
                animated_progress("🎬 Updating your personalized recommendations...")

                recs = existing_user_model.recommend(
                    recent_movies=recent_movies,
                    mood=mood,
                    top_n=10
                )

                st.session_state["recs"] = recs
                st.session_state["user_id"] = existing_user_id
                st.session_state["explanations"] = build_explanations(
                    recs=recs,
                    mood=mood,
                    recent_movies=recent_movies
                )


# -------------------------
# SHOW RECOMMENDATIONS
# -------------------------
if "recs" in st.session_state:
    st.subheader("🎬 Top Recommendations")

    recs = st.session_state["recs"]
    current_user_id = st.session_state.get("user_id", "")
    explanation_map = st.session_state.get("explanations", {})

    for i, row in enumerate(recs.head(10).itertuples(), 1):
        explanation = explanation_map.get(
            row.title,
            generate_explanation(
                movie_genres=row.genres,
                mood=mood,
                recent_movies=get_recent_history(current_user_id),
                recommended_title=row.title,
                movie_features_df=movie_features
            )
        )

        st.markdown(f"### {i}. {row.title}")
        st.write(f"⭐ Score: {round(row.final_score, 3)}")
        st.write(f"💡 {explanation}")

        if st.button(f"🎥 Watch {row.title}", key=f"watch_{row.movieId}_{i}"):
            add_clicked_movie(current_user_id, row.title)
            st.success(f"Added '{row.title}' to your history.")
            st.balloons()
            st.rerun()