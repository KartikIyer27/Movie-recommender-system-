# import json
# import os

# USER_FILE = "data/processed/user_profiles.json"


# def load_users():
#     if not os.path.exists(USER_FILE):
#         return {}

#     with open(USER_FILE, "r") as f:
#         return json.load(f)


# def save_users(users):
#     with open(USER_FILE, "w") as f:
#         json.dump(users, f, indent=4)


# # -------------------------
# # ADD CLICKED MOVIE
# # -------------------------
# def add_clicked_movie(user_id: str, movie_title: str):
#     users = load_users()

#     if user_id not in users:
#         users[user_id] = {
#             "recent_watch_history": []
#         }

#     history = users[user_id]["recent_watch_history"]

#     # Add new click
#     history.append(movie_title)

#     # Keep only last 5 movies
#     users[user_id]["recent_watch_history"] = history[-5:]

#     save_users(users)


# # -------------------------
# # GET USER HISTORY
# # -------------------------
# def get_recent_history(user_id: str):
#     users = load_users()

#     if user_id not in users:
#         return []

#     return users[user_id].get("recent_watch_history", [])

import json
import os

USER_FILE = "data/processed/user_profiles.json"


def load_users():
    if not os.path.exists(USER_FILE):
        return {}

    try:
        with open(USER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users):
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    with open(USER_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)


def user_exists(user_id: str) -> bool:
    users = load_users()
    return user_id in users


def create_user(user_id: str, liked_movies=None, preferred_genres=None):
    users = load_users()

    if user_id not in users:
        users[user_id] = {
            "recent_watch_history": [],
            "liked_movies": liked_movies or [],
            "preferred_genres": preferred_genres or []
        }

    save_users(users)

def get_user_preferences(user_id: str):
    users = load_users()

    if user_id not in users:
        return [], []

    liked = users[user_id].get("liked_movies", [])
    genres = users[user_id].get("preferred_genres", [])

    return liked, genres

def add_clicked_movie(user_id: str, movie_title: str):
    users = load_users()

    if user_id not in users:
        users[user_id] = {
            "recent_watch_history": []
        }

    history = users[user_id].get("recent_watch_history", [])

    history.append(movie_title)

    # remove duplicates while keeping order, then keep last 5
    deduped = []
    for movie in history:
        if movie in deduped:
            deduped.remove(movie)
        deduped.append(movie)

    users[user_id]["recent_watch_history"] = deduped[-5:]

    save_users(users)


def get_recent_history(user_id: str):
    users = load_users()

    if user_id not in users:
        return []

    return users[user_id].get("recent_watch_history", [])