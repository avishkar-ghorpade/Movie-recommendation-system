
from flask import Flask, send_from_directory, jsonify, request
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MOVIES_CSV = os.path.join(BASE_DIR, "backend", "movies.csv")


# --------------------------------------------------
# LOAD DATA AND TRAIN HYBRID ML MODELS
# --------------------------------------------------

def load_ml_model():
    movies = pd.read_csv(MOVIES_CSV, encoding="utf-8-sig")
    movies.columns = (
        movies.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
    )

    required = {
        "movieId", "title", "year", "avg", "language"
    }

    missing = required - set(movies.columns)
    if missing:
        raise ValueError(
            f"Missing columns in movies.csv: {sorted(missing)}. "
            f"Found: {movies.columns.tolist()}"
        )

    movies["movieId"] = pd.to_numeric(
        movies["movieId"], errors="raise"
    ).astype(int)

    movies["year"] = pd.to_numeric(
        movies["year"], errors="coerce"
    )

    movies["avg"] = pd.to_numeric(
        movies["avg"], errors="coerce"
    )

    movies["year"] = movies["year"].fillna(
        movies["year"].median()
    )
    movies["avg"] = movies["avg"].fillna(
        movies["avg"].mean()
    )

    movies["title"] = movies["title"].fillna("Unknown").astype(str)
    movies["language"] = (
        movies["language"].fillna("Unknown").astype(str)
    )

    # Genre columns are all columns other than these metadata fields.
    metadata_columns = {
        "movieId", "title", "year", "avg", "language"
    }

    genre_columns = [
        col for col in movies.columns
        if col not in metadata_columns
    ]

    if not genre_columns:
        raise ValueError(
            "No genre columns found in movies.csv."
        )

    for col in genre_columns:
        movies[col] = pd.to_numeric(
            movies[col], errors="coerce"
        ).fillna(0)

    # 1. CONTENT-BASED FILTERING
    # Combine language and active genres into text features.
    movies["content"] = movies.apply(
        lambda row: " ".join(
            [row["language"]]
            + [
                col for col in genre_columns
                if row[col] == 1
            ]
        ),
        axis=1
    )

    tfidf = TfidfVectorizer()
    content_matrix = tfidf.fit_transform(movies["content"])

    similarity_matrix = cosine_similarity(content_matrix)

    # Features for K-Means and Random Forest.
    language_features = pd.get_dummies(
        movies["language"],
        prefix="lang",
        dtype=int
    )

    numeric_features = movies[genre_columns + ["year"]].copy()

    features = pd.concat(
        [
            numeric_features.reset_index(drop=True),
            language_features.reset_index(drop=True)
        ],
        axis=1
    )

    # 2. UNSUPERVISED ML: K-MEANS CLUSTERING
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    cluster_count = min(5, len(movies))

    kmeans = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init=10
    )

    movies["cluster"] = kmeans.fit_predict(scaled_features)

    # 3. SUPERVISED ML: RANDOM FOREST REGRESSION
    # The dataset's avg column is the target rating.
    forest = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    forest.fit(features, movies["avg"])

    movies["predicted_rating"] = np.clip(
        forest.predict(features), 0, 5
    )

    return movies, similarity_matrix, genre_columns


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# --------------------------------------------------
# ML LAB PAGE
# --------------------------------------------------

@app.route("/ml")
def ml_lab():
    return send_from_directory(FRONTEND_DIR, "ml.html")


# --------------------------------------------------
# API: GET ALL MOVIES
# --------------------------------------------------

@app.route("/api/movies")
def api_movies():
    try:
        movies, _, genre_columns = load_ml_model()

        result = []

        for _, row in movies.iterrows():
            result.append({
                "id": int(row["movieId"]),
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "year": int(row["year"]),
                "avg": round(float(row["avg"]), 2),
                "language": row["language"],
                "genres": [
                    genre for genre in genre_columns
                    if row[genre] == 1
                ]
            })

        return jsonify({"movies": result})

    except Exception as exc:
        app.logger.exception("Could not load movies")
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------
# API: HYBRID MOVIE RECOMMENDATIONS
# --------------------------------------------------

@app.route("/api/recommend")
def api_recommend():
    try:
        movie_id = request.args.get("movie_id", type=int)

        if movie_id is None:
            return jsonify({
                "error": "Please provide movie_id."
            }), 400

        movies, similarity_matrix, genre_columns = load_ml_model()

        matches = movies.index[
            movies["movieId"] == movie_id
        ].tolist()

        if not matches:
            return jsonify({
                "error": f"No movie found with movieId {movie_id}."
            }), 404

        selected_index = matches[0]
        selected_movie = movies.iloc[selected_index]

        recommendations = []

        for i, row in movies.iterrows():
            if i == selected_index:
                continue

            # Signal A: content similarity, from 0 to 1.
            content_score = float(
                similarity_matrix[selected_index, i]
            )

            # Signal B: Random Forest predicted rating, normalized.
            rating_score = float(
                np.clip(row["predicted_rating"] / 5.0, 0, 1)
            )

            # Signal C: same K-Means cluster.
            cluster_score = float(
                row["cluster"] == selected_movie["cluster"]
            )

            # Hybrid ranking score.
            hybrid_score = (
                0.55 * content_score
                + 0.30 * rating_score
                + 0.15 * cluster_score
            )

            recommendations.append({
                "id": int(row["movieId"]),
                "movieId": int(row["movieId"]),
                "title": row["title"],
                "year": int(row["year"]),
                "avg": round(float(row["avg"]), 2),
                "language": row["language"],
                "genres": [
                    genre for genre in genre_columns
                    if row[genre] == 1
                ],
                "similarity": round(content_score * 100, 2),
                "predicted_rating": round(
                    float(row["predicted_rating"]), 2
                ),
                "cluster": int(row["cluster"]),
                "hybrid_score": round(hybrid_score * 100, 2)
            })

        recommendations.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        return jsonify({
            "selected_movie": selected_movie["title"],
            "selected_movie_id": int(selected_movie["movieId"]),
            "selected_cluster": int(selected_movie["cluster"]),
            "techniques": [
                "TF-IDF Content-Based Filtering",
                "Cosine Similarity",
                "K-Means Clustering",
                "Random Forest Regression",
                "Hybrid Ranking"
            ],
            "recommendations": recommendations[:8]
        })

    except Exception as exc:
        app.logger.exception("Recommendation failed")
        return jsonify({"error": str(exc)}), 500


# --------------------------------------------------
# RUN FLASK
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
