CineMatch: Indian Movie Recommendation System
A Netflix-style movie recommender that suggests Indian movies from a user's ratings, watch history, favourite genres and the behaviour of similar viewers. It combines supervised, unsupervised and advanced machine learning.
Machine learning used
Type
Technique
Job in the project
Unsupervised
K-Means (k = 4), PCA
Groups users with similar taste; plots clusters in 2D
Supervised
Random Forest (baselines: Linear Regression, KNN)
Predicts the star rating a user would give
Advanced
SVD matrix factorization, neural network (MLP)
Finds hidden taste factors, re-ranks results
Final score = 40% genre fit + 20% group affinity + 20% SVD + 20% Random Forest.
Folder structure
cinematch/
├── frontend/
│   └── index.html          Netflix-style UI + ML lab tab
├── backend/
│   ├── app.py              Flask API (trains the models, serves /api/recommend)
│   ├── movies.csv          20 Indian movies with genre columns
│   ├── ratings.csv         360 users, 3,960 synthetic ratings
│   ├── make_data.py        regenerates the two CSV files
│   └── requirements.txt
├── notebooks/
│   └── analysis.ipynb      charts and metrics for the report
├── .gitignore
└── README.md
Setup (VS Code terminal, from the cinematch folder)
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r backend/requirements.txt
Run
Start the API:
cd backend
python app.py
It prints the Random Forest test RMSE and serves http://127.0.0.1:5000.
Health check: open http://127.0.0.1:5000/api/health.
Open frontend/index.html with the Live Server extension.
The top-right pill shows Live ML API when the backend is connected. Without the backend the page runs in Demo mode with a built-in ranking.
Notebook
Open notebooks/analysis.ipynb in VS Code, pick the venv interpreter as the kernel, and choose Run All. It produces the rating histogram, genre chart, elbow and silhouette curves, PCA cluster plot, model comparison, feature importance, training curve and Precision@K.
API
POST /api/recommend
{ "ratings": { "0": 5, "1": 5 }, "genres": ["Action"] }
Returns every unrated movie with score, pred (predicted stars) and the score components content, group, svd, rf.
Notes
ratings.csv is synthetic (generated from four viewer groups). Replace it with real ratings, such as a survey or MovieLens, for a real evaluation.
Movie ids in movies.csv (0 to 19) must match the ids in frontend/index.html.
The ML lab charts in the frontend show sample values. Replace them with the numbers from the notebook.