"""Generates movies.csv and ratings.csv for the CineMatch project.
Run:  python make_data.py
Ratings are synthetic (4 viewer groups) - replace with real data later."""
import numpy as np, pandas as pd

G = ["Action","Sci-Fi","Romance","Comedy","Drama","Thriller","Family","Horror"]
M = [
 ("RRR",2022,4.7,"Telugu",["Action","Drama"]),
 ("Enthiran",2010,4.1,"Tamil",["Sci-Fi","Action"]),
 ("Baahubali 2",2017,4.5,"Telugu",["Action","Drama"]),
 ("3 Idiots",2009,4.6,"Hindi",["Comedy","Drama"]),
 ("Dangal",2016,4.6,"Hindi",["Drama","Family"]),
 ("Dilwale Dulhania Le Jayenge",1995,4.5,"Hindi",["Romance","Comedy"]),
 ("Drishyam",2015,4.5,"Malayalam",["Thriller","Drama"]),
 ("KGF Chapter 2",2022,4.3,"Kannada",["Action","Drama"]),
 ("Sairat",2016,4.4,"Marathi",["Romance","Drama"]),
 ("Kantara",2022,4.5,"Kannada",["Action","Thriller","Drama"]),
 ("Vikram",2022,4.3,"Tamil",["Action","Thriller"]),
 ("Jawan",2023,4.0,"Hindi",["Action","Thriller"]),
 ("PK",2014,4.4,"Hindi",["Comedy","Sci-Fi"]),
 ("Koi Mil Gaya",2003,4.0,"Hindi",["Sci-Fi","Family"]),
 ("Stree",2018,4.2,"Hindi",["Horror","Comedy"]),
 ("Tumbbad",2018,4.5,"Hindi",["Horror","Drama"]),
 ("Jab We Met",2007,4.4,"Hindi",["Romance","Comedy"]),
 ("Andhadhun",2018,4.6,"Hindi",["Thriller","Comedy"]),
 ("Lagaan",2001,4.5,"Hindi",["Drama","Family"]),
 ("Fandry",2013,4.3,"Marathi",["Drama"]),
]
rows = []
for i, (t, y, a, l, gs) in enumerate(M):
    rows.append({"id": i, "title": t, "year": y, "avg": a, "language": l,
                 **{g: int(g in gs) for g in G}})
movies = pd.DataFrame(rows)
movies.to_csv("movies.csv", index=False)

# 4 viewer groups, same centres as the frontend
centres = np.array([
 [1,.3,0,.1,.5,.8,0,.2],     # Action Lovers
 [.6,1,0,.3,.5,.4,.3,0],     # Sci-Fi Thinkers
 [0,0,1,.5,.9,0,.1,0],       # Romantic Dreamers
 [0,.2,.3,.9,.6,0,1,.2]])    # Family Entertainers
sizes = [112, 96, 84, 68]
rng = np.random.default_rng(42)
V = movies[G].values
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

out, uid = [], 1
for c, n in enumerate(sizes):
    for _ in range(n):
        taste = np.clip(centres[c] + rng.normal(0, .25, 8), 0, None)
        for m in rng.choice(len(movies), rng.integers(8, 15), replace=False):
            s = 1.6 + 3.0 * cos(taste, V[m]) + 0.8 * (movies.avg[m] - 4) + rng.normal(0, .5)
            out.append((uid, int(m), int(np.clip(round(s), 1, 5))))
        uid += 1
pd.DataFrame(out, columns=["user_id", "movie_id", "rating"]).to_csv("ratings.csv", index=False)
print("movies:", len(movies), "| users:", uid - 1, "| ratings:", len(out))