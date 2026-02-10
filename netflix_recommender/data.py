import csv
from pathlib import Path
from typing import Dict, List, Tuple

from .models import Movie, Rating, User

GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def _safe_year(raw_date: str) -> int | None:
    if not raw_date:
        return None
    parts = raw_date.split("-")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def load_movielens_100k(dataset_dir: str | Path) -> Tuple[Dict[int, User], Dict[int, Movie], List[Rating]]:
    dataset_dir = Path(dataset_dir)
    users: Dict[int, User] = {}
    movies: Dict[int, Movie] = {}
    ratings: List[Rating] = []

    with (dataset_dir / "u.user").open("r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            user_id, age, sex, occupation, _zip = row
            users[int(user_id)] = User(int(user_id), int(age), sex, occupation)

    with (dataset_dir / "u.item").open("r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            movie_id = int(row[0])
            title = row[1]
            year = _safe_year(row[2])
            genres = {genre: int(row[5 + idx]) for idx, genre in enumerate(GENRE_COLUMNS)}
            movies[movie_id] = Movie(movie_id, title, year, genres)

    with (dataset_dir / "u.data").open("r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            user_id, movie_id, rating, _ts = row
            ratings.append(Rating(int(user_id), int(movie_id), int(rating)))

    return users, movies, ratings
