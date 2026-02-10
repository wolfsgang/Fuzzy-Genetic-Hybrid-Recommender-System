import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .data import GENRE_COLUMNS, load_movielens_100k
from .models import Movie, Recommendation, Rating, User

LOGGER = logging.getLogger("recommender")


@dataclass
class UserTasteInput:
    preferred_genres: Dict[str, float]
    disliked_genres: Dict[str, float] = field(default_factory=dict)
    min_year: int | None = None
    max_year: int | None = None


@dataclass
class ObservabilityReport:
    candidate_movies: int
    filtered_movies: int
    elapsed_ms: float


class RecommendationEngine:
    def __init__(self, users: Dict[int, User], movies: Dict[int, Movie], ratings: List[Rating]) -> None:
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.genre_names = [g for g in GENRE_COLUMNS if g != "unknown"]

        self.user_ratings: Dict[int, List[Rating]] = {}
        for r in ratings:
            self.user_ratings.setdefault(r.user_id, []).append(r)

        self.movie_ratings: Dict[int, List[int]] = {}
        for r in ratings:
            self.movie_ratings.setdefault(r.movie_id, []).append(r.rating)

        self.user_profiles = self._build_user_profiles()

    @classmethod
    def from_movielens(cls, dataset_dir: str = "ml-100k") -> "RecommendationEngine":
        users, movies, ratings = load_movielens_100k(dataset_dir)
        return cls(users, movies, ratings)

    def _build_user_profiles(self) -> Dict[int, Dict[str, float]]:
        profiles: Dict[int, Dict[str, float]] = {}
        for user_id, ratings in self.user_ratings.items():
            genre_totals = {g: 0.0 for g in self.genre_names}
            weight_total = 0.0
            for rating in ratings:
                movie = self.movies.get(rating.movie_id)
                if not movie:
                    continue
                centered = rating.rating - 3.0
                if centered == 0:
                    continue
                for g in self.genre_names:
                    if movie.genres.get(g, 0):
                        genre_totals[g] += centered
                weight_total += abs(centered)

            if weight_total == 0:
                profiles[user_id] = {g: 0.0 for g in self.genre_names}
            else:
                profiles[user_id] = {g: genre_totals[g] / weight_total for g in self.genre_names}
        return profiles

    def recommend(self, taste: UserTasteInput, top_k: int = 10) -> tuple[List[Recommendation], ObservabilityReport]:
        start = time.perf_counter()
        LOGGER.info("recommendation_started", extra={"top_k": top_k})

        candidates = list(self.movies.values())
        filtered = [m for m in candidates if self._movie_in_range(m, taste.min_year, taste.max_year)]

        scored: List[Recommendation] = []
        for movie in filtered:
            content_score = self._content_score(movie, taste)
            collaborative_score = self._collaborative_score(movie, taste)
            final_score = 0.7 * content_score + 0.3 * collaborative_score
            explainability = self._explain(movie, content_score, collaborative_score, taste)
            scored.append(
                Recommendation(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    score=final_score,
                    content_score=content_score,
                    collaborative_score=collaborative_score,
                    explainability=explainability,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        elapsed = (time.perf_counter() - start) * 1000
        report = ObservabilityReport(
            candidate_movies=len(candidates),
            filtered_movies=len(filtered),
            elapsed_ms=elapsed,
        )
        LOGGER.info(
            "recommendation_completed",
            extra={
                "candidate_movies": report.candidate_movies,
                "filtered_movies": report.filtered_movies,
                "elapsed_ms": round(report.elapsed_ms, 2),
            },
        )
        return scored[:top_k], report

    def _movie_in_range(self, movie: Movie, min_year: int | None, max_year: int | None) -> bool:
        if movie.release_year is None:
            return True
        if min_year is not None and movie.release_year < min_year:
            return False
        if max_year is not None and movie.release_year > max_year:
            return False
        return True

    def _content_score(self, movie: Movie, taste: UserTasteInput) -> float:
        if not taste.preferred_genres and not taste.disliked_genres:
            return 0.0
        score = 0.0
        for genre, weight in taste.preferred_genres.items():
            if movie.genres.get(genre, 0):
                score += weight
        for genre, penalty in taste.disliked_genres.items():
            if movie.genres.get(genre, 0):
                score -= penalty
        return score

    def _collaborative_score(self, movie: Movie, taste: UserTasteInput) -> float:
        pseudo_profile = self._taste_to_profile(taste)
        neighbors = self._nearest_users(pseudo_profile, k=30)
        weighted_sum = 0.0
        weight_total = 0.0
        for user_id, sim in neighbors:
            user_movie_rating = self._rating_by_user_for_movie(user_id, movie.movie_id)
            if user_movie_rating is None:
                continue
            weighted_sum += sim * user_movie_rating
            weight_total += sim

        if weight_total > 0:
            return weighted_sum / weight_total

        ratings = self.movie_ratings.get(movie.movie_id, [])
        return sum(ratings) / len(ratings) if ratings else 0.0

    def _taste_to_profile(self, taste: UserTasteInput) -> Dict[str, float]:
        profile = {g: 0.0 for g in self.genre_names}
        for g, w in taste.preferred_genres.items():
            if g in profile:
                profile[g] += w
        for g, p in taste.disliked_genres.items():
            if g in profile:
                profile[g] -= p
        return profile

    def _nearest_users(self, pseudo_profile: Dict[str, float], k: int = 30) -> List[tuple[int, float]]:
        neighbors: List[tuple[int, float]] = []
        for user_id, profile in self.user_profiles.items():
            sim = self._cosine_similarity(pseudo_profile, profile)
            if sim > 0:
                neighbors.append((user_id, sim))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:k]

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        dot = sum(a[k] * b.get(k, 0.0) for k in a.keys())
        an = math.sqrt(sum(v * v for v in a.values()))
        bn = math.sqrt(sum(v * v for v in b.values()))
        if an == 0 or bn == 0:
            return 0.0
        return dot / (an * bn)

    def _rating_by_user_for_movie(self, user_id: int, movie_id: int) -> int | None:
        for rating in self.user_ratings.get(user_id, []):
            if rating.movie_id == movie_id:
                return rating.rating
        return None

    def _explain(self, movie: Movie, content_score: float, collaborative_score: float, taste: UserTasteInput) -> List[str]:
        reasons: List[str] = []
        matched = [g for g in taste.preferred_genres if movie.genres.get(g, 0)]
        avoided = [g for g in taste.disliked_genres if movie.genres.get(g, 0)]

        if matched:
            reasons.append(f"Matches your preferred genres: {', '.join(matched[:3])}.")
        if avoided:
            reasons.append(f"Contains some less preferred genres: {', '.join(avoided[:2])}.")
        reasons.append(f"Content score={content_score:.2f} and collaborative score={collaborative_score:.2f}.")

        if movie.movie_id in self.movie_ratings:
            avg_rating = sum(self.movie_ratings[movie.movie_id]) / len(self.movie_ratings[movie.movie_id])
            reasons.append(f"Community average rating is {avg_rating:.2f}/5.")
        return reasons


def normalize_genre_input(genres: Sequence[str]) -> Dict[str, float]:
    normalized = {}
    valid_genres = {g.lower(): g for g in GENRE_COLUMNS if g != "unknown"}
    for g in genres:
        token = g.strip().lower()
        if token in valid_genres:
            normalized[valid_genres[token]] = 1.0
    return normalized
