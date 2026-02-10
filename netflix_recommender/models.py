from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class User:
    user_id: int
    age: int
    sex: str
    occupation: str


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    release_year: int | None
    genres: Dict[str, int]


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: int


@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    content_score: float
    collaborative_score: float
    explainability: List[str] = field(default_factory=list)
