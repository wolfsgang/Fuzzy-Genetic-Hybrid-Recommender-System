#!/usr/bin/env python3
"""Simple CLI UI for Netflix-style movie recommendations."""

import logging

from netflix_recommender.engine import RecommendationEngine, UserTasteInput, normalize_genre_input


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def _prompt_genres(label: str) -> dict[str, float]:
    raw = input(label).strip()
    if not raw:
        return {}
    return normalize_genre_input([part.strip() for part in raw.split(",")])


def _prompt_optional_year(label: str):
    raw = input(label).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print("Invalid year entered. Ignoring this filter.")
        return None


def main() -> None:
    _configure_logging()
    print("\n🎬 Smart Watch Recommender (Netflix-style demo)")
    print("Tell us what you like and we will suggest what to watch next.\n")

    engine = RecommendationEngine.from_movielens("ml-100k")

    print("Available genres include: Action, Adventure, Animation, Children's, Comedy, Crime,")
    print("Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western")

    likes = _prompt_genres("Enter preferred genres (comma-separated): ")
    dislikes = _prompt_genres("Enter disliked genres (comma-separated, optional): ")
    min_year = _prompt_optional_year("Minimum release year (optional): ")
    max_year = _prompt_optional_year("Maximum release year (optional): ")

    taste = UserTasteInput(
        preferred_genres=likes,
        disliked_genres=dislikes,
        min_year=min_year,
        max_year=max_year,
    )

    recommendations, report = engine.recommend(taste, top_k=10)

    print("\nTop recommendations for you:\n")
    for idx, rec in enumerate(recommendations, start=1):
        print(f"{idx:>2}. {rec.title} | final={rec.score:.2f}")
        for reason in rec.explainability[:2]:
            print(f"    - {reason}")

    print("\nObservability summary:")
    print(f"- Candidate movies considered: {report.candidate_movies}")
    print(f"- Movies after filters: {report.filtered_movies}")
    print(f"- Recommendation latency: {report.elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
