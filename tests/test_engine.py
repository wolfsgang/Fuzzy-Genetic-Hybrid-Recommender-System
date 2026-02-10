import unittest

from netflix_recommender.engine import RecommendationEngine, UserTasteInput, normalize_genre_input
from netflix_recommender.models import Movie, Rating, User


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        users = {
            1: User(1, 25, "M", "engineer"),
            2: User(2, 30, "F", "artist"),
        }
        movies = {
            10: Movie(10, "Action Hit", 2020, {"Action": 1, "Drama": 0}),
            11: Movie(11, "Drama Gem", 2019, {"Action": 0, "Drama": 1}),
            12: Movie(12, "Action Drama", 2021, {"Action": 1, "Drama": 1}),
        }
        ratings = [
            Rating(1, 10, 5),
            Rating(1, 11, 2),
            Rating(2, 10, 4),
            Rating(2, 11, 5),
            Rating(2, 12, 4),
        ]
        self.engine = RecommendationEngine(users, movies, ratings)

    def test_recommend_prefers_matching_genres(self):
        taste = UserTasteInput(preferred_genres={"Action": 1.0}, disliked_genres={"Drama": 0.2})
        recs, report = self.engine.recommend(taste, top_k=2)

        self.assertEqual(len(recs), 2)
        self.assertGreaterEqual(recs[0].score, recs[1].score)
        self.assertGreater(report.candidate_movies, 0)
        self.assertGreater(report.filtered_movies, 0)

    def test_year_filter(self):
        taste = UserTasteInput(preferred_genres={"Action": 1.0}, min_year=2021)
        recs, report = self.engine.recommend(taste, top_k=5)

        self.assertEqual(report.filtered_movies, 1)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].title, "Action Drama")

    def test_explainability_contains_scores(self):
        taste = UserTasteInput(preferred_genres={"Drama": 1.0})
        recs, _ = self.engine.recommend(taste, top_k=1)

        explanation_text = " ".join(recs[0].explainability)
        self.assertIn("Content score", explanation_text)
        self.assertIn("collaborative score", explanation_text)


class NormalizeInputTests(unittest.TestCase):
    def test_normalize_genres(self):
        normalized = normalize_genre_input(["action", "Drama", "unknown", "invalid"])
        self.assertEqual(normalized, {"Action": 1.0, "Drama": 1.0})


if __name__ == "__main__":
    unittest.main()
