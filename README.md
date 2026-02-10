# Hybrid Fuzzy-Genetic Recommender (Revamped)

This repository now includes a **modern, product-style movie recommender demo** inspired by Netflix-like “what should I watch next?” experiences.

## What’s new

The previous research-script style code has been complemented with a practical application layer:

- ✅ **Simple UI**: interactive CLI (`app.py`) for entering tastes.
- ✅ **Explainability**: every recommendation includes human-readable reasons.
- ✅ **Observability**: recommendation pipeline emits logging and runtime metrics.
- ✅ **Unit tests**: behavior-focused tests for ranking, filtering, and explainability.

---

## Product use case

Imagine a user opening a streaming app and typing:

- Likes: `Action, Sci-Fi`
- Dislikes: `Horror`
- Release range: `2010+`

The app computes recommendations by combining:

1. **Content affinity** (genre match to user tastes)
2. **Collaborative signal** (ratings from users with similar inferred profiles)

Then it returns top movies along with reasons like:

- “Matches your preferred genres: Action, Sci-Fi.”
- “Content score=1.20 and collaborative score=4.10.”
- “Community average rating is 3.85/5.”

---

## Architecture

### New modules

- `netflix_recommender/models.py`
  - Data classes: `User`, `Movie`, `Rating`, `Recommendation`.
- `netflix_recommender/data.py`
  - Loads MovieLens 100K (`u.user`, `u.item`, `u.data`) using stdlib `csv`.
- `netflix_recommender/engine.py`
  - Core recommendation engine, explainability, and observability report.
- `app.py`
  - CLI UI.
- `tests/test_engine.py`
  - Unit test suite.

### Legacy modules

Legacy research files are still present (`fgrs.py`, `gim.py`, `fuzzy_sets.py`, `genetic.py`) for reference, but the primary runnable user experience is now `app.py`.

---

## How recommendations are computed

For each candidate movie:

1. **Filter** by optional year bounds.
2. Compute **content score** from preferred/disliked genres.
3. Build a pseudo user profile from taste input.
4. Find nearest users by cosine similarity against learned user profiles.
5. Compute **collaborative score** using similarity-weighted ratings.
6. Blend scores:

```text
final_score = 0.7 * content_score + 0.3 * collaborative_score
```

7. Attach explainability strings and return top-k sorted results.

---

## Explainability and observability

### Explainability output

Each recommendation includes:

- Matched preferred genres.
- Presence of disliked genres (if any).
- Numeric content and collaborative scores.
- Community average rating.

### Observability output

The engine returns an `ObservabilityReport` containing:

- `candidate_movies`
- `filtered_movies`
- `elapsed_ms`

And also writes structured log events:

- `recommendation_started`
- `recommendation_completed`

---

## Run the UI

```bash
python app.py
```

You’ll be prompted for:

- Preferred genres (comma-separated)
- Disliked genres (optional)
- Min and max release year (optional)

---

## Run tests

```bash
python -m unittest discover -s tests -v
```

---

## Notes

- The implementation intentionally uses Python standard library for loading/parsing data to keep local setup lightweight.
- Movie source data is the bundled `ml-100k/` dataset.
