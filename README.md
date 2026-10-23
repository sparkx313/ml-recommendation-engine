# Movie Recommendation Engine

A command-line movie recommender that calculates recommendations from a ratings CSV supplied by the user. It does not bundle or download a sample dataset.

## How recommendations are scored

The engine builds a sparse user-item ratings matrix and computes item-to-item cosine similarity. For a user with rating history, it predicts ratings for movies they have not rated by weighting their rating deviations against similar movies. Estimates are kept within the rating range in the input file.

Movies without useful item overlap use a popularity fallback: each movie's average rating is shrunk toward the dataset-wide average, with `--prior-strength` controlling the shrinkage. A user with no history receives that popularity ranking. Movies already rated by a known user are excluded.

## Input file

Pass a CSV with these required columns:

| Column | Meaning |
| --- | --- |
| `user_id` | Stable identifier for the person who rated the movie |
| `item_id` | Stable identifier for the movie |
| `rating` | Numeric rating on the scale used by the dataset |
| `item_title` | Optional display title; each `item_id` must have one consistent title |

Repeated ratings for the same user and movie are averaged. IDs are read as text, so leading zeroes are retained. Rows with missing IDs or non-numeric ratings produce an error.

## Run locally

Requires Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt
python recommender.py --ratings path/to/ratings.csv --user-id 42 --top-n 10 --output recommendations.csv
```

The output CSV contains the user ID, movie ID, optional title, predicted rating, number of distinct users who rated the movie, and the scoring method. It only includes movies absent from the requested user's history. If that user does not appear in the input, the output uses the popularity fallback.

## Run with Docker

Build the image from this directory:

```bash
docker build -t movie-recommender .
```

Mount the directory containing your ratings file and pass the container paths to the command:

```bash
docker run --rm -v /absolute/path/to/data:/data movie-recommender --ratings /data/ratings.csv --user-id 42 --output /data/recommendations.csv
```

Replace `/absolute/path/to/data` with the host directory containing your own CSV. The program reads and writes local files and makes no network requests.

## Scope

This is explicit-rating, item-item collaborative filtering. It does not claim a measured accuracy score, train/test benchmark, real-time service, or support for implicit click and view events. Recommendation quality depends on the coverage and consistency of the supplied ratings.

## License

MIT. See [LICENSE](LICENSE).
