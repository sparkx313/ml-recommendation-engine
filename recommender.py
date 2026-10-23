"""Recommend unseen movies from a user-item ratings CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

REQUIRED_COLUMNS = {"user_id", "item_id", "rating"}


def load_ratings(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load and validate ratings, averaging repeated ratings for each pair."""
    if not path.is_file():
        raise ValueError(f"Ratings file does not exist: {path}")

    ratings = pd.read_csv(path, dtype={"user_id": "string", "item_id": "string"})
    missing_columns = REQUIRED_COLUMNS - set(ratings.columns)
    if missing_columns:
        raise ValueError(
            "Ratings CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    ratings["user_id"] = ratings["user_id"].str.strip().fillna("")
    ratings["item_id"] = ratings["item_id"].str.strip().fillna("")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    rating_values = ratings["rating"].to_numpy(dtype=np.float64, na_value=np.nan)
    invalid = (
        ratings["user_id"].eq("")
        | ratings["item_id"].eq("")
        | ~np.isfinite(rating_values)
    )
    if invalid.any():
        raise ValueError(
            f"Found {int(invalid.sum())} row(s) with a missing user_id, item_id, "
            "or a rating that is not finite numeric. Correct those rows and run again."
        )
    if ratings.empty:
        raise ValueError("The ratings CSV contains no rows.")

    titles: dict[str, str] = {}
    if "item_title" in ratings.columns:
        title_values = ratings["item_title"].astype("string").str.strip()
        has_title = title_values.notna() & title_values.ne("")
        titled_rows = ratings.loc[has_title, ["item_id"]].copy()
        titled_rows["item_title"] = title_values[has_title]
        distinct_titles = titled_rows.drop_duplicates()
        conflicting = distinct_titles.groupby("item_id", sort=False)["item_title"].nunique()
        conflicting = conflicting[conflicting > 1]
        if not conflicting.empty:
            examples = ", ".join(conflicting.index.astype(str).tolist()[:5])
            raise ValueError(
                "Each item_id must map to one item_title; conflicting IDs include: "
                + examples
            )
        titles = dict(
            zip(distinct_titles["item_id"], distinct_titles["item_title"], strict=False)
        )

    interactions = (
        ratings.groupby(["user_id", "item_id"], as_index=False, sort=False)["rating"]
        .mean()
    )
    return interactions, titles


def recommend(
    ratings_path: Path,
    user_id: str,
    top_n: int = 10,
    prior_strength: float = 10.0,
) -> pd.DataFrame:
    """Generate item-item recommendations or a popularity cold-start ranking."""
    if top_n < 1:
        raise ValueError("--top-n must be at least 1.")
    if not np.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("--prior-strength must be a finite, non-negative number.")

    interactions, titles = load_ratings(ratings_path)
    users = pd.Index(interactions["user_id"].drop_duplicates())
    items = pd.Index(interactions["item_id"].drop_duplicates())
    user_codes = users.get_indexer(interactions["user_id"])
    item_codes = items.get_indexer(interactions["item_id"])
    values = interactions["rating"].to_numpy(dtype=np.float64)

    user_item = csr_matrix(
        (values, (user_codes, item_codes)), shape=(len(users), len(items))
    )
    item_user = user_item.T.tocsr()

    item_counts = np.bincount(item_codes, minlength=len(items)).astype(np.int64)
    item_rating_sums = np.asarray(user_item.sum(axis=0)).ravel()
    item_means = item_rating_sums / item_counts
    global_mean = float(values.mean())
    popular_scores = (
        item_rating_sums + prior_strength * global_mean
    ) / (item_counts + prior_strength)

    user_id = str(user_id).strip()
    if not user_id:
        raise ValueError("--user-id cannot be empty.")
    user_position = users.get_indexer(pd.Index([user_id]))[0]
    recommendation_methods = np.full(len(items), "popularity", dtype=object)

    if user_position < 0:
        scores = popular_scores.copy()
    else:
        history = user_item.getrow(user_position)
        history_item_indices = history.indices
        user_mean = float(history.data.mean())
        item_norms = np.sqrt(
            np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel()
        )
        inverse_norms = np.zeros_like(item_norms)
        np.divide(1.0, item_norms, out=inverse_norms, where=item_norms > 0)

        # Sparse item-item cosine similarities avoid building a full dense matrix.
        similarities = item_user[history_item_indices] @ item_user.T
        similarities = (
            diags(inverse_norms[history_item_indices], format="csr")
            @ similarities
            @ diags(inverse_norms, format="csr")
        ).tocsr()
        deviations = history.data - user_mean
        weighted_similarities = diags(deviations, format="csr") @ similarities
        numerator = np.asarray(weighted_similarities.sum(axis=0)).ravel()
        denominator = np.asarray(abs(similarities).sum(axis=0)).ravel()

        scores = popular_scores.copy()
        has_neighbors = denominator > 0
        scores[has_neighbors] = (
            user_mean + numerator[has_neighbors] / denominator[has_neighbors]
        )
        recommendation_methods[:] = "item-item collaborative filtering"
        recommendation_methods[~has_neighbors] = "popularity fallback"

    # Keep estimates on the same scale as the ratings supplied by the user.
    scores = np.clip(scores, float(values.min()), float(values.max()))
    candidates = np.ones(len(items), dtype=bool)
    if user_position >= 0:
        candidates[user_item.getrow(user_position).indices] = False
    candidate_indices = np.flatnonzero(candidates)
    item_values = items.to_numpy(dtype=object)
    order = np.lexsort(
        (
            item_values[candidate_indices],
            -item_counts[candidate_indices],
            -scores[candidate_indices],
        )
    )
    selected = candidate_indices[order[:top_n]]

    result = pd.DataFrame(
        {
            "user_id": user_id,
            "item_id": item_values[selected],
            "predicted_rating": scores[selected],
            "rating_count": item_counts[selected],
            "recommendation_method": recommendation_methods[selected],
        }
    )
    if titles:
        result.insert(
            2,
            "item_title",
            [titles.get(str(item_id), "") for item_id in result["item_id"]],
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recommend unrated movies from a user-item ratings CSV using "
            "sparse item-item collaborative filtering."
        )
    )
    parser.add_argument("--ratings", required=True, type=Path, help="Input ratings CSV.")
    parser.add_argument("--user-id", required=True, help="User to recommend movies for.")
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of movies to return (default: 10).",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=10.0,
        help="Global-mean pseudo-count for popularity fallback (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recommendations.csv"),
        help="Destination CSV (default: recommendations.csv).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.ratings.resolve() == args.output.resolve():
            raise ValueError("The output path must not overwrite the ratings input.")
        recommendations = recommend(
            args.ratings,
            args.user_id,
            top_n=args.top_n,
            prior_strength=args.prior_strength,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(args.output, index=False)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {len(recommendations)} recommendation(s) to {args.output}")
    if recommendations.empty:
        print("No unseen movies remain for this user.")
    elif recommendations["recommendation_method"].eq("popularity").all():
        print("The user has no ratings in the file; results use the popularity fallback.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
