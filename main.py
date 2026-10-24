from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent / "IMDB Dataset.csv"


def main() -> None:
    movies = pd.read_csv(DATASET_PATH)

    print("First 3 rows:")
    print(movies.head(3))

    print("\nLast 3 rows:")
    print(movies.tail(3))

    print("\nDataset information:")
    movies.info()

    print("\nDetailed descriptive statistics:")
    print(movies.describe(include="all"))

    print("\nNull values by column:")
    print(movies.isna().sum())
    print(f"\nTotal null values: {movies.isna().sum().sum()}")

    # Dataset row numbers are 1-based; iloc uses 0-based positions.
    rows_41_to_75 = movies.iloc[40:75]
    print("\nRows 41 to 75:")
    print(rows_41_to_75)

    highest_voted_movie = movies.loc[movies["No_of_Votes"].idxmax()]
    print("\nMovie with the highest number of votes:")
    print(highest_voted_movie)

    numeric_features = ["IMDB_Rating", "Runtime"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, feature in zip(axes, numeric_features):
        axis.boxplot(movies[feature].dropna())
        axis.set_title(f"{feature} boxplot")
        axis.set_ylabel(feature)
    fig.tight_layout()

    plt.figure(figsize=(8, 6))
    plt.scatter(movies["Runtime"], movies["IMDB_Rating"], alpha=0.6)
    plt.title("IMDb Rating vs Runtime")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("IMDb Rating")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, feature in zip(axes, numeric_features):
        axis.hist(movies[feature].dropna(), bins=15, edgecolor="black")
        axis.set_title(f"Distribution of {feature}")
        axis.set_xlabel(feature)
        axis.set_ylabel("Number of movies")
    fig.tight_layout()

    rating_counts = movies["IMDB_Rating"].value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    plt.bar(rating_counts.index.astype(str), rating_counts.values)
    plt.title("Number of Movies by IMDb Rating")
    plt.xlabel("IMDb Rating")
    plt.ylabel("Number of movies")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()