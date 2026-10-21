"""
Movie Rating Analysis
Libraries: pandas, numpy, matplotlib, seaborn
Run: python movie_rating_analysis.py
Install requirements: pip install -r requirements.txt
Note: movie_ratings.csv is a small practice dataset created for this project.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv("movie_ratings.csv")
print("First five records:")
print(df.head())
print("\nDataset information:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

# 2. Clean and prepare data
df = df.drop_duplicates()
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Release_Year"] = pd.to_numeric(df["Release_Year"], errors="coerce")
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")
df = df.dropna(subset=["Title", "Genre", "Rating", "Release_Year", "Votes"])
df = df[df["Rating"].between(0, 10)]
df["Release_Year"] = df["Release_Year"].astype(int)
df["Votes"] = df["Votes"].astype(int)

# 3. Summary statistics using NumPy and Pandas
print("\nNumber of movies:", len(df))
print("Average rating:", round(np.mean(df["Rating"]), 2))
print("Median rating:", round(np.median(df["Rating"]), 2))
print("Highest rating:", df["Rating"].max())
print("Lowest rating:", df["Rating"].min())
print("\nMovies by genre:")
print(df["Genre"].value_counts())
print("\nAverage rating by genre:")
genre_rating = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False)
print(genre_rating.round(2))
print("\nTop 10 rated movies:")
print(df.sort_values("Rating", ascending=False)[["Title", "Genre", "Rating", "Votes"]].head(10))

# 4. Visualizations
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(df["Rating"], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating (out of 10)")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("rating_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(11, 6))
genre_order = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False).index
sns.barplot(data=df, x="Genre", y="Rating", order=genre_order, errorbar=None)
plt.title("Average Rating by Genre")
plt.xlabel("Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig("average_rating_by_genre.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Votes", y="Rating", hue="Genre", alpha=0.8)
plt.title("Movie Votes vs. Rating")
plt.xlabel("Number of Votes")
plt.ylabel("Rating (out of 10)")
plt.tight_layout()
plt.savefig("votes_vs_rating.png", dpi=150)
plt.show()

year_counts = df.groupby("Release_Year")["Title"].count().reset_index(name="Movie_Count")
plt.figure(figsize=(11, 6))
sns.lineplot(data=year_counts, x="Release_Year", y="Movie_Count", marker="o")
plt.title("Number of Movies by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("movies_by_year.png", dpi=150)
plt.show()

# 5. Export useful results
genre_rating.rename("Average_Rating").round(2).to_csv("genre_rating_summary.csv")
df.sort_values("Rating", ascending=False).head(10).to_csv("top_10_movies.csv", index=False)
print("\nAnalysis complete. Charts and summary CSV files have been saved.")
