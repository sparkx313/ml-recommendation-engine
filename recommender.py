def recommend_movies(user_id, top_n=5):
    print(f"🎬 Generating Top {top_n} Movie Recommendations for User #{user_id}...")
    return ["Inception", "Interstellar", "The Dark Knight", "The Matrix", "Pulp Fiction"]

def main():
    movies = recommend_movies(42)
    print("✓ Recommendations: " + ", ".join(movies))

if __name__ == "__main__":
    main()
