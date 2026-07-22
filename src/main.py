"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from typing import Dict, List, Tuple

from recommender import load_songs, recommend_songs


def print_recommendations(recommendations: List[Tuple[Dict, float, str]]) -> None:
    """Render the ranked recommendations as a clean, readable terminal report."""
    width = 60
    print()
    print("=" * width)
    print("TOP RECOMMENDATIONS".center(width))
    print("=" * width)

    if not recommendations:
        print("\nNo recommendations found for this profile.\n")
        return

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        # Header line: rank, title, artist, and score as a percentage.
        print(f"\n{rank}. {song['title']} — {song['artist']}")
        print(f"   Score: {score:.2f}  ({score * 100:.0f}% match)")

        # recommend_songs joins reasons with "; " — split them back into bullets.
        reasons = explanation.split("; ") if explanation else []
        print("   Why:")
        for reason in reasons:
            print(f"     • {reason}")

    print("\n" + "=" * width + "\n")


def main() -> None:
    songs = load_songs("../data/songs.csv")

    # Keys must match what score_song reads (favorite_genre / favorite_mood /
    # target_energy / likes_acoustic).
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)
    print_recommendations(recommendations)


if __name__ == "__main__":
    main()
