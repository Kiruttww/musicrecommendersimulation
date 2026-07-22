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


# Three deliberately diverse profiles aimed at different corners of the catalog,
# so we can see how the recommender behaves for very different tastes. Keys must
# match what score_song reads (favorite_genre / favorite_mood / target_valence /
# target_energy / target_danceability / target_tempo_bpm / likes_acoustic). Each
# profile leaves some keys out on purpose — score_song renormalizes over only the
# features a user actually expresses.
PROFILES: List[Tuple[str, Dict]] = [
    (
        # The gym/party listener: loud, fast, upbeat, electronic. Should pull the
        # EDM/pop high-energy cluster (Neon Overdrive, Gym Hero, Sunrise City).
        "Hype Machine — high-energy EDM & pop for the gym",
        {
            "favorite_genre": "edm",
            "favorite_mood": "euphoric",
            "target_valence": 0.9,
            "target_energy": 0.95,
            "target_danceability": 0.9,
            "target_tempo_bpm": 130,
            "likes_acoustic": False,
        },
    ),
    (
        # The deep-focus studier: quiet, acoustic, low-energy background music.
        # Should pull the lofi/ambient/classical calm cluster (Library Rain,
        # Spacewalk Thoughts, Focus Flow).
        "Deep Focus — calm, acoustic study music",
        {
            "favorite_genre": "lofi",
            "favorite_mood": "focused",
            "target_energy": 0.35,
            "target_tempo_bpm": 75,
            "likes_acoustic": True,
        },
    ),
    (
        # The late-night romantic: warm, mid-tempo, soulful. Sits between the
        # extremes — should favor r&b/jazz (Velvet Hours, Coffee Shop Stories)
        # and lightly reach into adjacent genres.
        "Late Night Soul — warm, romantic r&b & jazz",
        {
            "favorite_genre": "r&b",
            "favorite_mood": "romantic",
            "target_valence": 0.72,
            "target_energy": 0.55,
            "target_danceability": 0.7,
            "likes_acoustic": True,
        },
    ),
]


def main() -> None:
    songs = load_songs("../data/songs.csv")

    for name, user_prefs in PROFILES:
        print("\n" + "#" * 60)
        print(f"PROFILE: {name}")
        print("#" * 60)
        recommendations = recommend_songs(user_prefs, songs, k=3)
        print_recommendations(recommendations)


if __name__ == "__main__":
    main()
