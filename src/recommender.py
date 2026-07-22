import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file into a list of dictionaries.

    Numeric fields are converted from strings to int/float so downstream
    scoring can do math on them directly. Required by src/main.py
    """
    # Fields that must be coerced from CSV strings into numbers.
    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}

    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)

    print(len(songs))
    return songs

# --- Scoring configuration (content-based "recipe") ------------------------
# Weights are the maximum points each feature can contribute; they sum to 1.0
# so a full score lands in [0, 1]. Tier 1 (valence/mood/energy) leads; genre is
# a tiebreaker kept below the smallest Tier-1 weight; acousticness is a gate.
WEIGHTS = {
    "valence": 0.24,
    "mood": 0.22,
    "energy": 0.20,
    "genre": 0.12,
    "acousticness": 0.10,
    "tempo_bpm": 0.06,
    "danceability": 0.06,
}

# Catalog BPM range, used to normalize tempo to 0-1 before differencing so it
# can't dominate the other (already 0-1) features.
TEMPO_MIN, TEMPO_MAX = 60.0, 152.0

# Unlisted category pairs fall back to this floor (not 0) so cross-genre /
# cross-mood jumps stay possible instead of being hard-filtered out.
SIM_FLOOR = 0.15

# Mood positions on Russell's circumplex of affect: (valence, arousal), 0-1.
# Similarity is derived from distance, so adjacent moods (happy/euphoric) score
# high instead of the all-or-nothing an exact match would give.
MOOD_COORDS = {
    "euphoric": (0.95, 0.85),
    "happy": (0.85, 0.65),
    "confident": (0.75, 0.80),
    "romantic": (0.78, 0.45),
    "relaxed": (0.68, 0.22),
    "chill": (0.60, 0.28),
    "focused": (0.55, 0.48),
    "nostalgic": (0.48, 0.38),
    "moody": (0.38, 0.52),
    "intense": (0.50, 0.95),
    "melancholic": (0.25, 0.32),
}

# Genre closeness by cultural lineage (symmetric, unlisted pairs use SIM_FLOOR).
# Deliberately NOT derived from audio features, so genre adds an axis of its own
# rather than re-measuring energy/acousticness.
GENRE_SIMILARITY = {
    frozenset({"pop", "indie pop"}): 0.85,
    frozenset({"synthwave", "edm"}): 0.80,
    frozenset({"hip hop", "r&b"}): 0.75,
    frozenset({"lofi", "ambient"}): 0.70,
    frozenset({"rock", "indie pop"}): 0.60,
    frozenset({"folk", "classical"}): 0.60,
    frozenset({"jazz", "r&b"}): 0.60,
    frozenset({"jazz", "folk"}): 0.55,
    frozenset({"pop", "synthwave"}): 0.55,
    frozenset({"pop", "edm"}): 0.55,
    frozenset({"ambient", "classical"}): 0.55,
    frozenset({"jazz", "lofi"}): 0.55,
    frozenset({"synthwave", "ambient"}): 0.55,
    frozenset({"lofi", "r&b"}): 0.50,
    frozenset({"pop", "r&b"}): 0.50,
    frozenset({"hip hop", "edm"}): 0.50,
    frozenset({"pop", "hip hop"}): 0.45,
    frozenset({"pop", "rock"}): 0.45,
    frozenset({"folk", "indie pop"}): 0.45,
    frozenset({"ambient", "jazz"}): 0.45,
}


def _clamp(x: float) -> float:
    """Constrain a value to [0, 1]."""
    return max(0.0, min(1.0, x))


def _numeric_sim(target: float, value: float) -> float:
    """Closeness of two 0-1 values: 1 when equal, 0 at opposite ends."""
    return _clamp(1.0 - abs(target - value))


def _normalize_tempo(bpm: float) -> float:
    """Map a BPM onto 0-1 using the catalog range."""
    return _clamp((bpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN))


def _mood_sim(user_mood: str, song_mood: str) -> float:
    """Circumplex-distance similarity between two moods."""
    if user_mood == song_mood:
        return 1.0
    a, b = MOOD_COORDS.get(user_mood), MOOD_COORDS.get(song_mood)
    if a is None or b is None:
        return SIM_FLOOR
    dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    return _clamp(1.0 - dist)


def _genre_sim(user_genre: str, song_genre: str) -> float:
    """Lineage similarity between two genres."""
    if user_genre == song_genre:
        return 1.0
    return GENRE_SIMILARITY.get(frozenset({user_genre, song_genre}), SIM_FLOOR)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.

    Returns (score, reasons): a weighted-sum score in [0, 1] and the
    human-readable matches that drove it (strongest first). Only the features
    the user actually expressed are scored; their weights are renormalized so
    the score stays in [0, 1] regardless of which preferences are present.
    Required by recommend_songs() and src/main.py
    """
    # Each entry: (weight, sim, reason). We collect only the features the user
    # expressed, then renormalize over whatever we collected.
    parts: List[Tuple[float, float, str]] = []

    # --- Numeric target features: sim = 1 - |target - value| ---
    numeric_feats = [
        ("valence", "target_valence", "valence", "positivity"),
        ("energy", "target_energy", "energy", "energy"),
        ("danceability", "target_danceability", "danceability", "danceability"),
    ]
    for feat, pref_key, song_key, label in numeric_feats:
        target = user_prefs.get(pref_key)
        if target is not None:
            sim = _numeric_sim(target, song[song_key])
            parts.append((WEIGHTS[feat], sim, f"{label} ({song[song_key]:.2f}) near your target"))

    # --- Tempo: normalize BPM before differencing ---
    target_tempo = user_prefs.get("target_tempo_bpm")
    if target_tempo is not None:
        sim = _numeric_sim(_normalize_tempo(target_tempo), _normalize_tempo(song["tempo_bpm"]))
        parts.append((WEIGHTS["tempo_bpm"], sim, f"tempo ({int(song['tempo_bpm'])} BPM) near your target"))

    # --- Mood: circumplex similarity (Tier 1) ---
    user_mood = user_prefs.get("favorite_mood")
    if user_mood:
        sim = _mood_sim(user_mood, song["mood"])
        relation = "matches" if song["mood"] == user_mood else "is close to"
        parts.append((WEIGHTS["mood"], sim, f"mood '{song['mood']}' {relation} your '{user_mood}'"))

    # --- Genre: lineage similarity (tiebreaker) ---
    user_genre = user_prefs.get("favorite_genre")
    if user_genre:
        sim = _genre_sim(user_genre, song["genre"])
        relation = "matches" if song["genre"] == user_genre else "is related to"
        parts.append((WEIGHTS["genre"], sim, f"genre '{song['genre']}' {relation} your '{user_genre}'"))

    # --- Acousticness gate: only when the user has a stated preference ---
    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        if likes_acoustic:
            sim, phrase = song["acousticness"], "acoustic sound, as you prefer"
        else:
            sim, phrase = 1.0 - song["acousticness"], "non-acoustic sound, as you prefer"
        parts.append((WEIGHTS["acousticness"], _clamp(sim), phrase))

    # --- Weighted sum, renormalized over the features actually scored ---
    total_weight = sum(w for w, _, _ in parts)
    if total_weight == 0:
        return 0.0, []
    score = sum(w * sim for w, sim, _ in parts) / total_weight

    # Surface the strongest matches (by contribution) as reasons; always keep at
    # least one so an explanation is never empty.
    ranked = sorted(parts, key=lambda p: p[0] * p[1], reverse=True)
    reasons = [r for _w, sim, r in ranked if sim >= 0.6]
    if not reasons and ranked:
        reasons = [ranked[0][2]]

    return round(score, 4), reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # Score every song once, pairing each with its (score, reasons).
    scored = [(song, *score_song(user_prefs, song)) for song in songs]

    # Best score first; sorted() returns a new list and leaves `songs` untouched.
    scored.sort(key=lambda item: item[1], reverse=True)

    # Keep the top k, turning each reason list into a single explanation string.
    return [
        (song, score, "; ".join(reasons) if reasons else "No strong matches")
        for song, score, reasons in scored[:k]
    ]
