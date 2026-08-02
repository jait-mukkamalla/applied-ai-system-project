import csv
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MODE = "balanced"

# Song feature fields expected in [0.0, 1.0].
UNIT_RANGE_FIELDS = ("energy", "valence", "danceability", "acousticness")
DEFAULT_UNIT_VALUE = 0.5
DEFAULT_TEMPO_BPM = 120.0

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

    All fields are optional with defaults, mirroring the fallbacks
    score_song previously applied via dict.get(key, default). This is what
    lets a profile carry only structured prefs, only a free-text query, or
    both, with no explicit mode flag.
    """
    favorite_genre: Optional[str] = None
    favorite_mood: Optional[str] = None
    target_energy: float = 0.5
    likes_acoustic: bool = False
    mode: str = DEFAULT_MODE
    query_text: Optional[str] = None

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k songs recommended for the given user profile."""
        recommendations = recommend_songs(user, self.songs, k=k)
        return [song for song, _score, _explanation in recommendations]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain why a given song was recommended to the user."""
        _score, reasons = score_song(user, song)
        return ", ".join(reasons) if reasons else "No strong matches on your preferences"

def _to_float(value, default: float) -> float:
    """Coerce value to float, falling back to default on missing/unparsable input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _clamp_unit(value: float) -> float:
    """Clamp a feature value into [0.0, 1.0], rescaling values that look like a 0-100 scale."""
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return max(0.0, min(1.0, value))

def sanitize_song(raw: Dict) -> Optional[Song]:
    """
    Validates and normalizes a single raw song row into a Song instance.

    This is the single ingestion boundary for song data, regardless of
    whether it originates from CSV (load_songs) or RAG (src/rag.py) — both
    route through here so scoring logic never has to defend against bad
    input. `id` is the only field treated as load-bearing: without a valid
    id the row can't be identified, so it's dropped (returns None). Every
    other field is coerced/clamped/defaulted rather than rejected.
    """
    if not raw:
        return None

    try:
        song_id = int(raw["id"])
    except (KeyError, TypeError, ValueError):
        return None

    title = str(raw.get("title") or "Unknown Title")
    artist = str(raw.get("artist") or "Unknown Artist")
    genre = str(raw.get("genre") or "unknown")
    mood = str(raw.get("mood") or "unknown")

    unit_values = {
        field: _clamp_unit(_to_float(raw.get(field), DEFAULT_UNIT_VALUE))
        for field in UNIT_RANGE_FIELDS
    }

    tempo_bpm = _to_float(raw.get("tempo_bpm"), DEFAULT_TEMPO_BPM)
    if tempo_bpm <= 0:
        tempo_bpm = DEFAULT_TEMPO_BPM

    return Song(
        id=song_id,
        title=title,
        artist=artist,
        genre=genre,
        mood=mood,
        tempo_bpm=tempo_bpm,
        **unit_values,
    )

def sanitize_songs(raw_list: List[Dict]) -> List[Song]:
    """Sanitizes a list of raw song rows, dropping unsalvageable ones and logging the count."""
    sanitized = []
    dropped = 0
    for raw in raw_list:
        song = sanitize_song(raw)
        if song is None:
            dropped += 1
            continue
        sanitized.append(song)

    if dropped:
        logger.warning("sanitize_songs: dropped %d of %d row(s)", dropped, len(raw_list))

    return sanitized

def load_songs(csv_path: str) -> List[Song]:
    """
    Loads songs from a CSV file, routed through sanitize_songs.
    Required by src/main.py
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)
    return sanitize_songs(raw_rows)

# Mood -> implied (valence_target, danceability_target).
# Lets score_song reason about valence/danceability even though UserProfile /
# user_prefs only carry a favorite_mood, not those raw feature values.
MOOD_TARGETS: Dict[str, Tuple[float, float]] = {
    "happy": (0.80, 0.80),
    "chill": (0.55, 0.55),
    "intense": (0.60, 0.85),
    "relaxed": (0.65, 0.55),
    "moody": (0.50, 0.70),
    "focused": (0.55, 0.55),
    "energetic": (0.65, 0.85),
    "peaceful": (0.55, 0.20),
    "nostalgic": (0.55, 0.40),
    "angry": (0.30, 0.50),
    "romantic": (0.70, 0.65),
    "euphoric": (0.80, 0.90),
    "warm": (0.70, 0.55),
    "laid-back": (0.75, 0.70),
    "melancholic": (0.30, 0.40),
    "rebellious": (0.55, 0.60),
}
DEFAULT_MOOD_TARGET: Tuple[float, float] = (0.60, 0.60)

# Algorithm Recipe v2 weight presets, one per scoring mode (each must sum to 1.0)
WEIGHT_PRESETS = {
    "balanced": {
        "genre": 0.25,
        "mood": 0.20,
        "energy": 0.20,
        "valence": 0.15,
        "danceability": 0.10,
        "acousticness": 0.10,
    },
    "genre_first": {
        "genre": 0.45,
        "mood": 0.15,
        "energy": 0.15,
        "valence": 0.10,
        "danceability": 0.075,
        "acousticness": 0.075,
    },
    "mood_first": {
        "genre": 0.15,
        "mood": 0.40,
        "energy": 0.10,
        "valence": 0.20,
        "danceability": 0.10,
        "acousticness": 0.05,
    },
    "energy_focused": {
        "genre": 0.10,
        "mood": 0.10,
        "energy": 0.40,
        "valence": 0.15,
        "danceability": 0.20,
        "acousticness": 0.05,
    },
}

def score_song(user: UserProfile, song: Song) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    Algorithm Recipe v2 (weighted feature distance):
      - genre / mood: exact match (1.0 or 0.0)
      - energy: 1 - abs(song.energy - target_energy)
      - valence / danceability: 1 - abs(song.value - implied target from favorite mood)
      - acousticness: acousticness if likes_acoustic else (1 - acousticness)
      score = 100 * sum(weight_i * match_i)

    user.mode selects which entry of WEIGHT_PRESETS to weight the
    features by (e.g. "genre_first", "mood_first", "energy_focused");
    defaults to DEFAULT_MODE ("balanced") when absent.
    """
    mode = user.mode
    if mode not in WEIGHT_PRESETS:
        valid = ", ".join(sorted(WEIGHT_PRESETS))
        raise ValueError(f"Unknown scoring mode '{mode}'. Valid modes: {valid}")
    weights = WEIGHT_PRESETS[mode]

    favorite_genre = user.favorite_genre
    favorite_mood = user.favorite_mood
    target_energy = user.target_energy
    likes_acoustic = user.likes_acoustic

    valence_target, danceability_target = MOOD_TARGETS.get(favorite_mood, DEFAULT_MOOD_TARGET)

    genre_match = 1.0 if song.genre == favorite_genre else 0.0
    mood_match = 1.0 if song.mood == favorite_mood else 0.0
    energy_match = 1.0 - abs(song.energy - target_energy)
    valence_match = 1.0 - abs(song.valence - valence_target)
    danceability_match = 1.0 - abs(song.danceability - danceability_target)

    acousticness = song.acousticness
    acousticness_match = acousticness if likes_acoustic else (1.0 - acousticness)

    matches = {
        "genre": genre_match,
        "mood": mood_match,
        "energy": energy_match,
        "valence": valence_match,
        "danceability": danceability_match,
        "acousticness": acousticness_match,
    }

    contributions = {feature: weights[feature] * match for feature, match in matches.items()}
    score = round(sum(contributions.values()) * 100, 2)

    reason_templates = {
        "genre": lambda: f"genre '{song.genre}' matches your favorite genre",
        "mood": lambda: f"mood '{song.mood}' matches your favorite mood",
        "energy": lambda: f"energy {song.energy:.2f} is close to your target {target_energy:.2f}",
        "valence": lambda: f"valence {song.valence:.2f} fits the '{favorite_mood}' mood profile",
        "danceability": lambda: f"danceability {song.danceability:.2f} fits the '{favorite_mood}' mood profile",
        "acousticness": lambda: (
            f"acousticness {acousticness:.2f} matches your preference for acoustic tracks"
            if likes_acoustic
            else f"acousticness {acousticness:.2f} matches your preference for non-acoustic tracks"
        ),
    }

    top_features = sorted(contributions, key=contributions.get, reverse=True)[:3]
    reasons = [reason_templates[feature]() for feature in top_features if matches[feature] > 0]

    return score, reasons

MAX_SONGS_PER_ARTIST = 2

def recommend_songs(user: UserProfile, songs: List[Song], k: int = 5) -> List[Tuple[Song, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py

    Scores every song, ranks highest-to-lowest, then applies a diversity cap
    so no artist appears more than MAX_SONGS_PER_ARTIST times in the top k.

    user.mode (see score_song) selects the scoring weight preset.
    """
    scored = sorted(
        ((song, *score_song(user, song)) for song in songs),
        key=lambda scored_song: scored_song[1],
        reverse=True,
    )

    artist_counts: Dict[str, int] = {}
    recommendations: List[Tuple[Song, float, str]] = []

    for song, score, reasons in scored:
        if len(recommendations) == k:
            break

        artist = song.artist
        if artist_counts.get(artist, 0) >= MAX_SONGS_PER_ARTIST:
            continue

        explanation = ", ".join(reasons) if reasons else "No strong matches on your preferences"
        recommendations.append((song, score, explanation))
        artist_counts[artist] = artist_counts.get(artist, 0) + 1

    return recommendations

# How many candidates search_similar retrieves before recommend_songs ranks
# and diversity-caps them down to k. Needs headroom above k so structured
# scoring/diversity capping has room to reorder within the semantic matches,
# not just rubber-stamp the embedding similarity order.
DEFAULT_RETRIEVAL_POOL_SIZE = 20

def get_recommendations(
    user: UserProfile,
    songs: List[Song],
    k: int = 5,
    retrieval_pool_size: int = DEFAULT_RETRIEVAL_POOL_SIZE,
) -> List[Tuple[Song, float, str]]:
    """
    Single entry point for producing recommendations, regardless of whether
    a profile carries structured prefs, a free-text query, or both.

    Mode is inferred from which fields are populated, not an explicit flag:
    if user.query_text is set, it's used to semantically narrow `songs` to
    a candidate pool via the vector store first. recommend_songs (structured
    scoring + diversity cap) always runs afterward, unchanged, over whatever
    candidate pool results. Callers (src/main.py, the Streamlit app) never
    need to branch on mode themselves.

    Retrieval is a pre-filter, not a scoring replacement: if it can't narrow
    anything (e.g. an empty/unpopulated vector store), scoring falls back to
    the full `songs` list rather than silently returning zero recommendations.
    """
    candidate_pool = songs

    if user.query_text:
        try:
            from vector_store import search_similar
        except ImportError:
            from src.vector_store import search_similar

        pool_size = max(retrieval_pool_size, k)
        candidate_pool = search_similar(user.query_text, k=pool_size) or songs

    return recommend_songs(user, candidate_pool, k=k)
