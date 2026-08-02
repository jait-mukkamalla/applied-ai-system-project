"""
RAG song source: fetches a large sample of real tracks (with real audio
features) from a public dataset to supplement the local CSV fallback.

Design (per the Phase 3 plan):
  - The source dataset (tidytuesday's Spotify Songs dataset) has real
    energy/valence/danceability/acousticness/tempo but no mood label, so
    mood is derived by nearest-match against recommender.MOOD_TARGETS --
    the same archetypes score_song already uses.
  - Every fetched row is routed through sanitize_songs (src/recommender.py)
    -- no RAG-specific validation logic lives here.
  - Any fetch/parse failure raises RagFetchError; get_song_pool() catches
    it and falls back to the local CSV rather than propagating.
"""
import csv
import io
import logging
import urllib.error
import urllib.request
from typing import Dict, List

try:
    # Matches main.py's convention: works when src/ itself is on sys.path
    # (e.g. `python src/main.py`).
    from recommender import MOOD_TARGETS, Song, load_songs, sanitize_songs
except ImportError:
    # Matches tests' convention: works when the repo root is on sys.path
    # and src/ is imported as a package (e.g. `from src import rag`).
    from src.recommender import MOOD_TARGETS, Song, load_songs, sanitize_songs

logger = logging.getLogger(__name__)

DEFAULT_SOURCE_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/"
    "data/2020/2020-01-21/spotify_songs.csv"
)
DEFAULT_SAMPLE_SIZE = 4000
DEFAULT_TIMEOUT_SECONDS = 15.0

# Offset RAG-assigned ids well clear of the local CSV's id range (currently
# < 200) so combining both pools never collides on id.
RAG_ID_OFFSET = 100_000


class RagFetchError(Exception):
    """Raised when the RAG source can't be fetched or parsed. Callers should
    catch this and fall back to the local CSV, not let it propagate."""


def _download(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "applied-ai-system-project/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise RagFetchError(f"unexpected HTTP status {status} from {url}")
        return response.read().decode("utf-8")


def _safe_float(value, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _derive_mood(valence: float, danceability: float) -> str:
    """Nearest-match a song's real valence/danceability to the closest MOOD_TARGETS archetype."""
    def distance(target):
        target_valence, target_danceability = target
        return (target_valence - valence) ** 2 + (target_danceability - danceability) ** 2

    return min(MOOD_TARGETS, key=lambda mood: distance(MOOD_TARGETS[mood]))


def _select_sample(raw_text: str, sample_size: int) -> List[Dict]:
    """Dedupes by (title, artist) and stratifies the sample evenly across genre."""
    rows = list(csv.DictReader(io.StringIO(raw_text)))

    seen = set()
    unique_rows = []
    for row in rows:
        title = (row.get("track_name") or "").strip().lower()
        artist = (row.get("track_artist") or "").strip().lower()
        if not title or not artist or (title, artist) in seen:
            continue
        seen.add((title, artist))
        unique_rows.append(row)

    by_genre: Dict[str, List[Dict]] = {}
    for row in unique_rows:
        by_genre.setdefault(row.get("playlist_genre") or "unknown", []).append(row)

    genres = sorted(by_genre)
    if not genres:
        return []

    per_genre_quota = max(1, sample_size // len(genres))
    sampled = []
    for genre in genres:
        sampled.extend(by_genre[genre][:per_genre_quota])
    return sampled[:sample_size]


def _row_to_raw_song(row: Dict, song_id: int) -> Dict:
    valence = _safe_float(row.get("valence"))
    danceability = _safe_float(row.get("danceability"))
    return {
        "id": song_id,
        "title": row.get("track_name"),
        "artist": row.get("track_artist"),
        "genre": row.get("playlist_genre"),
        "mood": _derive_mood(valence, danceability),
        "energy": row.get("energy"),
        "tempo_bpm": row.get("tempo"),
        "valence": valence,
        "danceability": danceability,
        "acousticness": row.get("acousticness"),
    }


def fetch_songs_from_rag(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    source_url: str = DEFAULT_SOURCE_URL,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> List[Song]:
    """
    Fetches and sanitizes a sample of real tracks from the RAG source.
    Raises RagFetchError on any network or parsing failure.
    """
    try:
        raw_text = _download(source_url, timeout)
    except RagFetchError:
        raise
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RagFetchError(f"failed to fetch RAG source '{source_url}': {exc}") from exc

    try:
        rows = _select_sample(raw_text, sample_size)
    except csv.Error as exc:
        raise RagFetchError(f"failed to parse RAG source '{source_url}': {exc}") from exc

    raw_songs = [_row_to_raw_song(row, song_id) for song_id, row in enumerate(rows, start=RAG_ID_OFFSET)]
    return sanitize_songs(raw_songs)


def get_song_pool(
    csv_path: str = "data/songs.csv",
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    combine_with_csv: bool = False,
) -> List[Song]:
    """
    Returns the working song pool. RAG is the primary/supplemental source;
    the local CSV is the offline/dev fallback used when RAG fails, and can
    optionally be combined with a successful RAG fetch for extra coverage
    of the hand-tuned dev rows.
    """
    try:
        rag_songs = fetch_songs_from_rag(sample_size=sample_size)
    except RagFetchError as exc:
        logger.warning("RAG fetch failed (%s); falling back to local CSV '%s'", exc, csv_path)
        return load_songs(csv_path)

    if combine_with_csv:
        return load_songs(csv_path) + rag_songs
    return rag_songs
