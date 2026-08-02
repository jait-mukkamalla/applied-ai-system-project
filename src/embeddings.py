"""
Turns a Song into embeddable text, and text into vectors.

Uses a local sentence-transformers model (all-MiniLM-L6-v2) so embedding
runs fully offline with no API key, cost, or rate limit -- appropriate for
a corpus in the low thousands. The model is loaded lazily on first use so
importing this module (or mocking embed_texts in tests) never requires
downloading/loading it.
"""
from typing import List

try:
    from recommender import Song
except ImportError:
    from src.recommender import Song

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Lazily loads and caches the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_song_text(song: Song) -> str:
    """Builds the natural-language description embedded for a song."""
    return f"{song.title} by {song.artist}. Genre: {song.genre}. Mood: {song.mood}."


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeds a batch of texts into vectors."""
    model = get_model()
    return model.encode(list(texts), convert_to_numpy=True).tolist()
