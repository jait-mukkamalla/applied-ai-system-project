"""
Persistent Chroma-backed vector store for semantic song retrieval.

embed_songs() is incremental: songs whose id is already present in the
collection are skipped, so repeated calls across app runs only pay the
embedding cost for genuinely new songs (important once the pool is
CSV + several thousand RAG-fetched tracks). search_similar() embeds a
free-text query and returns the nearest songs, reconstructed from the
metadata stored alongside each embedding.
"""
from typing import List

try:
    from embeddings import build_song_text, embed_texts
    from recommender import Song
except ImportError:
    from src.embeddings import build_song_text, embed_texts
    from src.recommender import Song

PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "songs"

_client = None
_collection = None


def get_collection():
    """Lazily creates/opens the persistent Chroma collection."""
    global _client, _collection
    if _collection is None:
        import chromadb
        _client = chromadb.PersistentClient(path=PERSIST_DIR)
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def _song_to_metadata(song: Song) -> dict:
    return {
        "title": song.title,
        "artist": song.artist,
        "genre": song.genre,
        "mood": song.mood,
        "energy": song.energy,
        "tempo_bpm": song.tempo_bpm,
        "valence": song.valence,
        "danceability": song.danceability,
        "acousticness": song.acousticness,
    }


def _song_from_metadata(song_id: str, metadata: dict) -> Song:
    return Song(
        id=int(song_id),
        title=metadata["title"],
        artist=metadata["artist"],
        genre=metadata["genre"],
        mood=metadata["mood"],
        energy=metadata["energy"],
        tempo_bpm=metadata["tempo_bpm"],
        valence=metadata["valence"],
        danceability=metadata["danceability"],
        acousticness=metadata["acousticness"],
    )


def embed_songs(songs: List[Song]) -> int:
    """
    Embeds and stores any song not already present in the collection (by id).
    Returns how many songs were newly embedded.
    """
    collection = get_collection()
    if not songs:
        return 0

    all_ids = [str(song.id) for song in songs]
    existing_ids = set(collection.get(ids=all_ids, include=[])["ids"])
    new_songs = [song for song in songs if str(song.id) not in existing_ids]
    if not new_songs:
        return 0

    texts = [build_song_text(song) for song in new_songs]
    vectors = embed_texts(texts)
    collection.upsert(
        ids=[str(song.id) for song in new_songs],
        embeddings=vectors,
        documents=texts,
        metadatas=[_song_to_metadata(song) for song in new_songs],
    )
    return len(new_songs)


def search_similar(query_text: str, k: int = 5) -> List[Song]:
    """Embeds query_text and returns the k most similar songs in the store."""
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    query_vector = embed_texts([query_text])[0]
    results = collection.query(query_embeddings=[query_vector], n_results=min(k, count))

    return [
        _song_from_metadata(song_id, metadata)
        for song_id, metadata in zip(results["ids"][0], results["metadatas"][0])
    ]
