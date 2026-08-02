import hashlib

from src import vector_store
from src.recommender import Song


def fake_embed_texts(texts):
    """Deterministic fake embeddings: same text always maps to the same vector,
    and vectors for similar text (here, just equal text) land close together.
    Keeps tests fast and independent of the real sentence-transformers model."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([b / 255.0 for b in digest[:8]])
    return vectors


def make_song(song_id, title, artist="Some Artist", genre="pop", mood="happy"):
    return Song(
        id=song_id,
        title=title,
        artist=artist,
        genre=genre,
        mood=mood,
        energy=0.7,
        tempo_bpm=120.0,
        valence=0.7,
        danceability=0.7,
        acousticness=0.2,
    )


def use_isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_store, "PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(vector_store, "_client", None)
    monkeypatch.setattr(vector_store, "_collection", None)
    monkeypatch.setattr(vector_store, "embed_texts", fake_embed_texts)


def test_embed_songs_is_incremental(monkeypatch, tmp_path):
    use_isolated_store(monkeypatch, tmp_path)
    songs = [make_song(1, "Song One"), make_song(2, "Song Two")]

    first_count = vector_store.embed_songs(songs)
    second_count = vector_store.embed_songs(songs)  # same ids, should embed nothing new

    assert first_count == 2
    assert second_count == 0
    assert vector_store.get_collection().count() == 2


def test_embed_songs_only_embeds_new_ids(monkeypatch, tmp_path):
    use_isolated_store(monkeypatch, tmp_path)
    vector_store.embed_songs([make_song(1, "Song One")])

    added = vector_store.embed_songs([make_song(1, "Song One"), make_song(2, "Song Two")])

    assert added == 1
    assert vector_store.get_collection().count() == 2


def test_search_similar_returns_matching_song(monkeypatch, tmp_path):
    use_isolated_store(monkeypatch, tmp_path)
    songs = [make_song(1, "Exact Match Title"), make_song(2, "Totally Different Track")]
    vector_store.embed_songs(songs)

    # Fake embeddings hash the exact document text, so querying with the
    # same text build_song_text() would produce for song 1 must return it first.
    from src.embeddings import build_song_text
    results = vector_store.search_similar(build_song_text(songs[0]), k=1)

    assert len(results) == 1
    assert results[0].id == 1


def test_search_similar_returns_empty_for_empty_store(monkeypatch, tmp_path):
    use_isolated_store(monkeypatch, tmp_path)

    assert vector_store.search_similar("anything", k=5) == []
