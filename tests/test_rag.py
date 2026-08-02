from src import rag
from src.recommender import Song

FAKE_CSV = """track_id,track_name,track_artist,playlist_genre,danceability,energy,acousticness,valence,tempo
t1,Real Track One,Real Artist A,pop,0.80,0.80,0.10,0.80,120
t2,Real Track Two,Real Artist B,rock,0.50,0.60,0.20,0.30,140
t3,Bad Row,Real Artist C,rock,not-a-number,0.60,0.20,0.30,140
"""


def test_fetch_songs_from_rag_parses_and_derives_mood(monkeypatch):
    monkeypatch.setattr(rag, "_download", lambda url, timeout: FAKE_CSV)

    songs = rag.fetch_songs_from_rag(sample_size=10)

    assert all(isinstance(song, Song) for song in songs)
    titles = {song.title for song in songs}
    assert "Real Track One" in titles
    assert "Real Track Two" in titles
    # High valence/danceability (0.80/0.80) should derive the "happy" archetype.
    happy_song = next(song for song in songs if song.title == "Real Track One")
    assert happy_song.mood == "happy"


def test_fetch_songs_from_rag_raises_on_download_failure(monkeypatch):
    def boom(url, timeout):
        raise OSError("network unreachable")

    monkeypatch.setattr(rag, "_download", boom)

    try:
        rag.fetch_songs_from_rag(sample_size=10)
        assert False, "expected RagFetchError"
    except rag.RagFetchError:
        pass


def test_get_song_pool_falls_back_to_csv_on_rag_failure(monkeypatch, tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,Fallback Song,Fallback Artist,pop,happy,0.8,120,0.8,0.8,0.1\n",
        encoding="utf-8",
    )

    def boom(sample_size=rag.DEFAULT_SAMPLE_SIZE, source_url=rag.DEFAULT_SOURCE_URL, timeout=rag.DEFAULT_TIMEOUT_SECONDS):
        raise rag.RagFetchError("simulated RAG outage")

    monkeypatch.setattr(rag, "fetch_songs_from_rag", boom)

    pool = rag.get_song_pool(csv_path=str(csv_path))

    assert len(pool) == 1
    assert pool[0].title == "Fallback Song"


def test_get_song_pool_combines_csv_and_rag_when_requested(monkeypatch, tmp_path):
    csv_path = tmp_path / "songs.csv"
    csv_path.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,CSV Song,CSV Artist,pop,happy,0.8,120,0.8,0.8,0.1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag, "_download", lambda url, timeout: FAKE_CSV)

    pool = rag.get_song_pool(csv_path=str(csv_path), combine_with_csv=True)

    titles = {song.title for song in pool}
    assert "CSV Song" in titles
    assert "Real Track One" in titles
