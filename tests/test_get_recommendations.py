from src import vector_store
from src.recommender import Song, UserProfile, get_recommendations


def make_song(song_id, title, artist, genre, mood, energy=0.5):
    return Song(
        id=song_id,
        title=title,
        artist=artist,
        genre=genre,
        mood=mood,
        energy=energy,
        tempo_bpm=120.0,
        valence=0.5,
        danceability=0.5,
        acousticness=0.5,
    )


SONGS = [
    make_song(1, "Pop Anthem", "Artist A", "pop", "happy", energy=0.9),
    make_song(2, "Rock Ballad", "Artist B", "rock", "melancholic", energy=0.3),
    make_song(3, "Lofi Loop", "Artist C", "lofi", "chill", energy=0.4),
]


def test_structured_only_never_calls_search_similar(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("search_similar should not be called without query_text")

    monkeypatch.setattr(vector_store, "search_similar", fail_if_called)

    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.9)
    results = get_recommendations(user, SONGS, k=2)

    assert len(results) == 2
    assert results[0][0].genre == "pop"


def test_free_text_only_narrows_via_search_similar(monkeypatch):
    narrowed_pool = [SONGS[1]]  # only the rock ballad
    calls = []

    def fake_search_similar(query_text, k=5):
        calls.append((query_text, k))
        return narrowed_pool

    monkeypatch.setattr(vector_store, "search_similar", fake_search_similar)

    user = UserProfile(query_text="a sad slow song")
    results = get_recommendations(user, SONGS, k=5)

    assert calls and calls[0][0] == "a sad slow song"
    # Structured prefs are all defaults here, but the candidate pool should
    # still be confined to what search_similar returned, not the full SONGS list.
    assert [song.id for song, _score, _explanation in results] == [2]


def test_combined_mode_narrows_then_scores_structured_prefs(monkeypatch):
    # Retrieval narrows to 2 candidates; structured prefs should then pick
    # the pop/happy one over the lofi/chill one within that narrowed pool.
    narrowed_pool = [SONGS[0], SONGS[2]]

    monkeypatch.setattr(vector_store, "search_similar", lambda query_text, k=5: narrowed_pool)

    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.9, query_text="energetic")
    results = get_recommendations(user, SONGS, k=5)

    assert [song.id for song, _score, _explanation in results] == [1, 3]


def test_empty_retrieval_falls_back_to_full_pool(monkeypatch):
    monkeypatch.setattr(vector_store, "search_similar", lambda query_text, k=5: [])

    user = UserProfile(favorite_genre="lofi", favorite_mood="chill", query_text="no matches expected")
    results = get_recommendations(user, SONGS, k=5)

    # Empty retrieval shouldn't collapse to zero recommendations -- scoring
    # should fall back to the full song pool.
    assert len(results) == 3
