from src.recommender import Song, UserProfile, Recommender, sanitize_song, sanitize_songs

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_sanitize_song_fills_missing_fields_with_defaults():
    song = sanitize_song({"id": "7", "title": "Untitled"})

    assert song.id == 7
    assert song.title == "Untitled"
    assert song.artist == "Unknown Artist"
    assert song.genre == "unknown"
    assert song.mood == "unknown"
    assert song.energy == 0.5
    assert song.tempo_bpm == 120.0


def test_sanitize_song_clamps_out_of_range_values():
    song = sanitize_song({
        "id": 1,
        "title": "Loud",
        "artist": "Someone",
        "genre": "pop",
        "mood": "happy",
        "energy": "150",  # looks like a 0-100 scale, should rescale to 1.5 -> clamp to 1.0
        "tempo_bpm": "-10",
        "valence": "1000",  # far outside any plausible scale, should clamp to 1.0
        "danceability": "0.5",
        "acousticness": "0.5",
    })

    assert song.energy == 1.0
    assert song.tempo_bpm == 120.0
    assert song.valence == 1.0


def test_sanitize_song_drops_unsalvageable_row():
    assert sanitize_song({"title": "No ID"}) is None
    assert sanitize_song({}) is None


def test_sanitize_songs_filters_dropped_rows():
    raw_rows = [
        {"id": 1, "title": "Good Row"},
        {"title": "Missing ID"},
    ]
    songs = sanitize_songs(raw_rows)

    assert len(songs) == 1
    assert songs[0].title == "Good Row"
