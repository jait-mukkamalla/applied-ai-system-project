import os

from src.recommender import UserProfile, load_songs, recommend_songs

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_songs.csv")


def test_load_songs_reads_fixture_end_to_end():
    songs = load_songs(FIXTURE_PATH)

    assert len(songs) == 10
    assert {song.id for song in songs} == set(range(1, 11))
    # Three artists deliberately appear twice, to exercise MAX_SONGS_PER_ARTIST.
    artists = [song.artist for song in songs]
    assert artists.count("The Ramblers") == 2
    assert artists.count("Mara Lin") == 2
    assert artists.count("Steel Wolves") == 2


def test_fixture_has_a_high_acousticness_song():
    songs = load_songs(FIXTURE_PATH)
    quiet_room = next(song for song in songs if song.title == "Quiet Room")
    assert quiet_room.acousticness >= 0.9


def test_diversity_cap_limits_songs_per_artist_across_fixture():
    songs = load_songs(FIXTURE_PATH)
    # Steel Wolves has the two highest-energy songs in the fixture; a user
    # who wants high energy should still only get MAX_SONGS_PER_ARTIST of them.
    user = UserProfile(target_energy=0.9, mode="energy_focused")
    results = recommend_songs(user, songs, k=10)

    artist_counts = {}
    for song, _score, _explanation in results:
        artist_counts[song.artist] = artist_counts.get(song.artist, 0) + 1

    assert all(count <= 2 for count in artist_counts.values())
