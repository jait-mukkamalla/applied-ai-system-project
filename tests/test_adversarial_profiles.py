"""
Pins the behavior of the adversarial USER_PROFILES in
tests/fixtures/adversarial_profiles.py -- these were manually verified to
behave sensibly in Phase 0 but had no automated regression coverage.
"""
import pytest

from src.recommender import load_songs, recommend_songs
from tests.fixtures.adversarial_profiles import USER_PROFILES

SONGS = load_songs("data/songs.csv")


def test_invalid_scoring_mode_raises_value_error():
    user = USER_PROFILES["Invalid Scoring Mode"]
    with pytest.raises(ValueError):
        recommend_songs(user, SONGS, k=5)


def test_empty_preferences_still_produces_a_ranked_list():
    user = USER_PROFILES["Empty Preferences"]
    results = recommend_songs(user, SONGS, k=5)

    assert len(results) == 5
    scores = [score for _song, score, _explanation in results]
    assert scores == sorted(scores, reverse=True)


def test_nonexistent_genre_and_mood_ranks_without_exact_matches():
    user = USER_PROFILES["Nonexistent Genre and Mood"]
    results = recommend_songs(user, SONGS, k=5)

    assert len(results) == 5
    for song, _score, _explanation in results:
        assert song.genre != user.favorite_genre
        assert song.mood != user.favorite_mood


def test_out_of_range_energy_still_ranks():
    user = USER_PROFILES["Out-of-Range Energy"]
    results = recommend_songs(user, SONGS, k=5)
    assert len(results) == 5


def test_negative_energy_still_ranks():
    user = USER_PROFILES["Negative Energy"]
    results = recommend_songs(user, SONGS, k=5)
    assert len(results) == 5


def test_contradictory_acoustic_metal_still_ranks():
    user = USER_PROFILES["Contradictory Acoustic Metal"]
    results = recommend_songs(user, SONGS, k=5)
    assert len(results) == 5
