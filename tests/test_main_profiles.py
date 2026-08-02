"""
Pins the behavior of the adversarial USER_PROFILES in src/main.py -- these
were manually verified to behave sensibly in Phase 0 but had no automated
regression coverage.

main.py imports `from recommender import ...` (a bare module name, since it's
designed to run as `python src/main.py` with src/ on sys.path), so src/ is
added to sys.path here before importing it, mirroring how main.py is
actually executed.
"""
import os
import sys

import pytest

from src.recommender import load_songs, recommend_songs

SONGS = load_songs("data/songs.csv")

# main.py imports `from recommender import ...` (a bare module name) because
# it's meant to run as `python src/main.py` with src/ on sys.path. The path
# entry is removed again right after the import so it doesn't linger for the
# rest of the test session -- otherwise get_recommendations' lazy
# `import vector_store` (src/recommender.py) would also resolve bare, giving
# a second, unpatched module object that other tests' monkeypatching misses.
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _src_dir)
try:
    import main as main_module
finally:
    sys.path.remove(_src_dir)
    # main.py's bare `import recommender` caches a *second* Song/UserProfile
    # module under sys.modules["recommender"], independent of src.recommender.
    # Left in place, it'd make other modules' `try: import X except: import
    # src.X` fallback (e.g. src/rag.py) silently resolve to this bare copy
    # for the rest of the test session, breaking their isinstance checks.
    sys.modules.pop("recommender", None)


def test_invalid_scoring_mode_raises_value_error():
    user = main_module.USER_PROFILES["Invalid Scoring Mode"]
    with pytest.raises(ValueError):
        recommend_songs(user, SONGS, k=5)


def test_empty_preferences_still_produces_a_ranked_list():
    user = main_module.USER_PROFILES["Empty Preferences"]
    results = recommend_songs(user, SONGS, k=5)

    assert len(results) == 5
    scores = [score for _song, score, _explanation in results]
    assert scores == sorted(scores, reverse=True)


def test_nonexistent_genre_and_mood_ranks_without_exact_matches():
    user = main_module.USER_PROFILES["Nonexistent Genre and Mood"]
    results = recommend_songs(user, SONGS, k=5)

    assert len(results) == 5
    for song, _score, _explanation in results:
        assert song.genre != user.favorite_genre
        assert song.mood != user.favorite_mood


def test_out_of_range_energy_still_ranks():
    user = main_module.USER_PROFILES["Out-of-Range Energy"]
    results = recommend_songs(user, SONGS, k=5)
    assert len(results) == 5


def test_negative_energy_still_ranks():
    user = main_module.USER_PROFILES["Negative Energy"]
    results = recommend_songs(user, SONGS, k=5)
    assert len(results) == 5


def test_contradictory_acoustic_metal_still_ranks():
    user = main_module.USER_PROFILES["Contradictory Acoustic Metal"]
    results = recommend_songs(user, SONGS, k=5)
    assert len(results) == 5
