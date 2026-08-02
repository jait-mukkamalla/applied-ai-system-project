"""
Adversarial/edge-case UserProfiles used to pin recommend_songs' behavior
under bad or contradictory input. Originally lived in src/main.py as a CLI
demo fixture; moved here once the CLI runner was retired in favor of the
Streamlit app, since test_main_profiles.py depends on this data for
regression coverage.
"""
from src.recommender import UserProfile

USER_PROFILES = {
    # --- Distinct "normal" taste profiles ---
    "High-Energy Pop": UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
        mode="genre_first",
    ),
    "Chill Lofi": UserProfile(
        favorite_genre="lofi",
        favorite_mood="chill",
        target_energy=0.35,
        likes_acoustic=True,
        mode="mood_first",
    ),
    "Deep Intense Rock": UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        target_energy=0.9,
        likes_acoustic=False,
        mode="energy_focused",
    ),
    # --- Adversarial / edge case profiles ---
    "Empty Preferences": UserProfile(),
    "Nonexistent Genre and Mood": UserProfile(
        favorite_genre="vaporwave-death-polka",
        favorite_mood="ecstatic-dread",
        target_energy=0.5,
        likes_acoustic=False,
    ),
    "Out-of-Range Energy": UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=5.0,
        likes_acoustic=False,
    ),
    "Negative Energy": UserProfile(
        favorite_genre="metal",
        favorite_mood="angry",
        target_energy=-2.0,
        likes_acoustic=True,
    ),
    "Contradictory Acoustic Metal": UserProfile(
        favorite_genre="metal",
        favorite_mood="angry",
        target_energy=0.95,
        likes_acoustic=True,
    ),
    "Invalid Scoring Mode": UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        mode="vibes_based",
    ),
}
