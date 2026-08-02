"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

import textwrap

from recommender import UserProfile, load_songs, recommend_songs

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

REASON_WRAP_WIDTH = 50


def print_recommendations(recommendations) -> None:
    """Render (song, score, explanation) tuples as a terminal table."""
    rows = []
    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        reasons = textwrap.fill(explanation, width=REASON_WRAP_WIDTH)
        rows.append([rank, song.title, f"{score:.2f}", reasons])

    headers = ["#", "Title", "Score", "Reasons"]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="grid", disable_numparse=True))
        return

    # Fallback: simple ASCII table if tabulate isn't installed.
    # Width is the longest individual line, since cells may wrap to multiple lines.
    col_widths = [
        max(
            len(line)
            for row in ([headers] + rows)
            for line in str(row[i]).split("\n")
        )
        for i in range(len(headers))
    ]

    def format_row(row):
        cells = []
        for i, value in enumerate(row):
            lines = str(value).split("\n")
            cells.append(lines)
        max_lines = max(len(c) for c in cells)
        out_lines = []
        for line_idx in range(max_lines):
            parts = []
            for i, lines in enumerate(cells):
                text = lines[line_idx] if line_idx < len(lines) else ""
                parts.append(text.ljust(col_widths[i]))
            out_lines.append("| " + " | ".join(parts) + " |")
        return "\n".join(out_lines)

    separator = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    print(separator)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
        print(separator)


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


def run_profile(name: str, user: UserProfile, songs, k: int = 5) -> None:
    """Print the recommendations (or the error) produced for a single named profile."""
    print(f"\n=== {name} ===")
    print(f"user = {user}")
    try:
        recommendations = recommend_songs(user, songs, k=k)
        if not recommendations:
            print("(no recommendations returned)")
        else:
            print_recommendations(recommendations)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


def main() -> None:
    """Load songs, generate recommendations for every sample and adversarial profile, and print them."""
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, user in USER_PROFILES.items():
        run_profile(name, user, songs, k=5)


if __name__ == "__main__":
    main()
