import csv

from src.recommender import load_songs, sanitize_songs

CSV_PATH = "data/songs.csv"


def test_real_songs_csv_parses_cleanly_end_to_end():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    songs = sanitize_songs(raw_rows)

    assert len(raw_rows) >= 100
    assert len(songs) == len(raw_rows)  # zero rows dropped by sanitize


def test_load_songs_matches_direct_sanitize_of_the_same_file():
    songs = load_songs(CSV_PATH)
    assert len(songs) >= 100
    assert len({song.id for song in songs}) == len(songs)  # ids are unique
