"""
Streamlit UI for the Music Recommender.

Thin presentation layer over get_recommendations() (src/recommender.py) --
the UI never branches on structured/free-text/combined mode itself; it just
populates a UserProfile and lets get_recommendations infer what to do from
which fields are set.
"""
import streamlit as st

from src.rag import DEFAULT_SAMPLE_SIZE, get_song_pool
from src.recommender import UserProfile, WEIGHT_PRESETS, get_recommendations, load_songs
from src.vector_store import embed_songs

CSV_PATH = "data/songs.csv"

SOURCE_CSV = "CSV (fast, local dev set)"
SOURCE_RAG = "RAG (live fetch, ~4k real tracks)"
SOURCE_COMBINED = "Combined (RAG + CSV)"
SOURCE_OPTIONS = [SOURCE_CSV, SOURCE_RAG, SOURCE_COMBINED]

GENRE_OPTIONS = [
    "ambient", "blues", "classical", "country", "edm", "folk", "hip hop",
    "house", "indie pop", "jazz", "latin", "lofi", "metal", "pop", "punk",
    "r&b", "rap", "reggae", "rock", "synthwave",
]
MOOD_OPTIONS = [
    "angry", "chill", "energetic", "euphoric", "focused", "happy", "intense",
    "laid-back", "melancholic", "moody", "nostalgic", "peaceful",
    "rebellious", "relaxed", "romantic", "warm",
]
NO_PREFERENCE = "(no preference)"


@st.cache_resource(show_spinner="Loading songs and building the semantic index...")
def load_and_index_songs(source: str, sample_size: int):
    """
    Loads the song pool for the chosen source and embeds it into the vector
    store. Cached per (source, sample_size) so this only runs once per
    combination for the life of the app process, not on every rerun/click --
    embedding thousands of RAG songs is too slow to repeat per interaction.
    """
    if source == SOURCE_CSV:
        songs = load_songs(CSV_PATH)
    elif source == SOURCE_RAG:
        songs = get_song_pool(csv_path=CSV_PATH, sample_size=sample_size, combine_with_csv=False)
    else:
        songs = get_song_pool(csv_path=CSV_PATH, sample_size=sample_size, combine_with_csv=True)

    embed_songs(songs)
    return songs


st.set_page_config(page_title="Music Recommender", page_icon="🎵")
st.title("🎵 Music Recommender")

with st.sidebar:
    st.header("Song source")
    source = st.selectbox("Where should songs come from?", SOURCE_OPTIONS, index=0)
    sample_size = DEFAULT_SAMPLE_SIZE
    if source in (SOURCE_RAG, SOURCE_COMBINED):
        sample_size = st.number_input(
            "RAG sample size", min_value=100, max_value=10000,
            value=DEFAULT_SAMPLE_SIZE, step=100,
        )

try:
    songs = load_and_index_songs(source, sample_size)
except Exception as exc:
    st.error(f"Failed to load songs: {exc}")
    st.stop()

st.caption(f"{len(songs)} songs loaded from: {source}")

st.header("Your preferences")
col1, col2 = st.columns(2)
with col1:
    favorite_genre = st.selectbox("Favorite genre", [NO_PREFERENCE] + GENRE_OPTIONS)
    target_energy = st.slider("Target energy", 0.0, 1.0, 0.5, step=0.05)
    mode = st.selectbox("Scoring mode", sorted(WEIGHT_PRESETS), index=sorted(WEIGHT_PRESETS).index("balanced"))
with col2:
    favorite_mood = st.selectbox("Favorite mood", [NO_PREFERENCE] + MOOD_OPTIONS)
    likes_acoustic = st.checkbox("I like acoustic tracks", value=False)

query_text = st.text_input(
    "Optional: describe what you're in the mood for (free text)",
    placeholder="e.g. late-night driving music with a dreamy vibe",
)

k = st.slider("How many recommendations?", 1, 20, 5)

if st.button("Get recommendations", type="primary"):
    user = UserProfile(
        favorite_genre=None if favorite_genre == NO_PREFERENCE else favorite_genre,
        favorite_mood=None if favorite_mood == NO_PREFERENCE else favorite_mood,
        target_energy=target_energy,
        likes_acoustic=likes_acoustic,
        mode=mode,
        query_text=query_text or None,
    )

    try:
        recommendations = get_recommendations(user, songs, k=k)
    except Exception as exc:
        st.error(f"Failed to generate recommendations: {exc}")
    else:
        if not recommendations:
            st.info("No recommendations matched your preferences.")
        else:
            st.header("Recommendations")
            for rank, (song, score, explanation) in enumerate(recommendations, start=1):
                with st.container(border=True):
                    st.subheader(f"{rank}. {song.title} — {song.artist}")
                    st.write(f"**Score:** {score:.2f}  |  **Genre:** {song.genre}  |  **Mood:** {song.mood}")
                    st.caption(explanation)
