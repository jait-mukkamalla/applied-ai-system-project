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


RANK_BADGES = {1: "🥇", 2: "🥈", 3: "🥉"}

st.set_page_config(page_title="Tunecraft", page_icon="🎵", layout="wide")

st.markdown(
    """
    <style>
      div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.08);
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
      }
      div[data-testid="stAlert"] {
        border-radius: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎵 Tunecraft")
st.caption("Tell it what you like, or just describe a vibe, and get a ranked, explained playlist.")

with st.sidebar:
    st.header("Song source")
    source = st.selectbox("Where should songs come from?", SOURCE_OPTIONS, index=SOURCE_OPTIONS.index(SOURCE_COMBINED))
    sample_size = DEFAULT_SAMPLE_SIZE
    if source in (SOURCE_RAG, SOURCE_COMBINED):
        sample_size = st.number_input(
            "RAG sample size", min_value=100, max_value=10000,
            value=1000, step=500,
        )
    st.caption("Switching source rebuilds the semantic index the first time it's used.")

try:
    songs = load_and_index_songs(source, sample_size)
except Exception as exc:
    st.error(f"⚠️ Failed to load songs: {exc}")
    st.stop()

# Genre/mood options are derived from whatever pool actually loaded, rather
# than a fixed list -- the CSV and RAG sources use different genre/mood
# taxonomies, so a static list would offer choices with zero matching songs.
genre_options = sorted({song.genre for song in songs})
mood_options = sorted({song.mood for song in songs})

metric_cols = st.columns(3)
metric_cols[0].metric("Songs loaded", len(songs))
metric_cols[1].metric("Genres", len(genre_options))
metric_cols[2].metric("Moods", len(mood_options))

st.divider()

with st.container(border=True):
    st.subheader("Your preferences")
    col1, col2 = st.columns(2)
    with col1:
        favorite_genre = st.selectbox("Favorite genre", [NO_PREFERENCE] + genre_options)
        target_energy = st.slider("Target energy", 0.0, 1.0, 0.5, step=0.05)
        mode = st.selectbox("Scoring mode", sorted(WEIGHT_PRESETS), index=sorted(WEIGHT_PRESETS).index("balanced"))
    with col2:
        favorite_mood = st.selectbox("Favorite mood", [NO_PREFERENCE] + mood_options)
        likes_acoustic = st.checkbox("I like acoustic tracks", value=False)

    query_text = st.text_input(
        "Optional: describe what you're in the mood for (free text)",
        placeholder="e.g. late-night driving music with a dreamy vibe",
    )

    k = st.slider("How many recommendations?", 1, 20, 5)
    submitted = st.button("Get recommendations", type="primary", use_container_width=True)

if submitted:
    user = UserProfile(
        favorite_genre=None if favorite_genre == NO_PREFERENCE else favorite_genre,
        favorite_mood=None if favorite_mood == NO_PREFERENCE else favorite_mood,
        target_energy=target_energy,
        likes_acoustic=likes_acoustic,
        mode=mode,
        query_text=query_text or None,
    )

    try:
        with st.status("Scoring your preferences against the song pool...", expanded=False) as status:
            recommendations = get_recommendations(user, songs, k=k)
            status.update(label="Done ranking recommendations.", state="complete")
    except Exception as exc:
        st.error(f"⚠️ Failed to generate recommendations: {exc}")
    else:
        if not recommendations:
            st.info("🤔 No recommendations matched your preferences. Try loosening a filter or the free-text description.")
        else:
            st.divider()
            st.subheader("Recommendations")
            for rank, (song, score, explanation) in enumerate(recommendations, start=1):
                with st.container(border=True):
                    title_col, score_col = st.columns([4, 1])
                    with title_col:
                        badge = RANK_BADGES.get(rank, f"{rank}.")
                        st.markdown(f"**{badge} {song.title}** — {song.artist}")
                        st.caption(f"{song.genre} · {song.mood}")
                    with score_col:
                        st.markdown(f"<div style='text-align:right'><b>{score:.1f}</b></div>", unsafe_allow_html=True)
                    st.progress(min(max(score, 0.0), 100.0) / 100.0)
                    st.caption(explanation)
