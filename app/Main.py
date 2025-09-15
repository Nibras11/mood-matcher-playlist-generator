"""AI Playlist Mood Matcher - Streamlit App (with RAG)"""

import os
import streamlit as st
from dotenv import load_dotenv

# Core modules
from app.core.mood import MoodParser
from app.core.rank import PlaylistRanker
from app.core.fetcher import Track
from app.services.rag import RAGRetriever
from app.services.models import ModelClientFactory, ModelManager

# Load env
load_dotenv()
APP_TITLE = os.getenv("APP_TITLE", "AI Playlist Mood Matcher")


# ---------- Cached components ----------
@st.cache_resource
def get_components():
    """Initialize and cache model + retriever + ranker"""
    try:
        # AI clients
        primary_client = None
        try:
            primary_client = ModelClientFactory.create_client(
                "huggingface", model_name="microsoft/DialoGPT-medium"
            )
        except Exception as e:
            st.warning(f"HF client not available: {e}")

        fallback_client = None
        try:
            fallback_client = ModelClientFactory.create_client("ollama", model_name="llama2")
        except Exception:
            pass

        model_manager = None
        if primary_client:
            model_manager = ModelManager(primary_client, fallback_client)
        elif fallback_client:
            model_manager = ModelManager(fallback_client)

        mood_parser = MoodParser()
        retriever = RAGRetriever("maharshipandya/spotify-tracks-dataset")
        ranker = PlaylistRanker(model_manager) if model_manager else None

        return mood_parser, retriever, ranker
    except Exception as e:
        st.error(f"Init failed: {e}")
        return None, None, None


# ---------- UI ----------
def display_playlist(ranked):
    if not ranked:
        st.warning("No matches found.")
        return

    st.subheader(f"🎵 Your Playlist ({len(ranked)} songs)")
    avg_score = sum(r.score for r in ranked) / len(ranked)
    st.metric("Avg Match Score", f"{avg_score:.1f}/10")

    for i, r in enumerate(ranked, 1):
        with st.container():
            st.markdown(f"**{i}. {r.track.title}** by *{r.track.artist}*")
            st.caption(f"Score: {r.score:.1f}/10 — {r.reason}")

            # Clickable links (mock Spotify)
            query = f"{r.track.title} {r.track.artist}".replace(" ", "+")
            link = f"https://open.spotify.com/search/{query}"
            st.markdown(f"[🎧 Listen on Spotify]({link})")

        st.divider()


# ---------- Main ----------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎵", layout="wide")
    st.title(APP_TITLE)
    st.caption("Match your mood to the perfect playlist with AI (RAG powered)")

    mood_parser, retriever, ranker = get_components()
    if not (mood_parser and retriever):
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        mood_input = st.text_area(
            "🎭 Describe your mood:",
            placeholder="I'm feeling nostalgic and want some upbeat indie rock...",
            key="mood_text",
            height=100,
        )

    with col2:
        st.markdown("### 💡 Examples")
        examples = [
            "Energetic workout motivation",
            "Chill evening relaxation",
            "Nostalgic 90s vibes",
            "Happy upbeat pop songs",
            "Melancholy rainy day",
        ]
        selected = st.selectbox("Choose example:", [""] + examples)
        if selected and st.button("Use this mood"):
            mood_input = selected

    if st.button("🎵 Find My Playlist", type="primary", use_container_width=True):
        if not mood_input.strip():
            st.warning("Please describe your mood first!")
            st.stop()

        # Parse mood
        mood_profile = mood_parser.parse(mood_input)
        with st.expander("🔍 Mood Analysis"):
            st.json(mood_profile.__dict__)

        # Retrieve
        with st.spinner("🔎 Retrieving candidate songs..."):
            candidates: list[Track] = retriever.search(mood_input, top_k=30)

        if not candidates:
            st.error("No songs retrieved. Try different mood.")
            st.stop()

        # Rank
        with st.spinner("🤖 Ranking with AI..."):
            if ranker:
                ranked = ranker.rank_and_explain(mood_profile, candidates, top_k=10)
            else:
                ranked = [r for r in candidates[:10]]

        display_playlist(ranked)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit + HuggingFace + RAG")


if __name__ == "__main__":
    main()
