# app/Main.py
"""AI Playlist Mood Matcher - Streamlit App"""

import os
from typing import Dict, List

from dotenv import load_dotenv
import streamlit as st

# Import relative to the app/ folder (run from repo root: `streamlit run app/Main.py`)
from core.mood import MoodParser, MoodProfile
from core.fetcher import SpotifyDatasetFetcher, Track
from core.rank import PlaylistRanker, RankedTrack
from services.models import ModelManager, ModelClientFactory


# Load environment variables
load_dotenv()

# App configuration
APP_TITLE = os.getenv("APP_TITLE", "AI Playlist Mood Matcher")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


@st.cache_resource
def get_components():
    """Initialize and cache app components"""
    try:
        # Initialize model clients with fallback (both optional)
        primary_client = None
        fallback_client = None

        try:
            primary_client = ModelClientFactory.create_client(
                "huggingface",
                model_name=os.getenv("HF_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3"),
            )
        except Exception as e:
            st.warning(f"HuggingFace client failed: {e}")
            primary_client = None

        try:
            fallback_client = ModelClientFactory.create_client(
                "ollama",
                model_name=os.getenv("OLLAMA_MODEL_NAME", "llama3"),
            )
        except Exception as e:
            if DEBUG:
                st.info(f"Ollama client not available: {e}")
            fallback_client = None

        model_manager = None
        if primary_client:
            model_manager = ModelManager(primary_client, fallback_client)
        elif fallback_client:
            model_manager = ModelManager(fallback_client)

        mood_parser = MoodParser()
        fetcher = SpotifyDatasetFetcher()
        ranker = PlaylistRanker(model_manager) if model_manager else None

        return mood_parser, fetcher, ranker
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None


@st.cache_data
def load_sample_tracks(n_samples: int = 200, seed: int = 42) -> List[Track]:
    """Load and cache sample tracks"""
    try:
        _, fetcher, _ = get_components()
        if fetcher:
            with st.spinner("Loading music dataset..."):
                tracks = fetcher.sample_tracks(n_samples=n_samples, seed=seed)
                if tracks and len(tracks) <= 5:
                    st.info("Loaded a small offline fallback set. Connect to the internet for the full dataset.")
                return tracks
        return []
    except Exception as e:
        st.error(f"Failed to load tracks: {e}")
        return []


def _search_links(title: str, artist: str) -> Dict[str, str]:
    """Build clickable search links (no API keys needed)."""
    t = (title or "").strip()
    a = (artist or "").strip()
    spotify_url = f"https://open.spotify.com/search/{t.replace(' ', '%20')}"
    yt_query = f"{t} {a}".strip().replace(" ", "+")
    youtube_url = f"https://www.youtube.com/results?search_query={yt_query}"
    apple_url = f"https://music.apple.com/search?term={yt_query}"
    return {"spotify": spotify_url, "youtube": youtube_url, "apple": apple_url}


def display_playlist(ranked_tracks: List[RankedTrack]):
    """Display ranked playlist with explanations (robust to None/empty fields)."""
    if not ranked_tracks:
        st.warning("No matching songs found. Try a different mood description.")
        return

    st.subheader(f"🎵 Your Personalized Playlist ({len(ranked_tracks)} songs)")

    # Topline metrics
    try:
        avg_score = sum(float(rt.score or 0.0) for rt in ranked_tracks) / max(1, len(ranked_tracks))
    except Exception:
        avg_score = 0.0

    # Safe top genres
    top_genres = [g for g in {getattr(rt.track, "genre", None) for rt in ranked_tracks[:5]} if g]
    top_genres_display = ", ".join(top_genres[:2]) if top_genres else "N/A"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Songs Found", len(ranked_tracks))
    with col2:
        st.metric("Avg Match Score", f"{avg_score:.1f}/10")
    with col3:
        st.metric("Top Genres", top_genres_display)

    st.divider()

    # Per-track cards
    for i, ranked_track in enumerate(ranked_tracks, 1):
        t = ranked_track.track
        title = (getattr(t, "title", "") or "Unknown Title").strip()
        artist = (getattr(t, "artist", "") or "Unknown Artist").strip()
        genre = (getattr(t, "genre", "") or "").strip()
        decade = (getattr(t, "decade", "") or "").strip()
        tempo = (getattr(t, "tempo", "") or "").strip()
        tags = getattr(t, "tags", []) or []
        reason = (getattr(ranked_track, "reason", "") or "Good match for your mood").strip()
        match_factors = getattr(ranked_track, "match_factors", []) or []
        score_val = float(getattr(ranked_track, "score", 0.0) or 0.0)

        with st.container():
            c1, c2, c3 = st.columns([0.1, 0.6, 0.3])

            with c1:
                if score_val >= 8:
                    emoji = "🥇"
                elif score_val >= 7:
                    emoji = "🥈"
                elif score_val >= 6:
                    emoji = "🥉"
                else:
                    emoji = "🎵"
                st.markdown(f"**{i}.** {emoji}")

            with c2:
                st.markdown(f"**{title}**")
                st.markdown(f"*by {artist}*")

                details = []
                if genre:
                    details.append(f"Genre: {genre.title()}")
                if decade:
                    details.append(f"Era: {decade}")
                if tempo:
                    tempo_disp = "Medium" if tempo.lower() in {"mid", "medium"} else tempo.title()
                    details.append(f"Tempo: {tempo_disp}")
                if details:
                    st.caption(" • ".join(details))

                tag_tokens = [f"`{str(tag).strip()}`" for tag in tags if isinstance(tag, str) and tag.strip()]
                if tag_tokens:
                    st.caption("Tags: " + " ".join(tag_tokens[:4]))

            with c3:
                if score_val >= 8:
                    score_color = "#28a745"
                elif score_val >= 6:
                    score_color = "#ffc107"
                else:
                    score_color = "#dc3545"

                st.markdown(
                    f"<div style='text-align:center; padding:10px; background-color:{score_color}; "
                    f"color:white; border-radius:5px; margin-bottom:10px;'>"
                    f"<strong>{score_val:.1f}/10</strong></div>",
                    unsafe_allow_html=True,
                )

                st.caption("💭 " + (reason if len(reason) <= 220 else reason[:217] + "…"))

                if match_factors:
                    factors = [str(f).strip() for f in match_factors if isinstance(f, str) and f.strip()]
                    if factors:
                        st.caption("🎯 **Match:** " + " • ".join(factors[:4]))

                links = _search_links(title, artist)
                with st.popover("🔗 Listen"):
                    st.markdown(f"**🎧 Listen to '{title}'**")
                    st.markdown(f"🟢 [Spotify]({links['spotify']})")
                    st.markdown(f"▶️ [YouTube]({links['youtube']})")
                    st.markdown(f"🎵 [Apple Music]({links['apple']})")

        if i < len(ranked_tracks):
            st.divider()


def create_mock_ranking(mood_profile: MoodProfile, tracks: List[Track]) -> List[RankedTrack]:
    """Create mock ranking when AI is not available (simple rule-based)."""
    ranked_tracks: List[RankedTrack] = []
    import random

    for track in tracks[:15]:
        score = 5.0

        if mood_profile.genres and track.genre:
            if any(g.lower() in (track.genre or "").lower() for g in mood_profile.genres):
                score += 2.0

        track_energy = track.features.get("energy", 0.5)
        if mood_profile.energy == "high" and track_energy > 0.7:
            score += 1.5
        elif mood_profile.energy == "low" and track_energy < 0.4:
            score += 1.5
        elif mood_profile.energy == "medium" and 0.4 <= track_energy <= 0.7:
            score += 1.0

        track_valence = track.features.get("valence", 0.5)
        if mood_profile.valence == "positive" and track_valence > 0.6:
            score += 1.5
        elif mood_profile.valence == "negative" and track_valence < 0.4:
            score += 1.5

        score += random.uniform(-0.5, 0.5)
        score = max(1.0, min(10.0, score))

        reasons = [
            f"Good {mood_profile.energy} energy match",
            f"Matches your {mood_profile.valence} mood",
            f"Great {(track.genre or 'music')} vibes",
            "Tempo fits your vibe",
            "Recommended based on features",
        ]
        reason = random.choice(reasons)
        factors = ["energy", "mood", "genre"][: random.randint(1, 3)]

        ranked_tracks.append(RankedTrack(track=track, score=score, reason=reason, match_factors=factors))

    ranked_tracks.sort(key=lambda x: x.score, reverse=True)
    return ranked_tracks[:10]


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Header
    st.title(APP_TITLE)
    st.markdown("**Match your mood to the perfect playlist with AI** 🎯")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
        This app analyzes your mood description and finds matching songs from a curated dataset.

        **How it works:**
        1. Describe your current mood  
        2. AI parses your emotions & preferences  
        3. Songs are ranked by compatibility  
        4. Get personalized explanations  
        """
        )

        st.header("Tips")
        st.markdown(
            """
        - Be specific about your mood  
        - Mention genres you like  
        - Include energy levels (calm, energetic)  
        - Describe the vibe you want  
        """
        )

        if DEBUG:
            st.header("Debug Info")
            st.caption("Debug mode is enabled")

    # Load components
    mood_parser, fetcher, ranker = get_components()
    if not mood_parser:
        st.error("Failed to initialize the app. Please refresh the page.")
        st.stop()

    # --------------------------
    # Mood input + examples (safe with callback)
    # --------------------------
    if "mood_text" not in st.session_state:
        st.session_state.mood_text = ""
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = ""

    def _apply_example():
        choice = st.session_state.get("example_choice", "")
        if choice:
            # This runs as a callback BEFORE the next render,
            # so it's safe to mutate the textarea's key.
            st.session_state["mood_text"] = choice

    col1, col2 = st.columns([2, 1])

    with col1:
        st.text_area(
            "🎭 Describe your current mood:",
            key="mood_text",
            placeholder=(
                "I'm feeling nostalgic and want something upbeat but not too energetic. "
                "Maybe some indie rock from the 2000s that makes me feel hopeful..."
            ),
            height=120,
        )

    with col2:
        st.markdown("### 💡 Example Moods")
        example_moods = [
            "Energetic workout motivation",
            "Chill evening relaxation",
            "Nostalgic 90s vibes",
            "Happy upbeat pop songs",
            "Melancholy rainy day mood",
            "Focus music for studying",
        ]

        st.selectbox(
            "Or choose an example:",
            [""] + example_moods,
            key="example_choice",
        )

        # ✅ Use a callback instead of direct assignment
        st.button("Use This Mood", type="secondary", on_click=_apply_example)

    # --------------------------
    # Playlist generation
    # --------------------------
    if st.button("🎵 Find My Playlist", type="primary", use_container_width=True):
        mood_input = st.session_state.get("mood_text", "").strip()
        if not mood_input:
            st.warning("Please describe your mood first!")
            st.stop()

        # Parse mood
        mood_profile = mood_parser.parse(mood_input)

        # Show parsed mood info
        with st.expander("🔍 Mood Analysis", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Energy Level", mood_profile.energy.title())
            with c2:
                st.metric("Emotional Tone", mood_profile.valence.title())
            with c3:
                tempo_disp = "Medium" if mood_profile.tempo.lower() in {"mid", "medium"} else mood_profile.tempo.title()
                st.metric("Preferred Tempo", tempo_disp)

            if mood_profile.genres:
                st.markdown(f"**Detected Genres:** {', '.join(mood_profile.genres)}")

            if mood_profile.keywords:
                st.markdown(f"**Key Moods:** {', '.join(mood_profile.keywords[:8])}")

        # Load tracks
        tracks = load_sample_tracks()
        if not tracks:
            st.error("Failed to load music dataset. Please try again.")
            st.stop()

        # Generate playlist
        with st.spinner("🤖 AI is analyzing your mood and finding perfect songs..."):
            if ranker:
                try:
                    ranked_tracks = ranker.rank_and_explain(mood_profile, tracks, top_k=10)
                    ranked_tracks = ranker.validate_rankings(ranked_tracks)
                except Exception as e:
                    st.warning(f"AI ranking failed: {e}. Using fallback ranking.")
                    ranked_tracks = create_mock_ranking(mood_profile, tracks)
            else:
                st.info("Using demo mode (AI services not available)")
                ranked_tracks = create_mock_ranking(mood_profile, tracks)

        # Display results
        if ranked_tracks:
            display_playlist(ranked_tracks)

            # Export
            with st.expander("💾 Export Playlist"):
                playlist_text = f"# {APP_TITLE} - Generated Playlist\n\n"
                playlist_text += f"**Mood:** {mood_input}\n\n"
                for i, rt in enumerate(ranked_tracks, 1):
                    playlist_text += f"{i}. **{rt.track.title}** by {rt.track.artist}\n"
                    playlist_text += f"   Score: {rt.score:.1f}/10 - {rt.reason}\n\n"

                st.download_button(
                    label="📄 Download as Text",
                    data=playlist_text,
                    file_name="my_playlist.txt",
                    mime="text/plain",
                )
        else:
            st.error("No matching songs found. Try a different mood description.")

    # Footer
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Built with ❤️ using Streamlit")
    with c2:
        st.caption("Powered by AI & Spotify Data")
    with c3:
        st.caption(f"v{os.getenv('APP_VERSION', '1.0.0')}")


if __name__ == "__main__":
    main()