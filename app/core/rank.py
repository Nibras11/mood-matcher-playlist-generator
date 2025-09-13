# app/core/rank.py
"""Song ranking and explanation system"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .mood import MoodProfile
from .fetcher import Track

if TYPE_CHECKING:
    # Only for type hints; does NOT import at runtime (prevents cycles)
    from app.services.models import ModelManager, ModelResponse


@dataclass
class RankedTrack:
    track: Track
    score: float  # 0-10
    reason: str   # ≤25 words
    match_factors: List[str]


class PlaylistRanker:
    """Rank and explain track matches for mood."""

    def __init__(self, model_manager: "ModelManager"):
        self.model_manager = model_manager

    def rank_and_explain(
        self, mood_profile: MoodProfile, tracks: List[Track], top_k: int = 10
    ) -> List[RankedTrack]:
        if not tracks:
            return []

        scored = [(t, self._calculate_base_score(mood_profile, t)) for t in tracks]
        scored.sort(key=lambda x: x[1], reverse=True)
        candidates = scored[: min(top_k * 2, len(scored))]

        ranked: List[RankedTrack] = []
        for track, base in candidates:
            try:
                ai_result = self._get_ai_ranking(mood_profile, track, base)
                ranked.append(ai_result or self._create_fallback_ranking(track, base))
            except Exception:
                ranked.append(self._create_fallback_ranking(track, base))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    # ---------- scoring helpers ----------
    def _calculate_base_score(self, mood: MoodProfile, track: Track) -> float:
        score = 5.0
        score += self._match_energy(mood.energy, track)
        score += self._match_valence(mood.valence, track)
        score += self._match_tempo(mood.tempo, track.tempo or "")
        score += self._match_genres(mood.genres, track.genre or "")
        score += self._match_keywords(mood.keywords, track)
        return max(0.0, min(10.0, score))

    def _match_energy(self, mood_energy: str, track: Track) -> float:
        energy = track.features.get("energy", 0.5)
        targets = {"low": 0.3, "medium": 0.6, "high": 0.8}
        diff = abs(energy - targets.get(mood_energy, 0.6))
        return max(0.0, 2.0 * (1 - diff))

    def _match_valence(self, mood_valence: str, track: Track) -> float:
        val = track.features.get("valence", 0.5)
        targets = {"negative": 0.3, "neutral": 0.5, "positive": 0.7}
        diff = abs(val - targets.get(mood_valence, 0.5))
        return max(0.0, 2.0 * (1 - diff))

    def _match_tempo(self, mood_tempo: str, track_tempo: str) -> float:
        mood_seq = ["slow", "medium", "fast"]
        track_seq = ["slow", "mid", "fast"]
        if mood_tempo in mood_seq and track_tempo in track_seq:
            if (mood_tempo == "medium" and track_tempo == "mid") or (mood_tempo == track_tempo):
                return 1.5
            if abs(mood_seq.index(mood_tempo) - track_seq.index(track_tempo)) == 1:
                return 0.5
        return 0.0

    def _match_genres(self, mood_genres: List[str], track_genre: str) -> float:
        if not mood_genres or not track_genre:
            return 0.0
        low = [g.lower() for g in mood_genres]
        if track_genre.lower() in low:
            return 1.0
        families = {
            "rock": ["alternative", "indie", "metal", "punk"],
            "pop": ["dance", "electronic", "disco"],
            "hip-hop": ["rap", "r&b"],
            "jazz": ["blues", "soul", "funk"],
            "folk": ["country", "acoustic"],
        }
        for g in low:
            if track_genre.lower() in families.get(g, []):
                return 0.5
        return 0.0

    def _match_keywords(self, mood_keywords: List[str], track: Track) -> float:
        if not mood_keywords:
            return 0.0
        tag_matches = len(set(k.lower() for k in mood_keywords) & set(t.lower() for t in track.tags))
        title_artist = f"{track.title} {track.artist}".lower()
        title_matches = sum(1 for k in mood_keywords if k.lower() in title_artist)
        total = tag_matches + 0.5 * title_matches
        return min(1.0, total * 0.3)

    # ---------- AI layer ----------
    def _get_ai_ranking(self, mood: MoodProfile, track: Track, base_score: float) -> Optional[RankedTrack]:
        prompt = self._build_prompt(mood, track, base_score)
        resp = self.model_manager.generate_json(prompt, temperature=0.3, max_length=200)  # type: ignore[name-defined]
        if not getattr(resp, "success", False):
            return None
        return self._parse_ai_response(track, getattr(resp, "data", {}), base_score)

    def _build_prompt(self, mood: MoodProfile, track: Track, base_score: float) -> str:
        return f"""Analyze song vs mood. Respond JSON only.

mood: "{mood.raw_text}"
parsed: energy={mood.energy}, valence={mood.valence}, tempo={mood.tempo}
keywords: {', '.join(mood.keywords[:5])}

song: "{track.title}" by {track.artist}
genre={track.genre}, decade={track.decade}, tempo={track.tempo}
tags: {', '.join(track.tags[:3])}
features: energy={track.features.get('energy', 0):.1f}, valence={track.features.get('valence', 0):.1f}

base_score: {base_score:.1f}

return:
{{
  "score": <0-10 float>,
  "reason": "<<=25 words>",
  "factors": ["<factor1>", "<factor2>"]
}}"""

    def _parse_ai_response(self, track: Track, data: Dict[str, Any], base_score: float) -> RankedTrack:
        try:
            score = float(data.get("score", base_score))
            score = max(0.0, min(10.0, score))
            reason = str(data.get("reason", "Good match for your mood"))
            if len(reason.split()) > 25:
                reason = " ".join(reason.split()[:25])
            factors = data.get("factors", [])
            if not isinstance(factors, list):
                factors = ["mood compatibility"]
            factors = [str(f) for f in factors[:3]]
            return RankedTrack(track=track, score=score, reason=reason, match_factors=factors)
        except Exception:
            return self._create_fallback_ranking(track, base_score)

    def _create_fallback_ranking(self, track: Track, base_score: float) -> RankedTrack:
        return RankedTrack(
            track=track,
            score=base_score,
            reason="Matches your mood based on musical features",
            match_factors=["tempo", "energy", "style"],
        )

    def validate_rankings(self, ranked: List[RankedTrack]) -> List[RankedTrack]:
        out: List[RankedTrack] = []
        for r in ranked:
            s = max(0.0, min(10.0, float(r.score)))
            words = r.reason.split()
            reason = " ".join(words[:25]) if len(words) > 25 else r.reason or "Good match for your mood"
            factors = r.match_factors or ["compatibility"]
            out.append(RankedTrack(track=r.track, score=s, reason=reason, match_factors=factors[:3]))
        return out


__all__ = ["RankedTrack", "PlaylistRanker"]