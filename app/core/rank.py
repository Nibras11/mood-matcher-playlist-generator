"""Ranking + AI explanations for retrieved tracks"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.core.mood import MoodProfile
from app.core.fetcher import Track
from app.services.models import ModelManager, ModelResponse


@dataclass
class RankedTrack:
    track: Track
    score: float
    reason: str
    match_factors: List[str]


class PlaylistRanker:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def rank_and_explain(self, mood: MoodProfile, tracks: List[Track], top_k: int = 10) -> List[RankedTrack]:
        ranked = []
        for t in tracks:
            base_score = self._calculate_score(mood, t)
            ai_ranking = self._get_ai_ranking(mood, t, base_score)
            ranked.append(ai_ranking or self._fallback(t, base_score))

        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:top_k]

    def _calculate_score(self, mood: MoodProfile, track: Track) -> float:
        score = 5.0
        if mood.energy == "high" and track.features.get("energy", 0.5) > 0.7:
            score += 1
        if mood.valence == "positive" and track.features.get("valence", 0.5) > 0.6:
            score += 1
        if mood.tempo == track.tempo:
            score += 1
        if mood.genres and track.genre and track.genre.lower() in [g.lower() for g in mood.genres]:
            score += 1
        return max(0, min(10, score))

    def _get_ai_ranking(self, mood: MoodProfile, track: Track, base: float) -> Optional[RankedTrack]:
        prompt = f"""
        User mood: {mood.raw_text}
        Song: {track.title} by {track.artist}, genre={track.genre}, tempo={track.tempo}
        Base score: {base:.1f}

        Respond JSON only:
        {{
          "score": <float 0-10>,
          "reason": "<short explanation>",
          "factors": ["energy", "genre"]
        }}
        """
        resp: ModelResponse = self.model_manager.generate_json(prompt, temperature=0.3)
        if not resp.success:
            return None

        data = resp.data
        return RankedTrack(
            track=track,
            score=float(data.get("score", base)),
            reason=str(data.get("reason", "Matches your mood")),
            match_factors=data.get("factors", ["compatibility"])
        )

    def _fallback(self, track: Track, base: float) -> RankedTrack:
        return RankedTrack(track=track, score=base, reason="Feature-based match", match_factors=["tempo", "energy"])
